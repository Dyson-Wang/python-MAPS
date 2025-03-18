import torch
import torch.nn.functional as F
from torch.distributions import Normal, Categorical
from collections import deque

from memory import HPPOBuffer
from model import Actor, Critic

class HPPO:
    def __init__(self, num_users, num_states, num_channels, lr_a, lr_c, pmax, gamma, lam, 
                 repeat_time, batch_size, eps_clip, w_entropy, seq_len=5):
        # 初始化Actor网络（带LSTM）
        self.actors = [Actor(num_states, 18, num_channels, pmax, lstm_hidden=128).cuda() 
                      for _ in range(num_users)]
        
        # 初始化Critic网络（带LSTM）
        self.critic = Critic(num_states, lstm_hidden=128).cuda()

        # 优化器配置
        self.optimizer_a = torch.optim.Adam(
            [{"params": actor.parameters(), "lr": lr_a} for actor in self.actors]
        )
        self.optimizer_c = torch.optim.Adam(self.critic.parameters(), lr=lr_c)

        # 经验回放缓冲区
        self.buffer = HPPOBuffer(num_users, seq_len=seq_len)
        
        # 超参数设置
        self.pmax = pmax
        self.gamma = gamma
        self.lam = lam
        self.repeat_time = repeat_time
        self.batch_size = batch_size
        self.eps_clip = eps_clip
        self.w_entropy = w_entropy
        self.seq_len = seq_len
        
        # 状态历史缓存
        self.state_history = deque(maxlen=seq_len)

    def select_action(self, state, test=False):
        with torch.no_grad():
            # 维护状态序列
            self._update_state_history(state)
            
            # 构建当前状态序列
            state_seq = self._get_current_sequence().unsqueeze(0).cuda()  # (1, seq_len, state_dim)
            
            actions = []
            for i, actor in enumerate(self.actors):
                # 初始化隐藏状态
                if test or not hasattr(actor, 'hidden') or actor.hidden is None:
                    actor.hidden = actor.init_hidden(batch_size=1)
                
                # LSTM前向传播
                prob_points, prob_channels, (power_mu, power_sigma), new_hidden = actor(
                    state_seq, 
                    actor.hidden
                )
                
                # 更新隐藏状态
                actor.hidden = (new_hidden[0].detach(), new_hidden[1].detach())
                
                # 动作采样
                dist_point = Categorical(prob_points)
                point = dist_point.sample()
                dist_channel = Categorical(prob_channels)
                channel = dist_channel.sample()
                dist_power = Normal(power_mu, power_sigma)
                power = dist_power.sample().clamp(1e-10, self.pmax)

                if not test:
                    # 存储动作信息
                    self.buffer.actor_buffer[i].actions_point.append(point.item())
                    self.buffer.actor_buffer[i].actions_channel.append(channel.item())
                    self.buffer.actor_buffer[i].actions_power.append(power.item())
                    
                    self.buffer.actor_buffer[i].logprobs_point.append(dist_point.log_prob(point))
                    self.buffer.actor_buffer[i].logprobs_channel.append(dist_channel.log_prob(channel))
                    self.buffer.actor_buffer[i].logprobs_power.append(dist_power.log_prob(power))

                actions.append((point.item(), channel.item(), power.item()))
                
                # 记录网络信息
                self.buffer.actor_buffer[i].info.update({
                    'prob_points': prob_points.detach().cpu().numpy(),
                    'prob_channels': prob_channels.detach().cpu().numpy(),
                    'power_mu': power_mu.item(),
                    'power_sigma': power_sigma.item()
                })
            
            return actions

    def update(self):
        # 构建序列化经验数据
        self.buffer.build_tensor()
        
        # 计算GAE
        with torch.no_grad():
            values, _ = self.critic(self.buffer.states_tensor)
            values = values.squeeze()
            targets, advantages = self._compute_gae(values)
        
        # 策略更新
        for _ in range(int(self.repeat_time * (len(self.buffer) / self.batch_size))):
            indices = torch.randint(len(self.buffer), (self.batch_size,)).cuda()
            
            # 获取批次数据
            batch_states = self.buffer.states_tensor[indices]
            batch_targets = targets[indices]
            batch_advantages = advantages[indices]
            
            # Critic更新
            self._update_critic(batch_states, batch_targets)
            
            # Actor更新
            self._update_actors(batch_states, batch_advantages, indices)
        
        # 清空缓冲区
        self.buffer.init()

    def _update_critic(self, states, targets):
        pred_values, _ = self.critic(states)
        loss_c = F.mse_loss(pred_values.squeeze(), targets)
        
        self.optimizer_c.zero_grad()
        loss_c.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.optimizer_c.step()

    def _update_actors(self, states, advantages, indices):
        loss_components = []
        
        for i, actor in enumerate(self.actors):
            # 获取旧动作的概率
            old_logprob_p = self.buffer.actor_buffer[i].logprobs_point_tensor[indices]
            old_logprob_c = self.buffer.actor_buffer[i].logprobs_channel_tensor[indices]
            old_logprob_pow = self.buffer.actor_buffer[i].logprobs_power_tensor[indices]
            
            # 评估新策略
            new_logprob_p, new_logprob_c, new_logprob_pow, entropy_p, entropy_c, entropy_pow = self._evaluate(
                actor, 
                states,
                self.buffer.actor_buffer[i].actions_point_tensor[indices],
                self.buffer.actor_buffer[i].actions_channel_tensor[indices],
                self.buffer.actor_buffer[i].actions_power_tensor[indices]
            )
            
            # 计算各项损失
            loss_p = self._compute_loss(new_logprob_p, old_logprob_p, advantages, entropy_p)
            loss_c = self._compute_loss(new_logprob_c, old_logprob_c, advantages, entropy_c)
            loss_pow = self._compute_loss(new_logprob_pow, old_logprob_pow, advantages, entropy_pow)
            
            loss_components.extend([loss_p, loss_c, loss_pow])
        
        # 合并损失并更新
        total_loss = torch.mean(torch.stack(loss_components))
        self.optimizer_a.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5) 
        for actor in self.actors:
            self.optimizer_a.step()

    def _compute_gae(self, values):
        targets = torch.zeros_like(values)
        advantages = torch.zeros_like(values)
        next_value = 0
        next_advantage = 0
        
        for t in reversed(range(len(self.buffer))):
            mask = 1 - self.buffer.is_terminals_tensor[t]
            delta = self.buffer.rewards_tensor[t] + self.gamma * next_value * mask - values[t]
            advantages[t] = delta + self.gamma * self.lam * next_advantage * mask
            targets[t] = values[t] + advantages[t]
            
            next_value = values[t]
            next_advantage = advantages[t]
        
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return targets, advantages

    def _evaluate(self, actor, states, act_p, act_c, act_pow):
        prob_points, prob_channels, (power_mu, power_sigma), _ = actor(states)
        
        dist_point = Categorical(prob_points)
        dist_channel = Categorical(prob_channels)
        dist_power = Normal(power_mu, power_sigma)
        
        logprob_p = dist_point.log_prob(act_p.long())
        logprob_c = dist_channel.log_prob(act_c.long())
        logprob_pow = dist_power.log_prob(act_pow)
        
        entropy_p = dist_point.entropy()
        entropy_c = dist_channel.entropy()
        entropy_pow = dist_power.entropy()
        
        return logprob_p, logprob_c, logprob_pow, entropy_p, entropy_c, entropy_pow

    def _compute_loss(self, new_logprob, old_logprob, advantages, entropy):
        ratio = (new_logprob - old_logprob).exp()
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-self.eps_clip, 1+self.eps_clip) * advantages
        return -torch.min(surr1, surr2).mean() - self.w_entropy * entropy.mean()

    def _update_state_history(self, state):
        """维护状态序列缓存"""
        self.state_history.append(torch.FloatTensor(state))
        if len(self.state_history) < self.seq_len:
            padding = [torch.zeros_like(self.state_history[0])] * (self.seq_len - len(self.state_history))
            self.state_history.extendleft(padding)

    def _get_current_sequence(self):
        """获取当前状态序列"""
        if len(self.state_history) < self.seq_len:
            padding = [torch.zeros_like(self.state_history[0])] * (self.seq_len - len(self.state_history))
            return torch.stack(padding + list(self.state_history))
        return torch.stack(list(self.state_history))

    def save_model(self, filename, args, info=None):
        save_dict = {
            'actors': [actor.state_dict() for actor in self.actors],
            'critic': self.critic.state_dict(),
            'args': args,
            'info': info
        }
        torch.save(save_dict, filename)

    def load_model(self, checkpoint):
        for actor, state_dict in zip(self.actors, checkpoint['actors']):
            actor.load_state_dict(state_dict)
        self.critic.load_state_dict(checkpoint['critic'])