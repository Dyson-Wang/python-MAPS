import torch
from collections import deque

class ActorBuffer:
    def __init__(self):
        self.actions_point = []
        self.logprobs_point = []
        self.actions_channel = []
        self.logprobs_channel = []
        self.actions_power = []
        self.logprobs_power = []
        self.info = {}

    def build_tensor(self):
        # 离散动作转换为long类型
        self.actions_point_tensor = torch.tensor(self.actions_point, dtype=torch.long).cuda()
        self.logprobs_point_tensor = torch.tensor(self.logprobs_point, dtype=torch.float32).cuda()
        self.actions_channel_tensor = torch.tensor(self.actions_channel, dtype=torch.long).cuda()
        self.logprobs_channel_tensor = torch.tensor(self.logprobs_channel, dtype=torch.float32).cuda()
        
        # 连续功率保持float类型
        self.actions_power_tensor = torch.tensor(self.actions_power, dtype=torch.float32).cuda()
        self.logprobs_power_tensor = torch.tensor(self.logprobs_power, dtype=torch.float32).cuda()

    def init(self):
        # 清空缓存并安全释放显存
        self.actions_point.clear()
        self.logprobs_point.clear()
        self.actions_channel.clear()
        self.logprobs_channel.clear()
        self.actions_power.clear()
        self.logprobs_power.clear()
        
        # 安全删除张量引用
        tensor_attrs = [
            'actions_point_tensor', 'logprobs_point_tensor',
            'actions_channel_tensor', 'logprobs_channel_tensor',
            'actions_power_tensor', 'logprobs_power_tensor'
        ]
        for attr in tensor_attrs:
            if hasattr(self, attr):
                delattr(self, attr)

class HPPOBuffer:
    def __init__(self, num_actors, seq_len=5):
        self.seq_len = seq_len  # LSTM序列长度
        self.states = deque(maxlen=seq_len*2)  # 保留足够的历史状态
        self.rewards = []
        self.is_terminals = []
        self.actor_buffer = [ActorBuffer() for _ in range(num_actors)]
        self.info = {}

    def build_tensor(self):
        """构建LSTM需要的序列化张量"""
        # 生成状态序列 (batch_size, seq_len, state_dim)
        state_sequences = []
        for i in range(len(self.rewards)):
            start = max(0, i - self.seq_len + 1)
            seq = list(self.states)[start:i+1]
            
            # 前向填充
            if len(seq) < self.seq_len:
                padding = [torch.zeros_like(seq[0])] * (self.seq_len - len(seq))
                seq = padding + seq
            
            # 截取最近seq_len个状态
            seq = seq[-self.seq_len:]
            state_sequences.append(torch.stack(seq))

        self.states_tensor = torch.stack(state_sequences).cuda()
        self.rewards_tensor = torch.tensor(self.rewards, dtype=torch.float32).cuda()
        self.is_terminals_tensor = torch.tensor(self.is_terminals, dtype=torch.float32).cuda()
        
        # 构建各Actor的缓冲张量
        for actor_buffer in self.actor_buffer:
            actor_buffer.build_tensor()

    def init(self):
        """重置缓冲区"""
        self.states.clear()
        self.rewards.clear()
        self.is_terminals.clear()
        
        # 安全删除张量引用
        tensor_attrs = ['states_tensor', 'rewards_tensor', 'is_terminals_tensor']
        for attr in tensor_attrs:
            if hasattr(self, attr):
                delattr(self, attr)
        
        for actor_buffer in self.actor_buffer:
            actor_buffer.init()

    def __len__(self):
        return len(self.rewards)

    def add_experience(self, state, reward, done, actions_info):
        """添加新的经验"""
        # 状态存储为张量
        self.states.append(torch.tensor(state, dtype=torch.float32))
        self.rewards.append(reward)
        self.is_terminals.append(done)
        
        # 存储各智能体的动作信息
        for i, actor_info in enumerate(actions_info):
            point, channel, power, logprob_p, logprob_c, logprob_pow = actor_info
            self.actor_buffer[i].actions_point.append(point)
            self.actor_buffer[i].actions_channel.append(channel)
            self.actor_buffer[i].actions_power.append(power)
            self.actor_buffer[i].logprobs_point.append(logprob_p)
            self.actor_buffer[i].logprobs_channel.append(logprob_c)
            self.actor_buffer[i].logprobs_power.append(logprob_pow)