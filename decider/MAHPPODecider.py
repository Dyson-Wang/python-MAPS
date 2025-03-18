from .SplitDecision import *
import random
import numpy as np
import torch
from copy import deepcopy

class HPPODecider(SplitDecision):
    def __init__(self, num_users, num_states, num_channels, lr_a, lr_c, pmax, gamma, lam, repeat_time, batch_size,
                 eps_clip, w_entropy, train=False):
        super().__init__()
        
        # 初始化HPPO的相关参数
        self.hppo = HPPO(num_users, num_states, num_channels, lr_a, lr_c, pmax, gamma, lam, repeat_time, batch_size,
                         eps_clip, w_entropy)
        self.train = train
        self.workflowids_checked = []
        self.layer_intervals = np.zeros(num_users)  # 保存每层的划分间隔
        self.average_layer_intervals = np.zeros(num_users)  # 用于计算每个用户的平均划分间隔
        self.epsilon = 0.95  # epsilon-greedy策略
        self.r_thresh = 0.45  # 奖励阈值
        
        random.seed(1)

    def updateAverages(self):
        # 更新每个用户的平均划分间隔
        for WorkflowID in self.env.destroyedworkflows:
            if WorkflowID not in self.workflowids_checked:
                dict_ = self.env.destroyedworkflows[WorkflowID]
                decision = dict_['application'].split('/')[1].split('_')[1]
                decision = 0 if decision == self.choices[0] else 1
                intervals = dict_['destroyAt'] - dict_['createAt']
                self.average_layer_intervals[WorkflowID] = 0.1 * intervals + 0.9 * self.average_layer_intervals[WorkflowID]

    def updateRewards(self):
        rewards = []
        for WorkflowID in self.env.destroyedworkflows:
            if WorkflowID not in self.workflowids_checked:
                self.workflowids_checked.append(WorkflowID)
                dict_ = self.env.destroyedworkflows[WorkflowID]
                sla = dict_['sla']
                intervals = dict_['destroyAt'] - dict_['createAt']
                sla_reward = 1 if intervals <= sla else 0
                acc_reward = max(0, (dict_['result'][0] / dict_['result'][1] - 0.9)) * 10
                reward = Coeff_SLA * sla_reward + Coeff_Acc * acc_reward
                rewards.append(reward)
        return sum(rewards) / (len(rewards) + 1e-5)

    def decision(self, workflowlist):
        # 根据工作流列表进行决策，决定在哪一层进行划分
        self.updateAverages()
        avg_reward = self.updateRewards()
        decisions = []

        for _, _, sla, workflow in workflowlist:
            if self.train and random.random() < self.epsilon:
                # 探索：随机选择一个划分层次
                decisions.append(random.choice(self.choices))
            else:
                # 利用：选择当前已知的最优划分层次
                low = sla < self.average_layer_intervals[workflow.lower()]
                if low:
                    decisions.append(self.choices[0])  # 低回报情况下选择第一个划分点
                else:
                    decisions.append(self.choices[1])  # 高回报情况下选择第二个划分点

        # 根据奖励值调整epsilon和阈值
        if avg_reward >= self.r_thresh:
            self.epsilon *= 0.98
            self.r_thresh = min(1, 1.05 * self.r_thresh)

        return decisions
