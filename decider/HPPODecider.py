from .SplitDecision import SplitDecision
from .agent.hppo import HPPO
import torch
import numpy as np
import pickle
from os import path, mkdir
from copy import deepcopy
from torch.distributions import Normal, Categorical

class HPPODecider(SplitDecision):
    def __init__(self, num_layers=4, train=False, agent_params=None):
        super().__init__()
        # self.applications = ['mnist', 'fashionmnist', 'cifar100']
        # layer_intervals = [5, 8, 15] # 用于不同应用程序的层间隔配置
        # self.average_layer_intervals = dict(zip(self.applications, layer_intervals))
        # self.workflowids_checked = []
        # self.epsilon = 0.95 # 以 epsilon 概率进行随机选择（探索）以 1 - epsilon 概率选择当前回报最高的动作（开发）
        # self.r_thresh = 0.45 # r_thresh 是一个阈值参数，用于区分“低回报”和“高回报”
        # self.low_rewards, self.low_counts = np.zeros(2), np.zeros(2)
        # self.high_rewards, self.high_counts = np.zeros(2), np.zeros(2)
        # self.train = train
        # random.seed(1)
        # self.load_model()

        self.num_layers = num_layers  # 模型的总层数，决定划分点范围
        self.train = train
        
        # 初始化HPPO智能体（调整参数适配层划分决策）
        default_params = {
            'num_users': 10,  # 假设单个用户决策
            'num_states': 4,  # 状态维度（根据特征设计）
            'num_channels': 0,  # 不需要信道参数
            'lr_a': 1e-4,
            'lr_c': 1e-4,
            'pmax': 1.0,
            'gamma': 0.95,
            'lam': 0.95,
            'repeat_time': 20,
            'batch_size': 256,
            'eps_clip': 0.2,
            'w_entropy': 0.001
        }
        if agent_params: default_params.update(agent_params)
        
        self.hppo = HPPO(
            num_users=default_params['num_users'],
            num_states=default_params['num_states'],
            num_channels=default_params['num_channels'],
            lr_a=default_params['lr_a'],
            lr_c=default_params['lr_c'],
            pmax=default_params['pmax'],
            gamma=default_params['gamma'],
            lam=default_params['lam'],
            repeat_time=default_params['repeat_time'],
            batch_size=default_params['batch_size'],
            eps_clip=default_params['eps_clip'],
            w_entropy=default_params['w_entropy']
        )
        
        self.workflowids_checked = []
        self.load_model()

    def get_state(self, workflow_info):
        """将工作流信息转换为状态向量"""
        # workflow_info包含(sla, workflow_type, current_depth等)
        sla, workflow_type, current_depth = workflow_info
        
        # 示例特征工程（需根据实际情况调整）:
        state = np.array([
            sla / 1000.0,  # 归一化SLA
            current_depth / self.num_layers,  # 当前深度比例
            *self._encode_workflow_type(workflow_type)  # 类别特征编码
        ], dtype=np.float32)
        return state

    def _encode_workflow_type(self, wf_type):
        """简单独热编码示例"""
        types = ['mnist', 'fashionmnist', 'cifar100']
        return [1.0 if wf_type == t else 0.0 for t in types]

    def decision(self, workflowlist):
        """返回每个工作流的层划分点决策（1到num_layers之间的整数）"""
        states = []
        for _, _, sla, workflow in workflowlist:
            state = self.get_state((sla, workflow.lower(), 0))  # 假设current_depth需要获取
            states.append(state)
        
        # 批量获取动作（划分点）
        with torch.no_grad():
            states_tensor = torch.FloatTensor(np.array(states)).cuda()
            actions = []
            # 修改HPPO的select_action返回划分点（见HPPO调整部分）
            for i in range(len(workflowlist)):
                prob_points, _, _ = self.hppo.actors[0](states_tensor[i])
                point = Categorical(prob_points).sample().item()
                actions.append(point + 1)  # 假设输出0~n-1对应1~n层
        
        # 训练模式下收集经验
        if self.train:
            # 需要后续调用update方法（需与环境交互收集奖励）
            pass
        
        return actions

    def update(self, rewards, dones):
        """更新HPPO模型（需在环境反馈后调用）"""
        # 假设能获取到完整的回合数据
        # 需要将奖励等信息存入buffer并调用hppo.update()
        # （需要根据具体环境交互逻辑调整）
        pass

    def save_model(self, filename):
        self.hppo.save_model(filename, args=None)

    def load_model(self):
        # 加载预训练模型的逻辑
        pass