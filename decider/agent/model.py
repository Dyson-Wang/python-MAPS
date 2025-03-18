import torch.nn as nn


# model.py中修改Actor定义
class Actor(nn.Module):
    def __init__(self, num_states, num_points, num_channels, pmax):
        super().__init__()
        # 仅保留划分点决策部分
        self.fc = nn.Sequential(
            nn.Linear(num_states, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        self.point_head = nn.Linear(64, num_points)  # 输出各划分点的概率
        self.pmax = pmax

    def forward(self, x):
        x = self.fc(x)
        prob_points = F.softmax(self.point_head(x), dim=-1)
        return prob_points, None, (None, None)  # 保持接口兼容


class Critic(nn.Module):
    def __init__(self, num_states):
        super(Critic, self).__init__()
        self.net = nn.Sequential(nn.Linear(num_states, 256),
                                 nn.ReLU(),
                                 nn.Linear(256, 128),
                                 nn.ReLU(),
                                 nn.Linear(128, 64),
                                 nn.ReLU(),
                                 nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x)
