import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, num_states, num_points, num_channels, pmax, lstm_hidden=128):
        super(Actor, self).__init__()
        self.lstm = nn.LSTM(num_states, lstm_hidden, batch_first=True)
        self.point_header = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, num_points),
            nn.Softmax(dim=-1)
        )
        self.channel_header = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, num_channels),
            nn.Softmax(dim=-1))
        self.power_mu_header = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid())
        self.power_sigma_header = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Softplus())
        self.pmax = pmax
        self.hidden_size = lstm_hidden
        self.hidden = None

    def init_hidden(self, batch_size=1):
        return (torch.zeros(1, batch_size, self.hidden_size).cuda(),
                torch.zeros(1, batch_size, self.hidden_size).cuda())

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        lstm_out, new_hidden = self.lstm(x, hidden)
        last_out = lstm_out[:, -1, :]
        
        prob_points = self.point_header(last_out)
        prob_channels = self.channel_header(last_out)
        power_mu = self.power_mu_header(last_out) * (self.pmax - 1e-10) + 1e-10
        power_sigma = self.power_sigma_header(last_out)
        
        return prob_points, prob_channels, (power_mu, power_sigma), new_hidden
class Critic(nn.Module):
    def __init__(self, num_states, lstm_hidden=128):
        super(Critic, self).__init__()
        self.lstm = nn.LSTM(num_states, lstm_hidden, batch_first=True)
        self.value_head = nn.Sequential(
            nn.Linear(lstm_hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1))
        self.hidden_size = lstm_hidden
        self.hidden = None

    def init_hidden(self, batch_size=1):
        return (torch.zeros(1, batch_size, self.hidden_size).cuda(),
                torch.zeros(1, batch_size, self.hidden_size).cuda())

    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        lstm_out, new_hidden = self.lstm(x, hidden)
        last_out = lstm_out[:, -1, :]
        return self.value_head(last_out), new_hidden