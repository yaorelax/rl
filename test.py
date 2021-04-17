import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ANet(nn.Module):  # ae(s)=a
    def __init__(self, s_dim, a_dim):
        super(ANet, self).__init__()
        self.fc1 = nn.Linear(s_dim, 30)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(30, a_dim)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.out(x)
        x = torch.tanh(x)
        actions_value = x * 2
        return actions_value

class CNet(nn.Module):  # c(s,a)=q
    def __init__(self, s_dim, a_dim):
        super(CNet, self).__init__()
        self.fcs = nn.Linear(s_dim, 30)
        self.fcs.weight.data.normal_(0, 0.1)  # initialization
        self.fca = nn.Linear(a_dim, 30)
        self.fca.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(30, 1)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, s, a):
        x = self.fcs(s)
        y = self.fca(a)
        net = F.relu(x + y)
        actions_value = self.out(net)
        return actions_value

actor = ANet(2, 2)
critic = CNet(2, 2)
bs = torch.FloatTensor(
    [[1, 2],
     [3, 4],
     [5, 6]]
)
a = actor(bs)
q = critic(bs, a)
loss = -torch.mean(q)
ctrain = torch.optim.Adam(critic.parameters(), lr=0.01)
ctrain.zero_grad()
loss.backward()
ctrain.step()
pass
pass

