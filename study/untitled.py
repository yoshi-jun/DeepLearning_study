import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

class env():
    def __init__(self):
        self.angle = 0
        self.r = 0
        self.done = False

        self.ang_range = 45
        self.ev_array = np.ones((31,31,30))
        
    def reset(self):
        #ここでもrを計算するために線量分布を計算した方が良いかも？
        self.r = 0
        self.angle = 0
        self.done = False

    def step(self,action):
        self.angle += 5 * action
        
        if (np.abs(self.angle) > self.ang_range):
            self.done = True
        
        dose = np.random.rand(31,31,30)
#         
        # cmd = "source /opt/geant4/re10.07.p03-mt/bin/geant4.sh; ./bin/Application_Main 100 -phi " + str(self.angle)
        # test = subprocess.check_output(cmd, shell = True, executable = "/bin/bash")
        
#         dose = pd.readcsv(filename)['dose']
#         dose = dose.reshape(31,31,30)
        
        reword = self.ev_array * dose
        reword = reword.sum() 
        self.r = reword - self.r
        
        return self.angle, reword, self.done
      
class PolicyNet(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, nch_g=64):
        super(PolicyNet, self).__init__()
        
        self.layers = nn.ModuleDict({
            'Layer1': nn.Sequential(
                nn.Conv3d(1, nch_g, 2),
                nn.BatchNorm3d(nch_g),
                nn.ReLU()
            ),
            'Layer2': nn.Sequential(
                nn.Conv3d(nch_g, nch_g * 2, 2),
                nn.BatchNorm3d(nch_g * 2),
                nn.ReLU()
            ),
            'Layer3': nn.Sequential(
                nn.Conv3d(nch_g * 2, nch_g * 4, 2),
                nn.BatchNorm3d(nch_g * 4),
                nn.ReLU()
            ),
            'Layer4': nn.Sequential(
                nn.Flatten(),
                nn.Linear(5038848, 3),
                nn.Sigmoid()
            )
        })

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x
      
class ValueNet(nn.Module):
    """
    implements both actor and critic in one model
    """
    def __init__(self, nch_g=64):
        super(ValueNet, self).__init__()
        
        self.layers = nn.ModuleDict({
            'Layer1': nn.Sequential(
                nn.Conv3d(1, nch_g, 2),
                nn.BatchNorm3d(nch_g),
                nn.ReLU()
            ),
            'Layer2': nn.Sequential(
                nn.Conv3d(nch_g, nch_g * 2, 2),
                nn.BatchNorm3d(nch_g * 2),
                nn.ReLU()
            ),
            'Layer3': nn.Sequential(
                nn.Conv3d(nch_g * 2, nch_g * 4, 2),
                nn.BatchNorm3d(nch_g * 4),
                nn.ReLU()
            ),
            'Layer4': nn.Sequential(
                nn.Flatten(),
                nn.Linear(5038848, 1),
                nn.Sigmoid()
            )
        })

    def forward(self, x):
        for layer in self.layers.values():
            x = layer(x)
        return x
      
class Agent:
    def __init__(self):
        self.gamma = 0.98
        self.lr_pi = 0.0002
        self.lr_v = 0.0005
        self.action_size = 2

        self.pi = PolicyNet()
        self.v = ValueNet()
        self.optimizer_pi = optim.Adam(self.pi.parameters(), lr=self.lr_pi)
        self.optimizer_v = optim.Adam(self.v.parameters(), lr=self.lr_v)

    def get_action(self, state):
        # state = torch.from_numpy(state).float()
        probs = self.pi(state)
        action = torch.multinomial(probs, 1).item()
        return action, probs[:,action].item()

    def update(self, state, action_prob, reward, next_state, done):
        # state = torch.from_numpy(state).float()
        next_state = torch.randn(1, 1, 30, 30, 30)

        # ========== (1) Update V network ===========
        with torch.no_grad():
            target = reward + self.gamma * self.v(next_state) * (1 - done)
        v = self.v(state)
        loss_v = F.mse_loss(v, target)

        # ========== (2) Update pi network ===========
        delta = target - v
        loss_pi = -torch.log(action_prob) * delta.item()

        self.v.zero_grad()
        self.pi.zero_grad()
        loss_v.backward()
        loss_pi.backward()
        self.optimizer_v.step()
        self.optimizer_pi.step()