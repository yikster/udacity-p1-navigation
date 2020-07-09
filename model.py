import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, 
                 use_duel_dqn=False):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            use_duel_dqn (boolean): use duel DQN
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_duel_dqn = use_duel_dqn
        
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_size)        
        self.v=nn.Linear(32,1)
        
       
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        if self.use_duel_dqn:
            return self.fc3(x)+self.v(x)
        else:
            return self.fc3(x)