import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, 
                 use_duel_dqn=False, model_layout="64_2_layers"):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            use_duel_dqn (boolean): use duel DQN
            model_layout (string): model name (64_2_layers, 64_3_layers, 128_2_layers, 128_3_layers )
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.use_duel_dqn = use_duel_dqn
        self.model_layout = model_layout
        
        if "64_2_layers" == self.model_layout:
            self.fc1 = nn.Linear(state_size, 64)
            self.fc2 = nn.Linear(64, 32)
            self.fc3 = nn.Linear(32, action_size)        
            self.v=nn.Linear(32,1)
            
        elif "128_2_layers" == model_layout:
            self.fc1 = nn.Linear(state_size, 128)
            self.fc2 = nn.Linear(128, 32)
            self.fc3 = nn.Linear(32, action_size)
            self.v=nn.Linear(32,1)
            
        elif "64_3_layers" == model_layout:
            self.fc1 = nn.Linear(state_size, 64)
            self.fc2 = nn.Linear(64, 32 )
            self.fc3 = nn.Linear(32, 16)
            self.fc4 = nn.Linear(16, action_size)
            self.v=nn.Linear(16,1)
        elif "128_3_layers" == model_layout:
            self.fc1 = nn.Linear(state_size, 128)
            self.fc2 = nn.Linear(128, 64 )
            self.fc3 = nn.Linear(64, 32)
            self.fc4 = nn.Linear(32, action_size)
            self.v=nn.Linear(32,1)
        
       
    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        if self.model_layout in ("64_2_layers", "128_2_layers"):
            if self.use_duel_dqn:
                return self.fc3(x)+self.v(x)
            else:
                return self.fc3(x)
            
        else:
            x=F.relu(self.fc3(x))
            if self.use_duel_dqn:
                return self.fc4(x)+self.v(x)
            else: 
                return self.fc4(x)
        