import torch
import torch.nn as nn
import torch.nn.functional as F

class MuZeroNet(nn.Module):
    def __init__(self, observation_size, action_size, hidden_size, device):
        super().__init__()
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.device = device
        
        # Representation network (s -> h)
        self.representation = nn.Sequential(
            nn.Conv2d(observation_size[0], hidden_size, 3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
            nn.BatchNorm2d(hidden_size),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(hidden_size * observation_size[1] * observation_size[2], hidden_size)
        )
        
        # Dynamics network (h, a -> h)
        self.dynamics = nn.Sequential(
            nn.Linear(hidden_size + action_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Prediction networks
        self.policy = nn.Linear(hidden_size, action_size)
        self.value = nn.Linear(hidden_size, 1)
        self.reward = nn.Linear(hidden_size, 1)
        
    def representation_network(self, obs):
        return self.representation(obs)
        
    def dynamics_network(self, hidden_state, action):
        # One-hot encode the action
        action_one_hot = torch.zeros(action.size(0), self.action_size, device=self.device)
        action_one_hot.scatter_(1, action, 1)
        
        # Concatenate hidden state and action
        x = torch.cat([hidden_state, action_one_hot], dim=1)
        return self.dynamics(x)
        
    def prediction_network(self, hidden_state):
        policy_logits = self.policy(hidden_state)
        value = self.value(hidden_state)
        return policy_logits, value
        
    def initial_inference(self, obs):
        hidden_state = self.representation_network(obs)
        policy_logits, value = self.prediction_network(hidden_state)
        return hidden_state, policy_logits, value
        
    def recurrent_inference(self, hidden_state, action):
        next_hidden_state = self.dynamics_network(hidden_state, action)
        policy_logits, value = self.prediction_network(next_hidden_state)
        reward = self.reward(next_hidden_state)
        return next_hidden_state, reward, policy_logits, value 