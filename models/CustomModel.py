import gym
import torch
import torch.nn as nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class LSTMfeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(LSTMfeatures, self).__init__(observation_space, features_dim)
        self.hidden_size = 256
        self.num_layers = 6
        self.lstm = nn.LSTM(observation_space.shape[1], self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, features_dim)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        
        h0 = torch.zeros(self.num_layers, observations.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, observations.size(0), self.hidden_size).to(self.device)
        out, hidden = self.lstm(observations, (h0, c0))

        out = self.fc(out[:, -1, :])
        return out