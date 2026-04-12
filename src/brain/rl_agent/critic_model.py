import torch
from torch import nn


class CriticNetwork(nn.Module):
    def __init__(self, in_embed_dim: int):
        super().__init__()

        self.layers = nn.Sequential(
            # Input Layer where the embed_dim represents the encoder's output
            nn.Linear(in_features=in_embed_dim, out_features=384),
            nn.ReLU(),

            nn.Linear(in_features=384, out_features=196),
            nn.ReLU(),
            nn.Linear(in_features=196, out_features=98),
            nn.ReLU(),

            # Output layer mapping to the len(k) where k is the number of controllers/functions
            nn.Linear(in_features=98, out_features=1)
        )
    
    def forward(self, input_embedding):
        x = self.layers(input_embedding)
        return x
