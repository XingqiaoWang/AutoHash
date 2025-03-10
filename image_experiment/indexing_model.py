import torch
import torch.nn as nn

class AutoHash_Model(nn.Module):
    def __init__(self, input_dim, encoding_dim, initial_position=0.0):
        super(AutoHash_Model, self).__init__()
        self.margin_position = nn.Parameter(torch.full((encoding_dim,), initial_position))
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, encoding_dim),
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.BatchNorm1d(32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Linear(512, input_dim),
            nn.Tanh()
        )

    
    def forward(self, x, encode_only=False):
        encoded = self.encoder(x)
        if encode_only:
            return encoded
        decoded = self.decoder(encoded)
        return decoded, encoded
