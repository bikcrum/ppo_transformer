import math

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def orthogonal_init(layer, gain=np.sqrt(2)):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    # x: [seq_len, batch_size, embedding_dim]
    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(args.state_dim, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hidden_dim, args.hidden_dim),
        )

        self.pos_encoder = PositionalEncoding(d_model=args.hidden_dim, dropout=args.transformer_dropout,
                                              max_len=args.transformer_window)
        encoder_layers = nn.TransformerEncoderLayer(d_model=args.hidden_dim, nhead=args.transformer_nhead,
                                                    dim_feedforward=args.transformer_dim_feedforward,
                                                    dropout=args.transformer_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=args.transformer_num_layers)

        self.mean_layer = nn.Linear(args.hidden_dim, args.action_dim)

        # self.std_layer = nn.Linear(args.hidden_dim * 2, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))

        if args.use_orthogonal_init:
            for layer in self.fc_layers:
                if isinstance(layer, nn.Linear):
                    orthogonal_init(layer)

            orthogonal_init(self.mean_layer, gain=0.01)
            # orthogonal_init(self.std_layer, gain=0.01)

    # s: [batch_size, seq_len, width, height, channel]
    def forward(self, s, mask, need_weights=False):
        s = self.fc_layers(s)
        # s: [batch_size, seq_len, hidden_dim]

        s = s.transpose(0, 1)
        # s: [seq_len, batch_size, hidden_dim * 2]

        s = self.pos_encoder(s)
        # s: [seq_len, batch_size, hidden_dim * 2]

        s, attn_maps = self.transformer_encoder(s, mask=mask, need_weights=need_weights)
        # s: [seq_len, batch_size, hidden_dim * 2]

        s = s.transpose(0, 1)
        # s: [batch_size, seq_len, hidden_dim * 2]

        mean = torch.tanh(self.mean_layer(s))
        # mean: [batch_size, seq_len, action_dim]

        # std = F.softplus(self.std_layer(s))
        # std: [batch_size, seq_len, action_dim]

        if need_weights:
            return mean, self.log_std.expand_as(mean).exp(), attn_maps

        return mean, self.log_std.expand_as(mean).exp()

    def pdf(self, s, mask):
        mean, std = self.forward(s, mask)
        return Normal(mean, std)


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()

        self.fc_layers = nn.Sequential(
            nn.Linear(args.state_dim, args.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(args.hidden_dim, args.hidden_dim),
        )

        self.pos_encoder = PositionalEncoding(d_model=args.hidden_dim, dropout=args.transformer_dropout,
                                              max_len=args.transformer_window)
        encoder_layers = nn.TransformerEncoderLayer(d_model=args.hidden_dim, nhead=args.transformer_nhead,
                                                    dim_feedforward=args.transformer_dim_feedforward,
                                                    dropout=args.transformer_dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=args.transformer_num_layers)

        self.value_layer = nn.Linear(args.hidden_dim, 1)

        if args.use_orthogonal_init:
            for layer in self.fc_layers:
                if isinstance(layer, nn.Linear):
                    orthogonal_init(layer)

            orthogonal_init(self.value_layer, gain=0.01)

    # s: [batch_size, seq_len, width, height, channel]
    def forward(self, s, mask):
        s = self.fc_layers(s)
        # s: [batch_size, seq_len, hidden_dim]

        s = s.transpose(0, 1)
        # s: [seq_len, batch_size, hidden_dim * 2]

        s = self.pos_encoder(s)
        # s: [seq_len, batch_size, hidden_dim * 2]

        s, _ = self.transformer_encoder(s, mask=mask)
        # s: [seq_len, batch_size, hidden_dim * 2]

        s = s.transpose(0, 1)
        # s: [batch_size, seq_len, hidden_dim * 2]

        s = self.value_layer(s)
        # mean: [batch_size, seq_len, 1]

        return s
