from collections import OrderedDict
import torch
import torch.nn as nn


class RatingPrediction(nn.Module):
    """
    Follow the description in Section 3.2
    """
    def __init__(self, input_dim, hidden_size, hidden_layer_num=2):
        super(RatingPrediction, self).__init__()
        self.W1 = nn.Linear(input_dim, hidden_size, bias=True)

        hidden_layers = []
        for i in range(hidden_layer_num):
            hidden_layers.append((f'linear_{i}', nn.Linear(hidden_size, hidden_size)))
            hidden_layers.append((f'activ_func_{i}', nn.Sigmoid()))
        self.hidden_layers = nn.Sequential(OrderedDict(hidden_layers))

        self.final_layer = nn.Linear(hidden_size, 1)

    def forward(self, user_embeddings, item_embeddings):
        x = torch.cat([user_embeddings, item_embeddings], dim=-1)
        x = self.W1(x)
        features = self.hidden_layers(x)
        rating = self.final_layer(features)    # ref: Equation 3.
        return rating, features
