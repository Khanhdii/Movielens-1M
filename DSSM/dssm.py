import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, constant_

class MLPLayers(nn.Module):
    def __init__(self, size_list, dropout_prob, activation="relu", bn=False):
        super(MLPLayers, self).__init__()
        self.layers = nn.ModuleList()

        activation_fn = getattr(nn, activation) if hasattr(nn, activation) else nn.ReLU

        for in_size, out_size in zip(size_list[:-1], size_list[1:]):
            self.layers.append(nn.Linear(in_size, out_size))
            if bn:
                self.layers.append(nn.BatchNorm1d(out_size))
            self.layers.append(activation_fn())
            self.layers.append(nn.Dropout(dropout_prob))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DSSM(nn.Module):
    """DSSM: Deep Structured Semantic Model"""

    def __init__(self, num_users, num_movies, embedding_dim=64, mlp_hidden_size=[256, 128], dropout_prob=0.5, binary_classification=False):
        super(DSSM, self).__init__()

        self.binary_classification = binary_classification

        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)

        # MLP layers for user and item
        user_size_list = [embedding_dim] + mlp_hidden_size
        item_size_list = [embedding_dim] + mlp_hidden_size

        self.user_mlp_layers = MLPLayers(user_size_list, dropout_prob, activation="relu", bn=True)
        self.item_mlp_layers = MLPLayers(item_size_list, dropout_prob, activation="relu", bn=True)

        # Final layer (1 output for binary, 5 for multi-class classification)
        # Điều chỉnh số lượng đầu ra dựa vào `binary_classification`
        self.output_layer = nn.Linear(mlp_hidden_size[-1], 1 if self.binary_classification else 5)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight.data)
        elif isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0)

    def forward(self, user, movie):
        user_embed = self.user_embedding(user)
        movie_embed = self.movie_embedding(movie)

        user_embed = self.user_mlp_layers(user_embed)
        movie_embed = self.item_mlp_layers(movie_embed)

        combined = user_embed + movie_embed

        output = self.output_layer(combined)
        return output
