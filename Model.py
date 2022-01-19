import os
import torch
from torch import nn
from torch.utils.data import DataLoader


class TextClassificationModel(nn.Module):
    # def __init__(self):
    #     super(NeuralNetwork, self).__init__()
    #     # self.flatten = nn.Flatten()
    #     # self.linear_relu_stack = nn.Sequential(
    #     #     nn.Linear(28*28, 512),
    #     #     nn.ReLU(),
    #     #     nn.Linear(512, 512),
    #     #     nn.ReLU(),
    #     #     nn.Linear(512, 10),
    #     # )

    def __init__(self, vocab_size, embed_dim, num_class):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)
        self.init_weights()

    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets)
        return self.fc(embedded)

