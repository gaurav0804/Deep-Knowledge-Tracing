# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable



class LSTM(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim, layer_dim, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.embed_dim=embed_dim
        self.embeds=nn.Embedding(input_dim,embed_dim)
        self.model = nn.LSTM(embed_dim, hidden_dim, layer_dim)
        self.fc_1 = nn.Linear(self.hidden_dim, 10)
        self.fc = nn.Linear(10, self.output_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        embeddings=self.embeds(x)
        h0 = Variable(torch.zeros(self.layer_dim, embeddings.size(1), self.hidden_dim))
        c0 = Variable(torch.zeros(self.layer_dim, embeddings.size(1), self.hidden_dim))
        out,hn = self.model(embeddings,(h0,c0))
        out1 = self.fc_1(out)
        res = self.sig(self.fc(out1))
        return res
