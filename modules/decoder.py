import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy

import random
import math
import os
import time

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super().__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.pf_dim = pf_dim
        self.decoder_layer = decoder_layer
        self.self_attention = self_attention
        self.positionwise_feedforward = positionwise_feedforward
        self.dropout = dropout
        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(1000, hid_dim)

        self.layers = nn.ModuleList([decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward,
                                                   dropout, device)])

        self.fc = nn.Linear(hid_dim, output_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, src, trg_mask, src_mask):

        #trg = [batch_size, trg_sent_len]
        #src = [batch_size, src_sent_len]
        #trg_mask = [batch_size, trg_sent_len]
        # trg_mask = [batch_size, src_sent_len]

        pos = torch.arange(0, trg.shape[1]).unsqueeze(0).repeat(trg.shape[0], 1).to(self.device)
        trg = self.do((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        # trg = [batch_size, trg_sent_len, hid_dim]

        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)

        return self.fc(trg)

