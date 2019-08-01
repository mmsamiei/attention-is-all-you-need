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

class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        self.hid_dim = hid_dim
        self.n_heads = n_heads

        assert hid_dim % n_heads == 0

        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)

        self.fc = nn.Linear(hid_dim, hid_dim)

        self.do = nn.Dropout(dropout)

        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        #query  = key = value = [batch_size, sent_len, hid_dim]

        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        #Q, K, V = [batch_size, sent_len, hid_dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        #Q, K, V = [batch_size, n_heads, sent_len, hid_dim//n_heads]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2))

        #energy = [batch_size, n_heads, sent_len, sent_len]

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = self.do(F.softmax(energy, dim=-1))

        # attention [batch_size, n_heads, sent_len, sent_len]

        x = torch.matmul(attention, V)

        # x = [batch_size, n_heads, sent_len, hid_dim//n_head]

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch_size, sent_len, n_heads, hid_dim // n_head]

        x = x.view(batch_size, -1, self.n_heads * (self.hid_dim // self.n_heads))

        # x = [batch_size, src_sent_len, hid_dim]

        x = self.fc(x)

        # x = [batch_size, sent_len, hid_dim]

        return x

