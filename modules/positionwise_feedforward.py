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

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.pf_dim = pf_dim

        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)

        self.do = nn.Dropout(dropout)

    def forward(self, x):

        # x = [batch_size, sent_len, hid_dim]

        x = x.permute(0, 2, 1)

        # x = [batch_size, hid_dim, sent_len]

        x = self.do(F.relu(self.fc_1(x)))

        # x = [batch_size, ff_dim, sent_len]

        x = self.fc_2(x)

        # x = [batch_size, hid_dim, sent_len]

        x = x.permute(0, 2, 1)

        # x = [batch_size, sent_len, hid_dim]

        return x

if __name__ == '__main__':

    hid_dim = 512
    pf_dim = 128
    dropout = 0.5
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    positionwise_feedforward = PositionwiseFeedforward(hid_dim, pf_dim, dropout).to(device)

    batch_size = 64
    sent_len = 40
    x = torch.rand(batch_size, sent_len, hid_dim).to(device)
    result = positionwise_feedforward(x)
    print(result.shape)