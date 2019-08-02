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

class Trainer():
    def __init__(self, train_iterator, valid_iterator, model, optimizer, device):
        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.model = model
        self.optimizer = optimizer
        self.device = device

    def train_epoch(self):
        for i, batch in enumerate(self.train_iterator):
            src = batch.src.to(self.device) # [sent_len, batch]
            trg = batch.trg.to(self.device) # [sent_len, batch]
            src = src.permute(1, 0) # [batch, sent_len]
            trg = trg.permute(1, 0) # [batch, sent_len]
            self.optimizer.optimizer.zero_grad()
            output = self.model(src, trg)
