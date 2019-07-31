import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy

import random
import math
import time



class DataLoader:
    def __init__(self, dev):
        self.spacy_de = spacy.load('de')
        self.spacy_en = spacy.load('en')
        self.SRC = Field(tokenize="spacy", init_token = '<sos>', eos_token = '<eos>', lower = True)
        self.TRG = Field(tokenize="spacy", init_token='<sos>', eos_token='<eos>', lower=True)
        self.dev = dev

    def load_data(self, batch_size):
        train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(self.SRC, self.TRG))
        self.SRC.build_vocab(train_data, min_freq = 2)
        self.TRG.build_vocab(train_data, min_freq = 2)
        train_iterator, valid_iterator, test_iterator= BucketIterator.splits((train_data, valid_data, test_data), batch_size=batch_size, device=self.dev)
        return train_iterator, valid_iterator, test_iterator






