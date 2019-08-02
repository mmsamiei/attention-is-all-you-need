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

from modules.encoder import Encoder
from modules.encoder_layer import EncoderLayer
from modules.self_attention import SelfAttention
from modules.positionwise_feedforward import PositionwiseFeedforward
from modules.decoder import Decoder
from modules.decoder_layer import DecoderLayer
from modules.seq2seq import Seq2Seq

class Modeling():
    def __init__(self, vocab_size, pad_idx, device):
        self.encoder = None
        self.decoder = None
        self.model = None
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.device = device

    def create_model(self, hid_dim, num_encoder_layer, num_decoder_layer, num_head, pf_dim, dropout):
        self.encoder = Encoder(self, self.vocab_size, hid_dim, num_encoder_layer, num_head, pf_dim,
                               EncoderLayer, SelfAttention, PositionwiseFeedforward, dropout, self.device)
        self.decoder = dec = Decoder(self.vocab_size, hid_dim, num_decoder_layer, num_head, pf_dim,
                                     DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, self.device)
        self.model = Seq2Seq(self.encoder, self.decoder, self.pad_idx, self.device).to(self.device)
        return self.model








