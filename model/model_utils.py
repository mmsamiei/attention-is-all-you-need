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


def create_model(vocab_size, hid_dim, num_encoder_layer, num_decoder_layer, num_head, pf_dim, dropout, pad_idx, device):
    encoder = Encoder(vocab_size, hid_dim, num_encoder_layer, num_head, pf_dim,
                      EncoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    decoder = dec = Decoder(vocab_size, hid_dim, num_decoder_layer, num_head, pf_dim,
                            DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)
    model = Seq2Seq(encoder, decoder, pad_idx, device).to(device)
    return model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
