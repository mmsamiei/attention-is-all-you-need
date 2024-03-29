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


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, pad_idx, device):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.pad_idx = pad_idx
        self.device = device

    def make_mask(self, src, trg):
        # src = [batch_size, src_sent_len]
        # trg = [batch_size, src_sent_len]

        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)

        # src_mask = [batch_size, 1, 1, src_sent_len]

        trg_pad_mask = (trg != self.pad_idx).unsqueeze(1).unsqueeze(3) #[batch_size, 1, sent_len, 1]
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.uint8, device=self.device))
        trg_mask = trg_pad_mask & trg_sub_mask

        return src_mask, trg_mask

    def forward(self, src, trg):

        # src = [batch_size, src_sent_len]
        # trg = [batch_size, trg_sent_len]
        src_mask, trg_mask = self.make_mask(src, trg)
        enc_src = self.encoder(src, src_mask)
        # enc_src = [batch size, src sent len, hid dim]
        out = self.decoder(trg, enc_src, trg_mask, src_mask)
        # out = [batch size, trg sent len, output dim]
        return out
