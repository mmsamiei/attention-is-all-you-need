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
    def __init__(self, train_iterator, valid_iterator, model, optimizer, pad_idx, device):
        self.train_iterator = train_iterator
        self.valid_iterator = valid_iterator
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def train_one_epoch(self):
        epoch_loss = 0
        for i, batch in enumerate(self.train_iterator):
            src = batch.src.to(self.device) # [sent_len, batch]
            trg = batch.trg.to(self.device) # [sent_len, batch]
            src = src.permute(1, 0) # [batch, sent_len]
            trg = trg.permute(1, 0) # [batch, sent_len]
            self.optimizer.optimizer.zero_grad()
            output = self.model(src, trg)
            # output = [batch, sent_len, trg_vocab_size]
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:].contiguous().view(-1) # TODO
            # output = [batch * sent_len, trg_vocab_size]
            # trg = [batch_size * sent_len]
            loss = self.criterion(output, trg)
            epoch_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        avr_epoch_loss = epoch_loss / len(self.train_iterator)
        return avr_epoch_loss

    def train(self, N_epoch = 10):
        self.model.train()
        epoch_losses = []
        valid_losses = []

        for i_epoch in range(N_epoch):
            start_time = time.time()
            epoch_loss = self.train_one_epoch()
            epoch_losses.append(epoch_loss)
            valid_loss = self.evaluate()
            valid_losses.append(valid_loss)
            end_time = time.time()
            epoch_min, epoch_sec = self.cal_epoch_time(start_time, end_time)
            print("epoch {}, time elapse is {} mins and {} secs".format(i_epoch, epoch_min, epoch_sec))
            print("train loss: {} , valid loss: {}".format(epoch_loss, valid_loss))

    def cal_epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def evaluate(self):
        self.model.eval()
        epoch_loss = 0
        for i, batch in enumerate(self.valid_iterator):
            src = batch.src.to(self.device)  # [sent_len, batch]
            trg = batch.trg.to(self.device)  # [sent_len, batch]
            src = src.permute(1, 0)  # [batch, sent_len]
            trg = trg.permute(1, 0)  # [batch, sent_len]
            output = self.model(src, trg)
            # output = [batch, sent_len, trg_vocab_size]
            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:].contiguous().view(-1)  # TODO
            # output = [batch * sent_len, trg_vocab_size]
            # trg = [batch_size * sent_len]
            loss = self.criterion(output, trg)
            epoch_loss += loss.item()
        return epoch_loss / len(self.valid_iterator)

