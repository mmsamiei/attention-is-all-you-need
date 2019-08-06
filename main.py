from utils.dataloader import DataLoader
import torch
from model import model_utils
from optimizer.optimizer import NoamOpt
from train.trainer import Trainer

hidden_size = 256
num_encoder = 6
num_decoder = 6
n_head = 8
pf_dim = 1024
drop_out = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

dataloader = DataLoader(device)
train_iterator, valid_iterator, test_iterator = dataloader.load_data(64)
model = model_utils.create_model(dataloader.src_vocab_size(), dataloader.trg_vocab_size(), hidden_size, num_encoder, num_decoder, n_head, pf_dim,
                                 drop_out, dataloader.get_pad_idx(), device)

print(model_utils.count_parameters(model))
model_utils.init(model)
optimizer = NoamOpt(hidden_size , 1, 2000, torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

trainer = Trainer(train_iterator, valid_iterator, model, optimizer, dataloader.get_pad_idx(), device)
trainer.train(5)
# for i, batch in enumerate(train_iterator):
#     src = batch.src.permute(1, 0).to(device)
#     trg = batch.trg.permute(1, 0).to(device)
