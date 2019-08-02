from utils.dataloader import DataLoader
import torch
from model import model_utils

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataloader = DataLoader(device)
train_iterator, valid_iterator, test_iterator = dataloader.load_data(64)
model = model_utils.create_model(dataloader.src_vocab_size(), 512, 3, 3, 8, 1024, 0.5, dataloader.get_pad_idx(), device)
print(model_utils.count_parameters(model))
model_utils.init(model)

for i, batch in enumerate(train_iterator):
    src = batch.src
    trg = batch.trg
    #print(dataloader.src_vocab_size())
