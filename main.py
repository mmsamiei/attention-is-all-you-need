from utils.dataloader import DataLoader
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
a = DataLoader(device)
train_iterator, valid_iterator, test_iterator = a.load_data(64)
for i, batch in enumerate(train_iterator):
    src = batch.src
    trg = batch.trg
    print(src.shape)
