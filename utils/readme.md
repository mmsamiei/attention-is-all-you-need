**dataloader.py**
* a dataloader which is based on torchtext and spacy for use it just 
write following codes:

```python
dataloader = DataLoader(device)
train_iterator, valid_iterator, test_iterator = a.load_data(batchsize)
for i, batch in enumerate(train_iterator):   
    src = batch.src  
    trg = batch.trg
    print(src.shape)
```