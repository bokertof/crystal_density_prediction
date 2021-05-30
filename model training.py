from random import choice
from torchtext import data
from functions import *
import dill



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


dens_data = []
with open('datasets/CCDC_CHNO.txt','r') as file:
    
    for line in file:
        dens_data.append([[i.split(',') for i in line.strip().split(',,,')[:-1]], float(line.strip().split(',,,')[-1])])



train_dataset = dens_data[:33000]
val_dataset = dens_data[33000:]


SMILES = data.Field(sequential=True, init_token = '<sos>', eos_token='<eos>')
SMILES.build_vocab([i[0][0] for i in train_dataset])

size = len(SMILES.vocab.stoi.items())
vectors = torch.eye(size)
SMILES.vocab.set_vectors(SMILES.vocab.stoi, vectors, size)





save_dict(SMILES, 'CCDC_CHNO')

print(SMILES.vocab.stoi.items())




BATCH_SIZE = 256
hidden_size = 128
num_layers = 4
learning_rate = 0.003
num_epochs = 500
folds = 1


for fold in range(folds):

    trn_data = train_dataset
    
    if folds > 1:
        trn_data = []
        for j in range(len(train_dataset)):
            trn_data.append(choice(train))
        
    train_fold(trn_data, val_dataset, SMILES, BATCH_SIZE, num_epochs, device, fold, size, hidden_size,
               num_layers, bidirect = True, file_name = 'DATASET_', learning_rate = learning_rate, sched_step = 10, sched_gamma = 0.9)
    






