from random import choice
from torchtext import data
from functions import *




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = load_config("config.yaml")



dens_data = []
with open(config["training_data_name"],'r') as file:
    
    for line in file:
        dens_data.append([[i.split(',') for i in line.strip().split(',,,')[:-1]], float(line.strip().split(',,,')[-1])])




train_dataset = dens_data[:config["train_size"]]
val_dataset = dens_data[config["train_size"]:]



SMILES = data.Field(sequential=True, init_token = '<sos>', eos_token='<eos>')
SMILES.build_vocab([i[0][0] for i in train_dataset])



size = len(SMILES.vocab.stoi.items())
vectors = torch.eye(size)
SMILES.vocab.set_vectors(SMILES.vocab.stoi, vectors, size)




save_dict(SMILES, config["tokens_dict_name"])




for fold in range(config["folds"]):

    trn_data = train_dataset
    
    if config["folds"] > 1:
        trn_data = []
        for j in range(len(train_dataset)):
            trn_data.append(choice(train))
        
    train_fold(trn_data, val_dataset, SMILES, config["BATCH_SIZE"], config["val_BATCH_SIZE"], config["num_epochs"], device, fold, size, config["hidden_size"],
               config["num_layers"], config["bidirect"], config["file_name"], config["learning_rate"], config["sched_step"], config["sched_gamma"], config["weight_decay"])
