from tokenizer import ChemTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_sequence
import torch
from random import choice


class DENSDataset(Dataset):
        
    def __init__(self, data, SMILES):

        self.corpus = [i[0] for i in data]
        self.dens = [i[1] for i in data]
        
        self.sos = '<sos>'
        self.eos = '<eos>'

        self.SMILES = SMILES

        
    def __len__(self):
            
        return len(self.corpus)

    def __getitem__(self, idx):

        src = choice(self.corpus[idx])
        self.src = [self.sos] + src + [self.eos]

        return ([self.SMILES.vocab.stoi[i] for i in src], self.dens[idx]) 
        

    def collate_fn(self, batch):

        return pack_sequence([torch.tensor(pair[0]) for pair in batch], enforce_sorted=False), torch.tensor([pair[1] for pair in batch]).view(-1,1)


class infDENSDataset(Dataset):
        
    def __init__(self, data, SMILES):

        self.corpus = data
        self.SMILES = SMILES
        
        self.sos = '<sos>'
        self.eos = '<eos>'
        
    def __len__(self):
            
        return len(self.corpus)

    def __getitem__(self, idx):

        src = self.corpus[idx]
        random_SMILES = ChemTokenizer.randomize([src], 1, 1)[0]
        random_SMILES_tkn = ChemTokenizer.tokenize(random_SMILES)[0]
        prep_SMILES = [self.sos] + random_SMILES_tkn + [self.eos]

        return [self.SMILES.vocab.stoi[i] for i in prep_SMILES]
        

    def collate_fn(self, batch):

        return pack_sequence([torch.tensor(item) for item in batch], enforce_sorted=False)
