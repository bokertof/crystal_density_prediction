import torch
import torchtext
from torchtext import data
import numpy as np
from model import *
import matplotlib.pyplot as plt
import random

from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pack_sequence




tr = []

with open('datasets/' + 'CCDC_all.txt','r') as file:
    
    for line in file:
        tr.append([[i.split(',') for i in line.strip().split(',,,')[:-1]], float(line.strip().split(',,,')[-1])])




##check = [''.join(i[0][0]) for i in tr[:90000]]
##
##
##print(check[:10])
##
##
##with open('datasets/' + 'Casey_trn_test_valid.txt','r') as file: #C:/Users/Stanislav/Desktop/scripts/NN for density/dataset rice/Rice prep.txt
##    data_smiles = file.readlines()[:21012]
##    val_all2 = [[[i.split(',') for i in line.strip().split(',,,')[:-1]], float(line.strip().split(',,,')[-1])] for line in data_smiles]
##
##print(len(val_all2))
##val_all = []
##for i in val_all2:
##    if ''.join(i[0][0]) not in check:
##        val_all.append(i)
##
##
##
##print(len(val_all))     
        
print(len(tr))


data_smiles = tr[90000:]



TEXT = data.Field(sequential=True, init_token = '<sos>', eos_token='<eos>')
TEXT.build_vocab([i[0][0] for i in tr[:90000]])



class DENSDataset(Dataset):
    def __init__(self, data):

        self.corpus = [i[0] for i in data]
        self.dens = [i[1] for i in data]

        
        print('min density', min(self.dens))
        print('max density', max(self.dens))
        print('mean density', np.mean(self.dens))
        
        self.sos = '<sos>'
        self.eos = '<eos>'

        
    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):

        src = random.choice(self.corpus[idx])
        self.src = [self.sos] + src + [self.eos]

        return ([TEXT.vocab.stoi[i] for i in src], self.dens[idx]) 
        

    def collate_fn(self, batch):
        """
        Technical method to form a batch to feed into recurrent network
        """

        return pack_sequence([torch.tensor(pair[0]) for pair in batch], enforce_sorted=False), torch.tensor([pair[1] for pair in batch]).view(-1,1)







    
    
print(TEXT.vocab.stoi.items())


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedd_size = len(TEXT.vocab.stoi.items())
hidden_size = 128
num_layers = 4
input_size = embedd_size









dataset_smiles = DENSDataset(data_smiles)
dataloader = DataLoader(dataset_smiles, 
                              batch_size=2000,
                              shuffle=False,
                              collate_fn=dataset_smiles.collate_fn)






import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
criterion = nn.MSELoss()




               


checkpoints = ['Z1_0_.pth','Z1_1_.pth','Z1_2_.pth']





aver_preds = torch.zeros(len(data_smiles), 1)
print('amount of data in dataset:', len(data_smiles))






checkpoints = ['dens_Casey_boost1.pth', 'dens_Casey_boost2.pth', 'dens_Casey_boost3.pth', 'dens_Casey_boost4.pth', 'dens_Casey_boost5.pth']

checkpoints = ['CCDC_b1.pth', 'CCDC_b2.pth', 'CCDC_b3.pth', 'CCDC_b4.pth', 'CCDC_b5.pth']

checkpoints = ['CCDC_CHNO_b1.pth', 'CCDC_CHNO_b2.pth', 'CCDC_CHNO_b3.pth', 'CCDC_CHNO_b4.pth', 'CCDC_CHNO_b5.pth']

checkpoints = ['CCDC.pth']

true_dens = [i[-1] for i in data_smiles]

for check in checkpoints:
    
    model = RNN(input_size, embedd_size, hidden_size, num_layers, bidirect = True).to(device)
    model.load_state_dict(torch.load('checkpoints/' + check)['state_dict'])
    print(check, ' model successfuly loaded')

    model.eval()

    
    iter_n = 30
    for i in range(iter_n):
        #print(i+1, '- fold')

        index = 0
        with torch.no_grad():
            for batch in dataloader:
                smiles = batch[0]
                density = batch[1]

                preds = model.forward(smiles.to(device)).to('cpu')


                aver_preds[index:index+preds.shape[0],:] += preds
                

                index += preds.shape[0]


                #print('MAE: ', abs(preds - density.to(device)).mean())
                #print('RMSE: ', criterion(preds, density.to(device))**0.5)


        
    
    #print(str(abs(torch.tensor(true_dens).view(-1, 1) - aver_preds.to('cpu')).mean().item()).replace('.',','))





aver_preds /= (len(checkpoints)*(iter_n))



with open('val_all_densities.txt','w') as file:
    for i in range(aver_preds.shape[0]):

        file.write(''.join(data_smiles[i][0][0]) + '    ' + str(len(data_smiles[i][0][0])) + '  ' + str(format(data_smiles[i][-1], '.6f')) + '  ' + str(aver_preds[i].item()) + '  ' +  str(abs(data_smiles[i][-1] - aver_preds[i].item())) + '\n')





with open('val_all_densities.txt','r') as file:
    data = file.readlines()
    le = np.array([float(i.split()[1]) for i in data])
    true = np.array([float(i.split()[2]) for i in data])
    preds = np.array([float(i.split()[3]) for i in data])

##res = {}
##for i in range(0,150):
##    mean = 0
##    k = 0
##    for j in range(len(le)):
##        if i == le[j]:
##            mean += abs(true[j]-preds[j])
##            k+=1
##    if k != 0:
##        mean /= k
##        
##        res.update({i:mean})
##            
##for i in res.keys():
##    print(i, res[i])
from scipy.stats import pearsonr

print(pearsonr(preds, true))
print('min true:',min(true))
print('min preds:',min(preds))
print('max true:',max(true))
print('max preds:',max(preds))
print('max diff item:',max(abs(true-preds)), np.argmax(abs(true-preds)))
print('MAE:',sum(abs(true-preds))/len(true))
print('RMSP:', np.sqrt(np.mean(np.square(((true-preds) / true)), axis=0)))


k = abs(true-preds)
for i in range(len(k)):
    if k[i] > 0.5:
        print(k[i], i)


print('PERCENTAGE with error less 0.01',sum(k <= 0.01) / len(k))
print('PERCENTAGE with error less 0.02',sum(k <= 0.02) / len(k))
print('PERCENTAGE with error less 0.03',sum(k <= 0.03) / len(k))
print('PERCENTAGE with 0.03 < error <= 0.05',((0.03 < k) & (k <= 0.05)).sum() / len(k))
print('PERCENTAGE with 0.05 < error <= 0.1',((0.05 < k) & (k <= 0.1)).sum() / len(k))
print('PERCENTAGE with error > 0.1',sum(k > 0.1) / len(k))



_ = plt.hist(true, bins='auto', alpha=0.5, label='true',color='r')
_ = plt.hist(preds, bins='auto', alpha=0.5, label='preds',color='g')
plt.legend(loc='upper right')        
plt.show()


##plt.plot(le, abs(true-preds), 'ro', color = 'black', markersize = 1.5)
##plt.xlabel('SMILES length')
##plt.ylabel('Mean absolute error')
##plt.show()


plt.plot(true, preds, 'ro', color = 'black', markersize = 3)
plt.plot([0,10], [0,10], 'ro-', color = 'red')
axes = plt.gca()
axes.set_xlim([0,8])
axes.set_ylim([0,8])

plt.xlabel('True Density')
plt.ylabel('Predicted Density')

plt.show()

