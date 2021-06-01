import torch, random, dill, yaml, os
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import RNN
from data_structures import DENSDataset

def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)

    return config


def save_dict(SMILES, filename):
    
    with open(filename, "wb") as file:
        dill.dump(SMILES, file)


        
def load_dict(filename):

    with open(filename, "rb") as file:
         return dill.load(file)



def train_NN(model, dataset, optimizer, criterion, length, device, clip = 50):
    
        loss_sum = 0
        model.train()
        
        for i, batch in enumerate(dataset):

            x, density = batch
            optimizer.zero_grad()
            outputs = model.forward(x.to(device))

            loss = criterion(outputs, density.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
            optimizer.step()
            
            loss_sum += loss.item()

        loss = loss_sum / length
        return loss




def evaluate(model, dataset, criterion, length, device):

    with torch.no_grad():
        model.eval()
        loss_ev = 0
        
        for i, batch in enumerate(dataset):
            
            x, density = batch
            outputs = model.forward(x.to(device))
            loss = criterion(outputs, density.to(device))
            loss_ev += loss.item()
            
        loss = loss_ev / length

        return loss




def train_fold(train_dataset, val_dataset, SMILES, BATCH_SIZE, val_BATCH_SIZE, num_epochs, device, fold, input_size, hidden_size,
               num_layers, bidirect = True, file_name = 'DATASET_', learning_rate = 0.001, sched_step = 10, sched_gamma = 0.9, weight_decay = 0):


    model = RNN(input_size, input_size, hidden_size, num_layers, bidirect).to(device)

    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Number of parameters: ', total_params, '\n')


    criterion = nn.L1Loss(reduction = 'sum')
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay) 
    scheduler = optim.lr_scheduler.StepLR(optimizer, sched_step, sched_gamma)



    pretrained_embeddings = SMILES.vocab.vectors
    model.embedding.weight.data = pretrained_embeddings.to(device)
    model.embedding.weight.requires_grad = False
    

    print(str(fold+1) +'-BOOTSTRAP, ', 'TRAIN DATASET: ', len(train_dataset), ', VAL DATASET: ',len(val_dataset))
    filename = file_name + str(fold+1)



    train_prep_data = DENSDataset(train_dataset, SMILES)
    val_prep_data = DENSDataset(val_dataset, SMILES)

    train_dataloader = DataLoader(train_prep_data, 
                                  batch_size = BATCH_SIZE,
                                  shuffle = True,
                                  collate_fn = train_prep_data.collate_fn)
    
    val_dataloader = DataLoader(val_prep_data, 
                                  batch_size = val_BATCH_SIZE,
                                  shuffle = True,
                                  collate_fn = val_prep_data.collate_fn)

    prev_val_loss = float('Inf')
    losses = []

    
    for epoch in range(1, num_epochs+1):
        
        train_loss = train_NN(model, train_dataloader, optimizer, criterion, len(train_dataset), device)
        val_loss = evaluate(model, val_dataloader, criterion, len(val_dataset), device)

        print(f"| EPOCH {epoch:3.0f} | TRAIN LOSS {train_loss:.5f} | EVAL LOSS {val_loss:.5f} |")

        scheduler.step()
        losses.append(str(train_loss) + '     ' +str(val_loss))

        if val_loss < prev_val_loss:
                
            prev_val_loss = val_loss
            torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    }, filename +'.pth')
            
            print('~~SAVING THE BETTER MODEL WEIGHTS | Current val loss:', val_loss)

    
    with open(filename + '_losses.txt', 'w') as out:
        for i in losses:
            out.write(i+'\n')
 
