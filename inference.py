from functions import load_config, load_dict
from data_structures import infDENSDataset
from model import RNN
import torch
from torch.utils.data import DataLoader



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = load_config("config.yaml")
SMILES = load_dict(config["tokens_dict_name"])



with open(config["inference_data_name"],'r') as file:

    data_smiles = [line.strip() for line in file.readlines()]





input_size = embedd_size = len(SMILES.vocab.stoi.items())


dataset_smiles = infDENSDataset(data_smiles, SMILES)
dataloader = DataLoader(dataset_smiles, 
                        batch_size=config["inference_BATCH_SIZE"],
                        shuffle=False,
                        collate_fn=dataset_smiles.collate_fn)


     
        
aver_preds = torch.zeros(len(data_smiles), 1)
for checkpoint in config["model_names"]:
    
    model = RNN(input_size, embedd_size, config["hidden_size"], config["num_layers"], config["bidirect"]).to(device)
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    print(checkpoint.split('/')[-1], ' successfuly loaded')

    model.eval()

    for i in range(config["TTA_number"]):

        index = 0
        with torch.no_grad():
            for batch in dataloader:
                preds = model.forward(batch.to(device)).to('cpu')
                aver_preds[index:index+preds.shape[0],:] += preds

                index += preds.shape[0]


aver_preds /= (len(config["model_names"])*(config["TTA_number"]))




print(aver_preds.squeeze().tolist())
