from random import choice
from torchtext import data
from functions import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = load_config("config.yaml")
SMILES = load_dict(config["tokens_dict_name"])


data_smiles = []
with open(config["inference_data_name"],'r') as file:

    for line in file:
        data_smiles.append([[i.split(',') for i in line.strip().split(',,,')[:-1]], float(line.strip().split(',,,')[-1])])






dataset_smiles = DENSDataset(data_smiles, SMILES)
dataloader = DataLoader(dataset_smiles, 
                              batch_size=config["inference_BATCH_SIZE"],
                              shuffle=False,
                              collate_fn=dataset_smiles.collate_fn)



embedd_size = len(SMILES.vocab.stoi.items())
input_size = embedd_size


aver_preds = torch.zeros(len(data_smiles), 1)

for checkpoint in config["model_names"]:
    
    model = RNN(input_size, embedd_size, config["hidden_size"], config["num_layers"], config["bidirect"]).to(device)
    model.load_state_dict(torch.load(checkpoint)['state_dict'])
    print(checkpoint, ' model successfuly loaded')

    model.eval()

    for i in range(config["TTA_number"]):

        index = 0
        with torch.no_grad():
            for batch in dataloader:
                smiles = batch[0]
                density = batch[1]

                preds = model.forward(smiles.to(device)).to('cpu')

                aver_preds[index:index+preds.shape[0],:] += preds

                index += preds.shape[0]


aver_preds /= (len(config["model_names"])*(config["TTA_number"]))


print(aver_preds)

