import streamlit as st
import torch
from torch.utils.data import DataLoader
from functions import load_config, load_dict
from data_structures import infDENSDataset
from model import RNN

device = torch.device("cpu")

config = load_config("config.yaml")
SMILES = load_dict(config["tokens_dict_name"])

input_size = embedd_size = len(SMILES.vocab.stoi.items())

models = []
for checkpoint in config["model_names"]:
    model = RNN(input_size, embedd_size, config["hidden_size"], config["num_layers"],
                device, config["bidirect"]).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device)['state_dict'])
    model.eval()
    models.append(model)
    print(checkpoint.split('/')[-1], ' loaded')


st.title("Crystal Density Predictor")
st.write("Enter one or more SMILES strings (one per line) to predict crystal density.")

smiles_input = st.text_area("SMILES strings", height=200)

if st.button("Predict"):
    if not smiles_input.strip():
        st.warning("Please enter at least one SMILES string.")
    else:
        data_smiles = [line.strip() for line in smiles_input.split("\n") if line.strip()]
        dataset_smiles = infDENSDataset(data_smiles, SMILES)
        dataloader = DataLoader(dataset_smiles,
                                batch_size=config["inference_BATCH_SIZE"],
                                shuffle=False,
                                collate_fn=dataset_smiles.collate_fn)

        aver_preds = torch.zeros(len(data_smiles), 1)
        for model in models:
            for tta in range(config["TTA_number"]):
                index = 0
                with torch.no_grad():
                    for batch in dataloader:
                        preds = model.forward(batch.to(device)).to('cpu')
                        aver_preds[index:index+preds.shape[0], :] += preds
                        index += preds.shape[0]

        aver_preds /= (len(models) * config["TTA_number"])

        for smi, pred in zip(data_smiles, preds):
            st.write(f"**SMILES:** {smi} → **Predicted density:** {pred:.4f}")
