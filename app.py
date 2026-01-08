import streamlit as st
import torch
from torch.utils.data import DataLoader
from functions import load_config, load_dict
from data_structures import infDENSDataset
from model import RNN
from rdkit import Chem

device = torch.device("cpu")

config = load_config("config.yaml")
SMILES = load_dict(config["tokens_dict_name"])

input_size = embedd_size = len(SMILES.vocab.stoi.items())

@st.cache_resource
def load_models():
    models = []
    for checkpoint in config["model_names"]:
        model = RNN(
            input_size, embedd_size,
            config["hidden_size"], config["num_layers"],
            device, config["bidirect"]
        ).to(device)
        model.load_state_dict(
            torch.load(checkpoint, map_location=device)['state_dict']
        )
        model.eval()
        models.append(model)
    return models

models = load_models()


st.title("Crystal Density Predictor")
st.write("Enter one or more SMILES strings (one per line) to predict crystal density.")

tta_number = st.slider(
    "Test-Time Augmentation (number of random SMILES)",
    min_value=1,
    max_value=50,
    value=config["TTA_number"],
    step=1
)

if tta_number > 20:
    st.info("High TTA values improve stability but increase inference time.")


example_smiles = """c1ccc2cc3ccccc3cc2c1
Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]
c1(Cl)c(Cl)c(Cl)c(Cl)c(Cl)c1Cl
"""

if st.button("🧪 Load example SMILES"):
    st.session_state["smiles_input"] = example_smiles

smiles_input = st.text_area(
    "SMILES strings",
    height=200,
    key="smiles_input"
)

if st.button("Predict"):
    if not smiles_input.strip():
        st.warning("Please enter at least one SMILES string.")
    else:
        data_smiles = [line.strip() for line in smiles_input.split("\n") if line.strip()]

        invalid_smiles = []
        valid_smiles = []

        for smi in data_smiles:
            mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid_smiles.append(smi)
        else:
            valid_smiles.append(smi)

        if len(valid_smiles) < len(data_smiles):
            st.warning(f"⚠️ {len(data_smiles) - len(valid_smiles)} invalid SMILES were ignored.")
        
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
        preds = aver_preds.view(-1).tolist()

        for smi, pred in zip(data_smiles, preds):
            st.code(f"SMILES: {smi}\nPredicted density: {pred:.4f}", language="text")

