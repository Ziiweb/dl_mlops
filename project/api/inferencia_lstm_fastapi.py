import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import wandb


from sklearn.preprocessing import MinMaxScaler
from torch import nn


# Inicializar wandb (no inicia un run de tracking)
api = wandb.Api()

# Definir el nombre del artefacto y usuario/proyecto
artifact_path = os.getenv("WANDB_ARTIFACT_PATH", "javiergarpe1979-upm/dl_mlops/forecasting_temperature:v0")

# Descargar artefacto
artifact = api.artifact(artifact_path, type="model")
artifact_dir = artifact.download()


# Definir el esquema de datos para la petición de inferencia
class HouseData(BaseModel):
    feature_AA: float
    feature_AB: float
    feature_BA: float
    feature_BB: float
    feature_CA: float
    feature_CB: float



# === Config ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
QUANTILES = [0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
             0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975]
SEQ_LEN_IN = 24
FEATURES = ['feature_AA', 'feature_AB', 'feature_BA', 'feature_BB', 'feature_CA', 'feature_CB']

# === Model definition ===
class Seq2SeqLSTM_Quantile(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, quantiles, num_layers=1):
        super(Seq2SeqLSTM_Quantile, self).__init__()
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_size * self.num_quantiles, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size * self.num_quantiles)

    def forward(self, x):
        batch_size = x.size(0)
        _, (hidden, cell) = self.encoder(x)

        decoder_input = torch.zeros((batch_size, 1, self.num_quantiles), device=x.device)
        outputs = []

        for _ in range(1):  # SEQ_LEN_OUT = 1
            out, (hidden, cell) = self.decoder(decoder_input, (hidden, cell))
            pred = self.fc(out)
            outputs.append(pred)
            decoder_input = pred  # teacher forcing

        return torch.cat(outputs, dim=1)  # shape: (batch_size, 1, num_quantiles)

# === Load checkpoint ===
from torch.serialization import add_safe_globals
from sklearn.preprocessing._data import MinMaxScaler
add_safe_globals([MinMaxScaler])

checkpoint = torch.load('quantile_lstm_checkpoint.pth', map_location=DEVICE, weights_only=False)
model = Seq2SeqLSTM_Quantile(input_size=6, hidden_size=64, output_size=1, quantiles=QUANTILES).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

# === FastAPI setup ===
app = FastAPI()

class InputSequence(BaseModel):
    data: List[List[float]]  # Should be a 2D list of shape (24, 6)

@app.post("/predict")
def predict(input_seq: InputSequence):

    print(input_seq)
 
    try:
        input_array = np.array(input_seq.data)
        if input_array.shape != (SEQ_LEN_IN, 6):
            raise ValueError(f"Input shape must be (24, 6), but got {input_array.shape}")

        scaled_input = scaler_X.transform(input_array)
        tensor_input = torch.tensor(scaled_input[np.newaxis, :, :], dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            output = model(tensor_input).cpu().numpy()  # shape: (1, 1, num_quantiles)
            rescaled_output = scaler_y.inverse_transform(output[0])  # shape: (1, num_quantiles)

        return {
            "quantile_predictions": {
                str(q): float(p) for q, p in zip(QUANTILES, rescaled_output[0])
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
