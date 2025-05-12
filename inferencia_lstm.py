import torch
import pandas as pd
import numpy as np
import torch.nn as nn

# === Config ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
QUANTILES = [0.025, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
             0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975]
SEQ_LEN_IN = 24

# === Model class ===
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

        return torch.cat(outputs, dim=1)  # shape: (batch_size, SEQ_LEN_OUT, num_quantiles)

# === Load checkpoint ===
from torch.serialization import add_safe_globals
from sklearn.preprocessing._data import MinMaxScaler

# Allow MinMaxScaler to be unpickled
add_safe_globals([MinMaxScaler])

checkpoint = torch.load('quantile_lstm_checkpoint.pth', map_location=DEVICE, weights_only=False)

# Recreate model and optimizer
model = Seq2SeqLSTM_Quantile(input_size=6, hidden_size=64, output_size=1, quantiles=QUANTILES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load states
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
scaler_X = checkpoint['scaler_X']
scaler_y = checkpoint['scaler_y']

model.eval()
print(f"✅ Model loaded, ready for inference or resume from epoch {start_epoch}")

# === Load your data ===
df = pd.read_csv("train.csv")
features = ['feature_AA', 'feature_AB', 'feature_BA', 'feature_BB', 'feature_CA', 'feature_CB']
X_scaled = scaler_X.transform(df[features])

# === Create last input sequence ===
sample_input = X_scaled[-SEQ_LEN_IN:]  # shape: (24, 6)
sample_input = np.expand_dims(sample_input, axis=0)  # shape: (1, 24, 6)

sample_input_tensor = torch.tensor(sample_input, dtype=torch.float32).to(DEVICE)

# === Inference ===
with torch.no_grad():
    prediction = model(sample_input_tensor).cpu().numpy()  # (1, 1, num_quantiles)
    prediction_rescaled = scaler_y.inverse_transform(prediction[0])  # (1, num_quantiles)

    print("\n🌡️ Predicted Quantiles (Temperature °C):")
    for q, pred in zip(QUANTILES, prediction_rescaled[0]):  # SEQ_LEN_OUT=1
        print(f"Quantile {q}: {pred:.2f}")
