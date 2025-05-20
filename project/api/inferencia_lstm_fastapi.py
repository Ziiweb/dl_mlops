import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import os
import wandb
from torch import nn
from fastapi import Body
from fastapi import Request

# === Config ===
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
latent_dim = 16
num_classes = 10

# === Clases necesarias ===
class Encoder(nn.Module):
    def __init__(self, z_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28 + num_classes, 400)
        self.fc2_mean = nn.Linear(400, z_dim)
        self.fc2_logvar = nn.Linear(400, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x, label):
        x = x.view(-1, 28 * 28)
        x_label = torch.cat([x, label], dim=-1)
        x = self.relu(self.fc1(x_label))
        return self.fc2_mean(x), self.fc2_logvar(x)

class Decoder(nn.Module):
    def __init__(self, z_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim + num_classes, 400)
        self.fc2 = nn.Linear(400, 28 * 28)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, label):
        z_label = torch.cat([z, label], dim=-1)
        x = self.relu(self.fc1(z_label))
        x = self.fc2(x)
        return self.sigmoid(x).view(-1, 28, 28)

class CVAE(nn.Module):
    def __init__(self, z_dim):
        super(CVAE, self).__init__()
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, label):
        z_mean, z_logvar = self.encoder(x, label)
        z = self.reparameterize(z_mean, z_logvar)
        return self.decoder(z, label), z_mean, z_logvar

# === FastAPI setup ===
app = FastAPI()

class Input(BaseModel):
    label: int  # número del 0 al 9

# === Carga de modelo desde wandb ===
# El token para el ___LOGIN___ se puede estar cogiendo de el archivo "Dockerfile", el cual hace referencia a un
# archivo .env donde esta WANDB_API_KEY, PERO OJO no estoy seguro de que esto sea así, a lo mejor se esta cogiendo
# porque he hecho lo del "export" en la terminal.
api = wandb.Api()
artifact_path = os.getenv("WANDB_ARTIFACT_PATH", "javiergarpe1979-upm/dl_mlops/cvae_model:latest")
artifact = api.artifact(artifact_path, type="model")
artifact_dir = artifact.download()
model_path = os.path.join(artifact_dir, "cvae_model_state_dict.pth")

model = CVAE(z_dim=latent_dim).to(DEVICE)
state_dict = torch.load(model_path, map_location=DEVICE)
model.load_state_dict(state_dict)
model.eval()

@app.post("/predict")
async def predict(request: Request):
    try:
        json_data = await request.json()
        print("Recibido:", json_data)

        label = json_data["label"]
        if not (0 <= label < num_classes):
            raise ValueError("Label fuera de rango (0-9).")

        label_tensor = torch.tensor([label], device=DEVICE)
        label_one_hot = torch.zeros(1, num_classes, device=DEVICE).scatter_(1, label_tensor.view(-1, 1), 1)
        z = torch.randn(1, latent_dim, device=DEVICE)

        with torch.no_grad():
            generated_image = model.decoder(z, label_one_hot).cpu().numpy()

        return {
            "label": label,
            "image": generated_image[0].tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok"}