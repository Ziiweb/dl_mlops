import torch
import torch.nn as nn

# Encoder
class Encoder(nn.Module):
    def __init__(self, z_dim, num_classes=10):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28 * 28 + num_classes, 400)
        self.fc2_mean = nn.Linear(400, z_dim)
        self.fc2_logvar = nn.Linear(400, z_dim)
        self.relu = nn.ReLU()

    def forward(self, x, label):
        x = x.view(-1, 28 * 28)
        x_label = torch.cat([x, label], dim=-1)
        x_label_relu = self.relu(self.fc1(x_label))
        z_mean = self.fc2_mean(x_label_relu)
        z_logvar = self.fc2_logvar(x_label_relu)
        return z_mean, z_logvar

# Decoder
class Decoder(nn.Module):
    def __init__(self, z_dim, num_classes=10):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(z_dim + num_classes, 400)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(400, 28 * 28)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, label):
        z_label = torch.cat([z, label], dim=-1)
        x = self.relu(self.fc1(z_label))
        x = self.fc2(x)
        return self.sigmoid(x).view(-1, 28, 28)

# CVAE
class CVAE(nn.Module):
    def __init__(self, z_dim, num_classes=10):
        super(CVAE, self).__init__()
        self.encoder = Encoder(z_dim, num_classes)
        self.decoder = Decoder(z_dim, num_classes)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, label):
        z_mean, z_logvar = self.encoder(x, label)
        z = self.reparameterize(z_mean, z_logvar)
        reconstructed_x = self.decoder(z, label)
        return reconstructed_x, z_mean, z_logvar
