"""Model Architectures - EXACT MATCH"""
import torch
import torch.nn as nn


class VAEEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)

    def forward(self, x):
        x = self.encoder(x).view(x.size(0), -1)
        return self.fc_mu(x), self.fc_logvar(x)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 256*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.decoder(self.fc(z).view(-1, 256, 4, 4))


class VAE(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.encoder = VAEEncoder(latent_dim)
        self.decoder = VAEDecoder(latent_dim)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5*logvar)
        return self.decoder(z), mu, logvar


class ColorizerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),      # 64->32
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),     # 32->16
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),    # 16->8
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, 2, 1),   # 8->4  (THIS WAS MISSING!)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.fc = nn.Linear(128*4*4, 32)

    def forward(self, x):
        x = self.encoder(x).view(x.size(0), -1)
        return self.fc(x)


class ColorizerDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 128*4*4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 128, 4, 4)
        return self.decoder(x)


class Colorizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ColorizerEncoder()
        self.decoder = ColorizerDecoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))
