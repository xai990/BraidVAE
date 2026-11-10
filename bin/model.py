import pandas as pd 
import torch 
import scanpy as sc
import anndata as ad
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional, Tuple

# ===================== VAE Model Architecture =====================

class Encoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 hidden_dims: List):
        super(Encoder, self).__init__()

        self.in_channels = in_channels

        modules = [
        nn.Sequential(nn.Dropout())]
        # Build encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_channels, h_dim),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                    )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self,
                in_channels: int,
                hidden_dims: int,
                out_dims: int):
        super(Decoder, self).__init__()

        self.in_h = nn.Linear(in_channels, hidden_dims)
        self.bn = nn.BatchNorm1d(hidden_dims)
        self.relu = nn.LeakyReLU()
        self.fc = nn.Linear(hidden_dims, out_dims)
    
    def forward(self, x):
        x = self.in_h(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc(x)
        return torch.tanh(x)


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder for learning separable embeddings.
    The model conditions on class labels to learn disentangled representations.
    """
    
    def __init__(self, 
                 input_dim, 
                 latent_dim= 32,
                 hidden_dims= [512, 256, 128],
                 n_classes= 0,
                 dropout= 0.2):
        """
        Args:
            input_dim: Number of input features (genes)
            latent_dim: Dimension of latent space
            hidden_dims: List of hidden layer dimensions
            n_classes: Number of classes for conditional VAE (0 for standard VAE)
            dropout: Dropout rate for regularization
        """
        super(ConditionalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_classes = n_classes
        self.is_conditional = n_classes > 0
        
        # Adjust input dimension if conditional
        encoder_input_dim = input_dim + n_classes if self.is_conditional else input_dim
        
        # ===== Encoder =====
            
        self.encoder = Encoder(encoder_input_dim, hidden_dims)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # ===== Decoder =====
        decoder_input_dim = latent_dim + n_classes if self.is_conditional else latent_dim
        modules = []
        in_ch = decoder_input_dim
        for h_dim in reversed(hidden_dims):
            modules.append(
                    nn.Sequential(
                        nn.Linear(in_ch, h_dim),
                        nn.BatchNorm1d(h_dim),
                        nn.LeakyReLU())
                    )
            in_ch = h_dim
        modules.append(nn.Sequential(nn.Linear(in_ch, input_dim), nn.Tanh()))
        self.decoder = nn.Sequential(*modules)
        
        # Additional components for better separation
        if self.is_conditional:
            # Adversarial classifier to enforce disentanglement
            self.adversarial_classifier = nn.Sequential(
                nn.Linear(latent_dim, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, n_classes)
            )
    
    def encode(self, x, c= None):
        """
        Encode input to latent space parameters
        
        Args:
            x: Input data
            c: One-hot encoded class labels (optional)
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        """
        if self.is_conditional and c is not None:
            x = torch.cat([x, c], dim=-1)
            
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling from latent distribution
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
        
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c= None):
        """
        Decode latent vector to reconstruction
        
        Args:
            z: Latent vector
            c: One-hot encoded class labels (optional)
        
        Returns:
            Reconstructed input
        """
        if self.is_conditional and c is not None:
            z = torch.cat([z, c], dim=-1)
            
        return self.decoder(z)
    
    def forward(self, x, c= None):
        """
        Forward pass through the VAE
        
        Args:
            x: Input data
            c: Class labels (will be one-hot encoded if provided)
        
        Returns:
            Tuple of (reconstruction, input, mu, logvar, latent_z, adversarial_pred)
        """
        # One-hot encode labels if provided
        c_onehot = None
        if self.is_conditional and c is not None:
            c_onehot = F.one_hot(c, num_classes=self.n_classes).float()
        
        # Encode
        mu, logvar = self.encode(x, c_onehot)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Decode
        recon = self.decode(z, c_onehot)
        
        # Adversarial classification (for disentanglement)
        adv_pred = None
        if self.is_conditional:
            adv_pred = self.adversarial_classifier(z)
        
        return recon, x, mu, logvar, z, adv_pred



# ===================== Loss Functions =====================
class VAELoss(nn.Module):
    def __init__(self, beta: float = 1.0): # Remove gamma
        super(VAELoss, self).__init__()
        self.beta = beta
        # Remove adversarial weight for now, until it's fully implemented
        
    def forward(self, 
                recon_x, 
                x, 
                mu, 
                logvar,
                adv_pred= None, # Will be ignored for now
                labels= None,
                training=True): 
        
        # Both terms are per-sample: sum over genes/latent dimensions,
        # then average over the batch.
        recons_loss = F.mse_loss(recon_x, x, reduction="sum") / x.shape[0]
        kl_loss = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(), dim=1
        ).mean()
        # applying the beta correction
        if training:
            # https://openreview.net/forum?id=Sy2fzU9gl
            total_loss = recons_loss + self.beta * kl_loss
        else:
            total_loss = recons_loss
        loss_dict = {'recon': recons_loss.item(), 'kl': kl_loss.item()}
        
        # (Adversarial loss removed until classifier is trained)
        
        return total_loss, loss_dict

def train_vae_with_test_set(
    model,
    train_loader,
    test_loader,
    n_epochs=100,
    learning_rate=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    early_stopping_patience=10,
    warmup_epochs=50,
    ckpt_path="results/best_vae_model.pth",
):
    """Backward-compatible entry point for the shared VAE training loop."""
    # Import here to avoid a circular import: train_utils imports VAELoss.
    from .train_utils import train_vae_with_test_set as train

    return train(
        model, train_loader, test_loader, n_epochs=n_epochs,
        learning_rate=learning_rate, device=device,
        early_stopping_patience=early_stopping_patience,
        warmup_epochs=warmup_epochs, ckpt_path=ckpt_path,
    )
