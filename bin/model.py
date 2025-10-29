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
        
        # --- FIX 1: Use 'sum' and divide by batch size ---
        # This calculates loss "per-sample" not "per-gene"
        # recon_loss = F.mse_loss(recon_x, x, reduction='sum') / x.shape[0]
        recons_loss =F.mse_loss(recon_x, x)
        
        # --- FIX 2: Sum over latent dims, then mean over batch ---
        # This also calculates loss "per-sample"
        # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
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
    n_epochs= 100,
    learning_rate= 1e-3,
    device= 'cuda' if torch.cuda.is_available() else 'cpu',
    early_stopping_patience= 10
):
    """
    Modified training function that uses a separate test set instead of validation split.
    
    This function should replace the train_vae function in the main VAE script
    when using separate train/test files.
    """
   
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    criterion = VAELoss()
    n_epochs_warmup = 50
    history = {'train_loss': [], 'test_loss': [], 'train_recon': [], 
               'train_kl': [], 'test_recon': [], 'test_kl': []}
    
    best_test_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(n_epochs):
        
        # Calculate current beta for annealing
        if epoch < n_epochs_warmup:
            current_beta = epoch / n_epochs_warmup
        else:
            current_beta = 1.0
            
        criterion.beta = current_beta # Update beta in the loss function
        # Training phase
        model.train()
        train_loss = 0
        train_metrics = {'recon': 0, 'kl': 0, 'adversarial': 0}
        
        for batch in train_loader:
            if len(batch) == 2:
                data, labels = batch
                data, labels = data.to(device), labels.to(device)
            else:
                data = batch.to(device)
                labels = None
            
            optimizer.zero_grad()
            
            # Forward pass
            recon, input_data, mu, logvar, z, adv_pred = model(data, labels)
            
            # Calculate loss
            loss, loss_dict = criterion(recon, input_data, mu, logvar)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            for key in loss_dict:
                if key in train_metrics:
                    train_metrics[key] += loss_dict[key]
        
        # Average training metrics
        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
        
        history['train_loss'].append(train_loss)
        history['train_recon'].append(train_metrics['recon'])
        history['train_kl'].append(train_metrics['kl'])
        
        # Test phase
        model.eval()
        test_loss = 0
        test_metrics = {'recon': 0, 'kl': 0, 'adversarial': 0}
        
        with torch.no_grad():
            for batch in test_loader:
                if len(batch) == 2:
                    data, labels = batch
                    data, labels = data.to(device), labels.to(device)
                else:
                    data = batch.to(device)
                    labels = None
                
                recon, input_data, mu, logvar, z, adv_pred = model(data, labels)
                loss, loss_dict = criterion(recon, input_data, mu, logvar, adv_pred, labels)
                
                test_loss += loss.item()
                for key in loss_dict:
                    if key in test_metrics:
                        test_metrics[key] += loss_dict[key]
        
        test_loss /= len(test_loader)
        for key in test_metrics:
            test_metrics[key] /= len(test_loader)
        
        history['test_loss'].append(test_loss)
        history['test_recon'].append(test_metrics['recon'])
        history['test_kl'].append(test_metrics['kl'])
        
        # Learning rate scheduling
        scheduler.step(test_loss)
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'results/best_vae_model.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}]")
            print(f"  Train Loss: {train_loss:.4f} (Recon: {train_metrics['recon']:.4f}, KL: {train_metrics['kl']:.4f})")
            print(f"  Test Loss: {test_loss:.4f} (Recon: {test_metrics['recon']:.4f}, KL: {test_metrics['kl']:.4f})")
    
    # Load best model
    model.load_state_dict(torch.load('results/best_vae_model.pth'))
    
    return history
