import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from datasets.Curation import Study
from collections import Counter
import pickle
from itertools import combinations
from utils import pairwise_scores, levenshtein_dist_non_bin
from models.cvc_model import CVCClassifierModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import random
import argparse


STUDY_ID = 'PRJNA393498'  # Ankylosing Spondylitis study
STUDY_ID2 = 'immunoSEQ47'  # Hepatitis B virus study
STUDY_ID3 = 'immunoSEQ77'  # Rheumatoid arthritis study (plus healthy)
STUDY_ID4 = 'PRJNA258001'  # HIV study (plus healthy)
STUDY_ID5 = 'PRJNA390125'  # Only healthy study
STUDY_ID6 = 'PRJNA495603'  # Multiple sclerosis study (plus healthy)
STUDY_ID7 = 'PRJNA579190'  #  Multiple sclerosis study (plus healthy)
STUDY_ID8 = 'PRJNA280417'  #  Multiple sclerosis study
HEALTHY_STUDY_ID = STUDY_ID3  # ONLY CD8
HEALTHY_STUDY_ID2 = STUDY_ID4  # Both CD8 and CD4
HEALTHY_STUDY_ID3 = STUDY_ID5  # Larger both CD8 and CD4 (But fewer patients!)
HEALTHY_STUDY_ID4 = STUDY_ID6  # Other healthy study
HEALTHY_STUDY_ID5 = STUDY_ID7  # Other healthy study
STUDIES = [STUDY_ID, STUDY_ID2, STUDY_ID3, STUDY_ID4, STUDY_ID5, STUDY_ID6, STUDY_ID7]
VALID_SEQ_CACHE = "cache/valid_sequences"


# Custom dataset for imbalanced sampling
class ImbalancedProteinDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

        # Create indices for positive and negative samples
        self.pos_indices = torch.where(labels == 1)[0].tolist()
        self.neg_indices = torch.where(labels == 0)[0].tolist()

        self.num_pos = len(self.pos_indices)
        self.num_neg = len(self.neg_indices)

        print(f"Dataset created with {self.num_pos} positive and {self.num_neg} negative samples")

    def __len__(self):
        # Length is determined by positive samples (all will be used)
        return self.num_pos

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

    def get_pos_indices(self):
        return self.pos_indices

    def get_neg_indices(self):
        return self.neg_indices


# Custom batch sampler for imbalanced data
class ImbalancedBatchSampler(Sampler):
    def __init__(self, dataset, pos_per_batch=30, neg_per_batch=300):
        self.dataset = dataset
        self.pos_per_batch = pos_per_batch
        self.neg_per_batch = neg_per_batch

        self.pos_indices = dataset.get_pos_indices()
        self.neg_indices = dataset.get_neg_indices()

        # Calculate number of batches to cover all positive samples
        self.num_batches = (len(self.pos_indices) + self.pos_per_batch - 1) // self.pos_per_batch

    def __iter__(self):
        # Shuffle positive indices
        pos_indices = self.pos_indices.copy()
        random.shuffle(pos_indices)

        batches = []
        for i in range(self.num_batches):
            # Get positive indices for this batch
            start_idx = i * self.pos_per_batch
            end_idx = min(start_idx + self.pos_per_batch, len(pos_indices))
            pos_batch = pos_indices[start_idx:end_idx]

            # If we don't have enough positives to fill the batch, repeat some
            if len(pos_batch) < self.pos_per_batch:
                pos_batch = pos_batch + random.choices(pos_batch, k=self.pos_per_batch - len(pos_batch))

            # Sample negative indices for this batch
            neg_batch = random.choices(self.neg_indices, k=self.neg_per_batch)

            # Combine and return all indices for this batch
            batch = pos_batch + neg_batch
            random.shuffle(batch)  # Shuffle to mix positives and negatives
            batches.append(batch)

        # Shuffle the order of batches
        random.shuffle(batches)

        # Flatten the list of batches
        for batch in batches:
            yield batch
            # for idx in batch:
            #     yield idx

    def __len__(self):
        return self.num_batches * (self.pos_per_batch + self.neg_per_batch)


# Custom collate function
def custom_collate(batch):
    embeddings = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return embeddings, labels


# ProteinVAE class (unchanged)
class ProteinVAE(nn.Module):
    def __init__(self, input_dim=768, latent_dim=32, hidden_dims=[512, 256, 128],
                 device='cuda', dropout_rate=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.device = device

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        self.encoder = nn.Sequential(*encoder_layers).to(device)

        # Latent representation
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim).to(device)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim).to(device)

        # Decoder
        decoder_layers = []
        reversed_hidden_dims = hidden_dims[::-1]

        self.decoder_input = nn.Linear(latent_dim, reversed_hidden_dims[0]).to(device)

        prev_dim = reversed_hidden_dims[0]
        for h_dim in reversed_hidden_dims[1:]:
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers).to(device)

    def encode(self, x):
        x = self.encoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        x = self.decoder_input(z)
        x = F.relu(x)
        x = self.decoder(x)
        return x

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    def sample(self, n_samples=1):
        """Generate samples from the latent space"""
        z = torch.randn(n_samples, self.latent_dim).to(self.device)
        samples = self.decode(z)
        return samples

    def reconstruct(self, x):
        """Reconstruct input from latent space"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z)


# VAE loss function (unchanged)
def vae_loss_function(x_hat, x, mu, log_var, beta=1.0):
    """VAE loss function with reconstruction loss and KL divergence term"""
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')

    # KL divergence
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # Total loss
    loss = recon_loss + beta * kl_div

    return loss, recon_loss, kl_div


# Combined model (unchanged)
class CVCVAEModel(nn.Module):
    def __init__(self, base_model, vae_model, device='cuda'):
        super().__init__()
        self.base_model = base_model
        self.vae_model = vae_model
        self.device = device

    def encode_sequences(self, seqs):
        """Get embeddings from base model for sequences"""
        with torch.no_grad():
            embeddings = self.base_model.model(seqs)
        return embeddings

    def forward(self, seqs):
        """Forward pass through both models"""
        # Get embeddings from base model
        embeddings = self.encode_sequences(seqs)

        # Pass through VAE
        recon_embeddings, mu, log_var = self.vae_model(embeddings)

        return embeddings, recon_embeddings, mu, log_var


def extract_embeddings(base_model, seqs, embedding_file, batch_size=330):
    """
    Extract embeddings for a list of sequences using the base model
    """
    if os.path.exists(embedding_file):
        embeddings = torch.load(embedding_file)
        print(f"Loaded embeddings from {embedding_file}")
    else:
        print(f"Extracting embeddings and saving to {embedding_file}")
        embeddings = []
        # batches_iterator = tqdm(range(0, len(seqs), batch_size), desc="Extracting embeddings", total=len(seqs) // batch_size)
        for i in range(0, len(seqs), batch_size):
            batch_seqs = seqs[i:i + batch_size]
            with torch.no_grad():
                batch_embeddings = base_model.model(batch_seqs)
                embeddings.append(batch_embeddings.cpu())
        embeddings = torch.cat(embeddings, dim=0)
        torch.save(embeddings, embedding_file)

    return torch.tensor(embeddings, dtype=torch.float32)


# Modified training function
def train_vae(base_model, train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs, embedding_type,
              latent_dim=32, hidden_dims=[512, 256, 128], num_epochs=50,
              pos_per_batch=30, neg_per_batch=300, lr=1e-4, beta=1.0, device='cuda',
              checkpoint_path='cache/vae/model_saves/vae_checkpoint.pt', positive_weights=3.0):
    """
    Training function for VAE model with custom batch sampling
    """
    # Create datasets
    print("Preparing datasets...")
    os.makedirs('../cache/vae/model_saves', exist_ok=True)
    os.makedirs(f'cache/vae/{embedding_type}', exist_ok=True)

    # Extract embeddings
    print("Extracting embeddings for training set...")
    # Process positive sequences
    embeddings_file = f'cache/vae/{embedding_type}/train_pos_embeddings.pt'
    train_pos_embeddings = extract_embeddings(base_model, train_pos_seqs, embeddings_file)

    # Process negative sequences
    embeddings_file = f'cache/vae/{embedding_type}/train_neg_embeddings.pt'
    train_neg_embeddings = extract_embeddings(base_model, neg_seqs, embeddings_file)

    # Combine embeddings and create labels
    train_embeddings = torch.cat([train_pos_embeddings, train_neg_embeddings], dim=0)
    train_labels = torch.cat([
        torch.ones(len(train_pos_embeddings), dtype=torch.float32),
        torch.zeros(len(train_neg_embeddings), dtype=torch.float32)
    ])

    print("Extracting embeddings for validation set...")
    # Process validation positive sequences
    embeddings_file = f'cache/vae/{embedding_type}/valid_pos_embeddings.pt'
    valid_pos_embeddings = extract_embeddings(base_model, valid_pos_seqs, embeddings_file)

    # Process validation negative sequences
    embeddings_file = f'cache/vae/{embedding_type}/valid_neg_embeddings.pt'
    valid_neg_embeddings = extract_embeddings(base_model, valid_neg_seqs, embeddings_file)

    # Combine validation embeddings and create labels
    valid_embeddings = torch.cat([valid_pos_embeddings, valid_neg_embeddings], dim=0)
    valid_labels = torch.cat([
        torch.ones(len(valid_pos_embeddings), dtype=torch.float32),
        torch.zeros(len(valid_neg_embeddings), dtype=torch.float32)
    ])

    # Create custom datasets
    train_dataset = ImbalancedProteinDataset(train_embeddings, train_labels)
    valid_dataset = ImbalancedProteinDataset(valid_embeddings, valid_labels)

    # Create custom batch samplers
    train_batch_sampler = ImbalancedBatchSampler(
        train_dataset,
        pos_per_batch=pos_per_batch,
        neg_per_batch=neg_per_batch
    )

    # Create data loaders with custom batch samplers
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_batch_sampler,
        collate_fn=custom_collate
    )

    # For validation, we'll use a standard DataLoader
    valid_loader = DataLoader(valid_dataset, batch_size=pos_per_batch + neg_per_batch, shuffle=False)

    # Initialize VAE model
    vae_model = ProteinVAE(
        input_dim=768,
        latent_dim=latent_dim,
        hidden_dims=hidden_dims,
        device=device
    ).to(device)

    # Initialize optimizer
    optimizer = torch.optim.Adam(vae_model.parameters(), lr=lr)

    # Initialize lists to store losses
    train_losses = []
    valid_losses = []
    best_valid_loss = float('inf')

    # Training loop
    for epoch in range(num_epochs):
        # Training
        vae_model.train()
        train_loss = 0
        train_recon_loss = 0
        train_kl_loss = 0

        train_pos_errors = []
        train_neg_errors = []

        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # Process batches
        batch_count = 0
        train_bar = tqdm(train_loader, desc="Training", total=len(train_loader) // (pos_per_batch + neg_per_batch))
        for batch_embeddings, batch_labels in train_bar:
            batch_count += 1

            # Count actual positives and negatives in the batch
            num_pos = torch.sum(batch_labels == 1).item()
            num_neg = torch.sum(batch_labels == 0).item()

            # Forward pass
            recon_embeddings, mu, log_var = vae_model(batch_embeddings)

            # Calculate loss
            loss, recon, kl = vae_loss_function(recon_embeddings, batch_embeddings, mu, log_var, beta)

            # Calculate reconstruction error for tracking
            batch_recon_error = torch.mean((recon_embeddings - batch_embeddings) ** 2, dim=1).detach().cpu().numpy()

            # Separate positive and negative samples
            pos_indices = batch_labels == 1
            neg_indices = batch_labels == 0

            # Multiply reconstruction error of positive samples by positive_weights
            # batch_recon_error[pos_indices] *= positive_weights

            if pos_indices.any():
                train_pos_errors.extend(batch_recon_error[pos_indices.cpu().numpy()])
            if neg_indices.any():
                train_neg_errors.extend(batch_recon_error[neg_indices.cpu().numpy()])

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update losses
            train_loss += loss.item()
            train_recon_loss += recon.item()
            train_kl_loss += kl.item()

            # Update progress bar
            train_bar.set_postfix({
                "loss": loss.item() / len(batch_embeddings),
                "pos": num_pos,
                "neg": num_neg
            })

        # Adjust the loss by the number of samples processed
        train_samples = batch_count * (pos_per_batch + neg_per_batch)
        train_loss /= train_samples
        train_recon_loss /= train_samples
        train_kl_loss /= train_samples
        train_losses.append(train_loss)

        # Calculate average reconstruction errors
        avg_pos_recon = np.mean(train_pos_errors) if train_pos_errors else 0
        avg_neg_recon = np.mean(train_neg_errors) if train_neg_errors else 0

        # Calculate AUC for training data
        if train_pos_errors and train_neg_errors:
            train_all_errors = np.concatenate([train_pos_errors, train_neg_errors])
            train_all_labels = np.concatenate([
                np.ones(len(train_pos_errors)),
                np.zeros(len(train_neg_errors))
            ])
            train_auc = roc_auc_score(train_all_labels, -np.array(train_all_errors))
        else:
            train_auc = 0

        # Validation
        vae_model.eval()
        valid_loss = 0
        valid_recon_loss = 0
        valid_kl_loss = 0

        pos_recon_errors = []
        neg_recon_errors = []

        with torch.no_grad():
            valid_bar = tqdm(valid_loader, desc="Validation", total=len(valid_loader) // (pos_per_batch + neg_per_batch))
            for batch_embeddings, batch_labels in valid_bar:
                batch_embeddings = batch_embeddings.to(device)

                # Forward pass
                recon_embeddings, mu, log_var = vae_model(batch_embeddings)

                # Calculate loss
                loss, recon, kl = vae_loss_function(recon_embeddings, batch_embeddings, mu, log_var, beta)

                # Update losses
                valid_loss += loss.item()
                valid_recon_loss += recon.item()
                valid_kl_loss += kl.item()

                # Calculate reconstruction error for each sample
                batch_recon_error = torch.mean((recon_embeddings - batch_embeddings) ** 2, dim=1).cpu().numpy()

                # Separate positive and negative samples
                pos_indices = batch_labels == 1
                neg_indices = batch_labels == 0

                if pos_indices.any():
                    pos_recon_errors.extend(batch_recon_error[pos_indices.numpy()])
                if neg_indices.any():
                    neg_recon_errors.extend(batch_recon_error[neg_indices.numpy()])

                # Update progress bar
                valid_bar.set_postfix({"loss": loss.item() / len(batch_embeddings)})

        # Calculate average losses
        valid_loss /= len(valid_loader.dataset)
        valid_recon_loss /= len(valid_loader.dataset)
        valid_kl_loss /= len(valid_loader.dataset)
        valid_losses.append(valid_loss)

        # Calculate AUC for reconstruction error as classifier
        if pos_recon_errors and neg_recon_errors:
            all_errors = np.concatenate([pos_recon_errors, neg_recon_errors])
            all_labels = np.concatenate([
                np.ones(len(pos_recon_errors)),
                np.zeros(len(neg_recon_errors))
            ])

            # Lower reconstruction error should predict positive class (1)
            # So we negate the errors to get proper AUC
            valid_auc = roc_auc_score(all_labels, -np.array(all_errors))
        else:
            valid_auc = 0

        # Print losses and stats
        print(f"Train Loss: {train_loss:.6f} | Recon: {train_recon_loss:.6f} | KL: {train_kl_loss:.6f}")
        print(f"Train Recon Error: Pos={avg_pos_recon:.6f}, Neg={avg_neg_recon:.6f}, AUC={train_auc:.4f}")
        print(f"Valid Loss: {valid_loss:.6f} | Recon: {valid_recon_loss:.6f} | KL: {valid_kl_loss:.6f}")

        # Print average reconstruction errors
        avg_pos_error = np.mean(pos_recon_errors) if pos_recon_errors else 0
        avg_neg_error = np.mean(neg_recon_errors) if neg_recon_errors else 0
        print(f"Valid Recon Error: Pos={avg_pos_error:.6f}, Neg={avg_neg_error:.6f}, AUC={valid_auc:.4f}")

        # Save checkpoint if validation loss improves
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': vae_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
                'train_auc': train_auc,
                'valid_auc': valid_auc,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('VAE Loss During Training')
    plt.savefig('plots/vae/vae_training_loss.png')
    plt.close()

    # Load best model
    # checkpoint = torch.load(checkpoint_path)
    # vae_model.load_state_dict(checkpoint['model_state_dict'])

    return vae_model


def evaluate_vae_distributions(vae_model, base_model, pos_seqs, neg_seqs, embedding_type, batch_size=330, device='cuda'):
    """
    Evaluate how well the VAE separates positive and negative samples
    """
    os.makedirs('../plots/vae', exist_ok=True)
    vae_model.eval()

    # Extract embeddings
    print("Extracting embeddings for evaluation...")

    # Process positive sequences
    embeddings_file = f'cache/vae/{embedding_type}/valid_pos_embeddings.pt'
    pos_embeddings = extract_embeddings(base_model, pos_seqs, embeddings_file, batch_size)

    # Process negative sequences
    embeddings_file = f'cache/vae/{embedding_type}/valid_neg_embeddings.pt'
    neg_embeddings = extract_embeddings(base_model, neg_seqs, embeddings_file, batch_size)

    # Get reconstructions and calculate errors for positive samples
    pos_reconstructions = []
    pos_latent_vectors = []
    with torch.no_grad():
        for i in range(0, len(pos_embeddings), batch_size):
            batch_embeddings = pos_embeddings[i:i + batch_size].to(device)
            mu, log_var = vae_model.encode(batch_embeddings)
            z = vae_model.reparameterize(mu, log_var)
            recon = vae_model.decode(z)
            pos_reconstructions.append(recon.cpu())
            pos_latent_vectors.append(z.cpu())

    pos_reconstructions = torch.cat(pos_reconstructions, dim=0)
    pos_latent_vectors = torch.cat(pos_latent_vectors, dim=0)
    pos_errors = torch.mean((pos_reconstructions - pos_embeddings) ** 2, dim=1).numpy()

    # Get reconstructions and calculate errors for negative samples
    neg_reconstructions = []
    neg_latent_vectors = []
    with torch.no_grad():
        for i in range(0, len(neg_embeddings), batch_size):
            batch_embeddings = neg_embeddings[i:i + batch_size].to(device)
            mu, log_var = vae_model.encode(batch_embeddings)
            z = vae_model.reparameterize(mu, log_var)
            recon = vae_model.decode(z)
            neg_reconstructions.append(recon.cpu())
            neg_latent_vectors.append(z.cpu())

    neg_reconstructions = torch.cat(neg_reconstructions, dim=0)
    neg_latent_vectors = torch.cat(neg_latent_vectors, dim=0)
    neg_errors = torch.mean((neg_reconstructions - neg_embeddings) ** 2, dim=1).numpy()

    # Plot histograms of reconstruction errors
    plt.figure(figsize=(10, 6))
    plt.hist(pos_errors, bins=50, alpha=0.5, label=f'Positive Samples (n={len(pos_errors)})')
    plt.hist(neg_errors, bins=50, alpha=0.5, label=f'Negative Samples (n={len(neg_errors)})')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Reconstruction Errors')
    plt.legend()
    plt.savefig('plots/vae/reconstruction_error_distribution.png')
    plt.close()

    # Create 2D visualization of latent space using first two dimensions
    plt.figure(figsize=(10, 8))

    # Sample at most 2000 points of each class for clearer visualization
    max_points = 2000
    pos_indices = np.random.choice(len(pos_latent_vectors), min(max_points, len(pos_latent_vectors)), replace=False)
    neg_indices = np.random.choice(len(neg_latent_vectors), min(max_points, len(neg_latent_vectors)), replace=False)

    # Plot the latent vectors
    plt.scatter(pos_latent_vectors[pos_indices, 0], pos_latent_vectors[pos_indices, 1],
                alpha=0.5, label=f'Positive Samples (n={len(pos_latent_vectors)})')
    plt.scatter(neg_latent_vectors[neg_indices, 0], neg_latent_vectors[neg_indices, 1],
                alpha=0.5, label=f'Negative Samples (n={len(neg_latent_vectors)})')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.title('2D Visualization of Latent Space')
    plt.legend()
    plt.savefig('plots/vae/latent_space_visualization_naive.png')
    plt.close()

    # Combine positive and negative latent vectors for PCA and t-SNE
    combined_latent_vectors = np.concatenate([pos_latent_vectors[pos_indices], neg_latent_vectors[neg_indices]])
    combined_labels = np.concatenate([np.ones(len(pos_indices)), np.zeros(len(neg_indices))])

    # Plot PCA result
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(combined_latent_vectors)
    plt.figure(figsize=(10, 8))
    plt.scatter(pca_result[combined_labels == 1, 0], pca_result[combined_labels == 1, 1],
                alpha=0.5, label=f'Positive Samples (n={len(pos_indices)})')
    plt.scatter(pca_result[combined_labels == 0, 0], pca_result[combined_labels == 0, 1],
                alpha=0.5, label=f'Negative Samples (n={len(neg_indices)})')
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.title('PCA Visualization of Latent Space')
    plt.legend()
    plt.savefig('plots/vae/latent_space_visualization_pca.png')
    plt.close()

    # Plot t-SNE result
    tsne = TSNE(n_components=2, random_state=42)
    tsne_result = tsne.fit_transform(combined_latent_vectors)
    plt.figure(figsize=(10, 8))
    plt.scatter(tsne_result[combined_labels == 1, 0], tsne_result[combined_labels == 1, 1],
                alpha=0.5, label=f'Positive Samples (n={len(pos_indices)})')
    plt.scatter(tsne_result[combined_labels == 0, 0], tsne_result[combined_labels == 0, 1],
                alpha=0.5, label=f'Negative Samples (n={len(neg_indices)})')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.title('t-SNE Visualization of Latent Space')
    plt.legend()
    plt.savefig('plots/vae/latent_space_visualization_tsne.png')
    plt.close()

    # Calculate AUC
    all_errors = np.concatenate([pos_errors, neg_errors])
    all_labels = np.concatenate([np.ones(len(pos_errors)), np.zeros(len(neg_errors))])
    auc = roc_auc_score(all_labels, -all_errors)
    print(f"Reconstruction Error AUC: {auc:.4f}")

    # Print average reconstruction errors
    print(f"Average Positive Reconstruction Error: {np.mean(pos_errors):.6f}")
    print(f"Average Negative Reconstruction Error: {np.mean(neg_errors):.6f}")

    return {
        'auc': auc,
        'pos_errors': pos_errors,
        'neg_errors': neg_errors,
        'pos_latent_vectors': pos_latent_vectors.numpy(),
        'neg_latent_vectors': neg_latent_vectors.numpy()
    }

def get_all_usable_disease_data(disease='Multiple sclerosis'):
    studies = []
    if disease == 'Ankylosing spondylitis':
        study_ids = [STUDY_ID]
    elif disease == 'Hepatitis B virus':
        study_ids = [STUDY_ID2]
    elif disease == 'Rheumatoid arthritis':
        study_ids = [STUDY_ID3]
    elif disease == 'HIV':
        study_ids = [STUDY_ID4]
    elif disease == 'Multiple sclerosis':
        study_ids = [STUDY_ID6, STUDY_ID7, STUDY_ID8]
    else:
        raise ValueError(f"Invalid disease: {disease}")

    for study_id in study_ids:
        study = Study(study_id)
        usable_samples = study._samples['usable']
        df = study.read_sample(usable_samples)
        df = df[df['condition'] == disease]
        df['study_id'] = study_id
        studies.append(df)

    return pd.concat(studies, ignore_index=True)


def get_all_usable_healthy_data():
    healthy_study_ids = [HEALTHY_STUDY_ID, HEALTHY_STUDY_ID2, HEALTHY_STUDY_ID3, HEALTHY_STUDY_ID4, HEALTHY_STUDY_ID5]
    healthy_studies = []
    for study_id in healthy_study_ids:
        study = Study(study_id)
        usable_samples = study._samples['usable']
        df = study.read_sample(usable_samples)
        df = df[df['condition'] == 'Healthy']
        df['study_id'] = study_id
        healthy_studies.append(df)
    df_concat = pd.concat(healthy_studies, ignore_index=True)
    df_concat = df_concat.dropna(subset=['AASeq'])
    return df_concat


def get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=False):
    # Defining constants
    synapse_db_folder = "db/synapse_Mal_ID"
    synapse_metadata_file = os.path.join(synapse_db_folder, "metadata.tsv")

    # Define the filename based on the get_all flag
    file_suffix = "_all" if get_all else "_healthy_only"
    df_filename = os.path.join(synapse_db_folder, f"synapse_mal_id_dataframe{file_suffix}.pkl")
    # Check if the DataFrame is already saved
    if not to_recalculate:
        if os.path.exists(df_filename):
            # Load the DataFrame from file
            df = pd.read_pickle(df_filename)
            return df

    # Reading metadata
    synapse_metadata = pd.read_csv(synapse_metadata_file, sep='\t')

    # Reading and interpreting data files
    synapse_datafiles = [x for x in os.listdir(synapse_db_folder) if x.endswith(".bz2")]

    healthy_samples = []
    for datafile in tqdm(synapse_datafiles, desc="Processing data files", total=len(synapse_datafiles)):
        datafile_id = datafile.split("_")[-1][:-4]
        datafile_path = os.path.join(synapse_db_folder, datafile)

        # Reading metadata and data
        metadata = synapse_metadata[synapse_metadata['participant_label'] == datafile_id]
        condition = metadata['disease'].values[0]
        if get_all or 'Healthy' in condition:
            data = pd.read_csv(datafile_path, sep='\t', compression='bz2')
            data = data.loc[:, ['cdr3_seq_aa_q', 'participant_label', 'specimen_tissue']]
            data['cdr3_seq_aa_q'] = data['cdr3_seq_aa_q'].str.replace(' ', '')
            # add the condition to the metadata
            if 'Healthy' in condition:
                condition = 'Healthy'
            data['condition'] = condition
            healthy_samples.append(data)

    # need to make a df with: AASeq, patient_id, tissue, cell_type
    df = pd.concat(healthy_samples, ignore_index=True)
    df = df.dropna(subset=['cdr3_seq_aa_q'])
    # rename columns
    df = df.rename(columns={'cdr3_seq_aa_q': 'AASeq', 'participant_label': 'patient_id', 'specimen_tissue': 'tissue'})

    df = df[~df['AASeq'].str.contains('[^ACDEFGHIKLMNPQRSTVWY]', regex=True)]
    df['AASeq'] = 'C' + df['AASeq'] + 'F'

    # Save the DataFrame for future use
    df.to_pickle(df_filename)

    return df


def find_all_common_sequences(df, num_of_patients=3):
    # Step 1: Group by 'patient_id' and get unique AASeqs
    grouped = df.groupby('patient_id')['AASeq'].unique()

    # Step 2: Count occurrences of each AASeq across different patient groups
    aa_seq_counter = Counter()
    for aa_seqs in grouped:
        aa_seq_counter.update(aa_seqs)

    # Step 3: Filter AASeqs that appear in at least num_of_patients different patients
    valid_aa_seqs = {aa_seq for aa_seq, count in aa_seq_counter.items() if count >= num_of_patients}

    return valid_aa_seqs


def helper_function_common_aaseq_analysis(df, lev_dist_accept, only_valid=False, verbose=True):
    # Step 1: Create patient-wise groups
    seqs_by_patient = df.groupby('patient_id')['AASeq'].unique().to_dict()

    # Step 2: Compare sequences across different patients
    valid_sequences = set()
    iterator = list(combinations(seqs_by_patient.items(), 2))
    if verbose and len(iterator) > 1:
        iterator = tqdm(iterator, total=len(iterator))
    for (pid1, seqs1), (pid2, seqs2) in iterator:
        # Sort seqs1 by length
        seqs1_by_len = {}
        for seq in seqs1:
            seq_len = len(seq)
            if seq_len not in seqs1_by_len:
                seqs1_by_len[seq_len] = []
            seqs1_by_len[seq_len].append(seq)

        # Convert lists to numpy arrays for efficiency
        seqs1_by_len = {k: np.array(v, dtype=object) for k, v in seqs1_by_len.items()}

        # Sort seqs2 by length for efficient filtering
        seqs2_by_len = np.array(sorted(seqs2, key=len), dtype=object)
        seqs2_lens = np.array([len(seq) for seq in seqs2_by_len])

        # Iterate over length groups in seqs1
        inner_itter = list(seqs1_by_len.items())
        if verbose:
            inner_itter = tqdm(inner_itter, total=len(inner_itter))
        for length, group1 in inner_itter:
            # Select seqs2 that are in the range [length-1, length+1] (or +- lev_dist_accept)
            min_len, max_len = length - lev_dist_accept, length + lev_dist_accept
            mask = (seqs2_lens >= min_len) & (seqs2_lens <= max_len)
            group2 = seqs2_by_len[mask]

            if len(group2) > 0:
                # Compute pairwise identity matrix
                pwc_mat = pairwise_scores(group1, group2, score=levenshtein_dist_non_bin)
                sim_inxs = np.where(pwc_mat <= lev_dist_accept)

                # Add matching sequences with patient_id to valid set
                for x, y in zip(sim_inxs[0], sim_inxs[1]):
                    if only_valid:
                        valid_sequences.add(group1[x])
                        valid_sequences.add(group2[y])
                    else:
                        valid_sequences.add((group1[x], pid1, pid2))  # Tuple (seq, originating pid x2)
                        valid_sequences.add((group2[y], pid1, pid2))  # Tuple (seq, originating pid x2)

    if only_valid:
        return valid_sequences

    # Step 3: Calculate the mean number of common sequences and percentages
    unique_patient_ids = df["patient_id"].unique()
    masks = []
    for patient in unique_patient_ids:
        # Get sequences that belong to the current patient
        patient_seqs = set(df.loc[df["patient_id"] == patient, "AASeq"])

        # Create a mask for sequences
        mask = np.array([1 if seq[0] in patient_seqs else 0 for seq in valid_sequences])
        masks.append(mask)

    # Convert to ndarray
    masks = np.array(masks)  # Shape: (num_unique_patients, len(sequences))
    return masks


def process_in_batches(df, all_common_seqs, batch_size, lev_dist_accept):
    all_common_seqs = list(all_common_seqs)
    # Split all_common_seqs into batches
    all_common_seqs_batches = [all_common_seqs[i:i + batch_size] for i in range(0, len(all_common_seqs), batch_size)]

    valid_seqs_set = set()  # Use a set to store valid sequences across batches

    for batch in tqdm(all_common_seqs_batches):
        df_new = df.copy()

        # Flag 'valid' for sequences in the current batch
        df_new['patient_id'] = df_new['AASeq'].apply(lambda x: 'valid' if x in batch else 'all')

        # Separate valid and all sequences
        valid_df = df_new[df_new['patient_id'] == 'valid'].drop_duplicates(subset='AASeq')
        all_df = df_new[df_new['patient_id'] == 'all'].drop_duplicates(subset='AASeq')

        # Concatenate and reset index
        df_combined = pd.concat([valid_df, all_df]).reset_index(drop=True)

        # Apply the analysis function to the valid sequences
        valid_seqs_batch = helper_function_common_aaseq_analysis(df_combined, lev_dist_accept, only_valid=True, verbose=False)

        # Accumulate valid sequences from this batch (as a set)
        valid_seqs_set.update(valid_seqs_batch)

    # The result is a set of valid sequences
    return valid_seqs_set


def calculate_valid_near_sequences(df, save_name, lev_dist_accept=1, num_of_patients=3, all_common_seqs=None):
    save_folder = "cache/valid_sequences/multiple_sclerosis"
    save_file = os.path.join(save_folder, f"{save_name}_valid_seqs_dist_{lev_dist_accept}.pkl")
    if not os.path.exists(save_file):
        if all_common_seqs is None:
            all_common_seqs = find_all_common_sequences(df, num_of_patients=num_of_patients)
        valid_seqs = process_in_batches(df, all_common_seqs, 512, lev_dist_accept)
        os.makedirs(save_folder, exist_ok=True)
        with open(save_file, "wb") as f:
            pickle.dump(valid_seqs, f)
    else:
        with open(save_file, "rb") as f:
            valid_seqs = pickle.load(f)
    return valid_seqs


def generate_neighbors(sequences, valid_letters):
    valid_letters = set(valid_letters)  # Ensure valid letters are a set for quick lookup
    neighbor_set = set(sequences)  # Start with the original sequences

    for seq in sequences:
        seq_len = len(seq)

        # Generate substitutions
        for i in range(seq_len):
            for letter in valid_letters:
                if seq[i] != letter:  # Avoid replacing with the same letter
                    neighbor_set.add(seq[:i] + letter + seq[i + 1:])

        # Generate insertions
        for i in range(seq_len + 1):
            for letter in valid_letters:
                neighbor_set.add(seq[:i] + letter + seq[i:])

        # Generate deletions
        if seq_len > 1:  # Ensure we don't delete the only character
            for i in range(seq_len):
                neighbor_set.add(seq[:i] + seq[i + 1:])

    return neighbor_set


def get_data(dataset_type):
    # Load studies
    disease = 'Multiple sclerosis'
    df = get_all_usable_disease_data(disease=disease)
    df_h = get_all_usable_healthy_data()

    cell_type = ['DC8', 'CD4', 'ALL'][2]
    if cell_type != 'ALL' and dataset_type == 'ms':
        # reading blood samples
        df_bld = df[df['cell_type'] == cell_type]
        # reading healthy study:
        df_hlt = df_h[df_h['cell_type'] == cell_type]
    else:
        if dataset_type == 'ms':
            df_bld, df_hlt = df, df_h
        elif dataset_type == 'article':
            df_article = get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=True)
            # TODO: Consider adding the other healthy dataset to the article healthy dataset!
            df_bld = df_article[df_article["condition"] == "T1D"]
            df_hlt = df_article[df_article["condition"] == "Healthy"]
        else:
            raise ValueError("Invalid dataset type")

    def get_positive_negative(num_of_patients=3):
        all_common_seqs = find_all_common_sequences(df_bld, num_of_patients=num_of_patients)
        valid_seqs_healthy = find_all_common_sequences(df_hlt, num_of_patients=num_of_patients)
        all_common_seqs = all_common_seqs - valid_seqs_healthy
        # choosing valid samples according to their re-occurrence in different patients and a given distance
        valid_seqs_disease = calculate_valid_near_sequences(df_bld,
                                                            save_name=f'disease_{dataset_type}_{cell_type}_neighbours{num_of_patients}',
                                                            lev_dist_accept=1,
                                                            num_of_patients=num_of_patients,
                                                            all_common_seqs=all_common_seqs)

        positive_seqs = set(valid_seqs_disease)
        print(f"Valid Disease Sequence (num of common = {num_of_patients}): {len(positive_seqs)}")

        # Extract valid letters
        valid_letters = set(''.join(valid_seqs_healthy))
        # Group healthy sequences by length
        length_groups = {}
        for seq in valid_seqs_healthy:
            length_groups.setdefault(len(seq), set()).add(seq)
        # Process each length group separately
        for seq_len, seq_group in length_groups.items():
            # Generate neighbors for this group
            neighbors = generate_neighbors(seq_group, valid_letters)
            # Remove neighbors from positive_seqs immediately
            positive_seqs -= neighbors  # This prevents storing all neighbors
        negative_seqs = set(valid_seqs_healthy)  # Negative sequences remain unchanged
        return positive_seqs, negative_seqs

    positive_seqs, negative_seqs = get_positive_negative(num_of_patients=3)
    all_common_seqs = find_all_common_sequences(df_bld, num_of_patients=3)
    valid_seqs_healthy = find_all_common_sequences(df_hlt, num_of_patients=3)
    all_common_seqs = all_common_seqs - valid_seqs_healthy
    positive_seqs.update(all_common_seqs)

    # make list and sort
    positive_seqs = list(positive_seqs)
    positive_seqs.sort()
    np.random.seed(42)
    np.random.shuffle(positive_seqs)

    # getting patient id masks in order to do k-fold by patient (according to synovial samples)
    unique_patient_ids = df_bld["patient_id"].unique()
    unique_patient_ids = np.random.permutation(unique_patient_ids)
    masks = []
    for patient in unique_patient_ids:
        # Get sequences that belong to the current patient
        patient_seqs = set(df_bld.loc[df_bld["patient_id"] == patient, "AASeq"])
        # Create a mask for sequences
        mask = np.array([1 if seq in patient_seqs else 0 for seq in positive_seqs])
        if 1 in mask:
            masks.append(mask)
    # Convert to ndarray
    patient_id_masks = np.array(masks)  # Shape: (num_unique_patients, len(positive_seqs))

    # pick index of 10 unique patients from unique_patient_ids as test patients and the rest as train patients
    num_test_patients = 8
    test_patient_ids = unique_patient_ids[:num_test_patients]
    test_patient_ids, valid_patient_ids = test_patient_ids[:num_test_patients // 2], test_patient_ids[
                                                                                     num_test_patients // 2:]
    train_patient_ids = unique_patient_ids[num_test_patients:]
    # now translate back to the inds according to unique_patient_ids
    test_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in test_patient_ids])
    valid_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in valid_patient_ids])
    train_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in train_patient_ids])

    # Get the masks for the test and train patients
    test_masks = patient_id_masks[test_patient_inds]
    valid_masks = patient_id_masks[valid_patient_inds]
    train_masks = patient_id_masks[train_patient_inds]
    test_inds = test_masks.any(axis=0)
    valid_inds = valid_masks.any(axis=0)
    valid_test_inds = test_inds & valid_inds
    if sum(valid_inds) > sum(test_inds):
        test_inds = test_inds | valid_test_inds
        valid_inds = valid_inds & ~valid_test_inds
    else:
        valid_inds = valid_inds | valid_test_inds
        test_inds = test_inds & ~valid_test_inds
    train_inds = (~test_inds) & (~valid_inds)
    print(f"Number of Positive Sequences in General: {len(positive_seqs)}")
    print(
        f"Number of Positive Sequences in Test: {sum(test_inds)}, Percentage: {sum(test_inds) / len(positive_seqs) * 100:.2f}%")
    print(
        f"Number of Positive Sequences in Valid: {sum(valid_inds)}, Percentage: {sum(valid_inds) / len(positive_seqs) * 100:.2f}%")
    print(
        f"Number of Positive Sequences in Train: {sum(train_inds)}, Percentage: {sum(train_inds) / len(positive_seqs) * 100:.2f}%\n")

    # Get the positive sequences for the test and train sets
    test_pos_seqs = np.array(positive_seqs)[test_inds]
    valid_pos_seqs = np.array(positive_seqs)[valid_inds]
    train_pos_seqs = np.array(positive_seqs)[train_inds]
    # Get the negative sequences
    neg_seqs = df_bld[df_bld['patient_id'].isin(train_patient_ids)]['AASeq'].unique()
    test_neg_seqs = df_bld[df_bld['patient_id'].isin(test_patient_ids)]['AASeq'].unique()
    valid_neg_seqs = df_bld[df_bld['patient_id'].isin(valid_patient_ids)]['AASeq'].unique()
    # Get all sequences that are in valid_neg_seqs and valid_neg_seqs
    valid_test_neg_seqs = np.array(list(set(test_neg_seqs) & set(valid_neg_seqs)))
    if len(test_neg_seqs) < len(valid_neg_seqs):
        # remove valid_test_neg_seqs from valid_neg_seqs
        valid_neg_seqs = valid_neg_seqs[~np.isin(valid_neg_seqs, valid_test_neg_seqs)]
    else:
        # remove valid_test_neg_seqs from test_neg_seqs
        test_neg_seqs = test_neg_seqs[~np.isin(test_neg_seqs, valid_test_neg_seqs)]
    # Remove all valid_neg_seqs and test_neg_seqs sequences from the negative sequences
    neg_seqs = np.array(list(set(neg_seqs) - set(np.concatenate((valid_neg_seqs, test_neg_seqs)))))

    return train_pos_seqs, valid_pos_seqs, test_pos_seqs, neg_seqs, valid_neg_seqs, test_neg_seqs


def one_hot_encode_protein(sequences, max_len=None, embed_size=768):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # Standard 20 amino acids
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}

    # Determine the max sequence length if not provided
    if max_len is None:
        max_len = max(len(seq) for seq in sequences)

    one_hot_dim = len(amino_acids)
    encoded_seqs = np.zeros((len(sequences), max_len, one_hot_dim), dtype=np.float32)

    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq[:max_len]):
            if aa in aa_to_idx:
                encoded_seqs[i, j, aa_to_idx[aa]] = 1.0

    # Flatten the one-hot encoding for each sequence
    flattened_encodings = encoded_seqs.reshape(len(sequences), -1)

    # Ensure embeddings reach size 768 by padding with zeros
    if flattened_encodings.shape[1] < embed_size:
        pad_width = embed_size - flattened_encodings.shape[1]
        padded_encodings = np.pad(flattened_encodings, ((0, 0), (0, pad_width)), mode='constant')
    else:
        padded_encodings = flattened_encodings[:, :embed_size]  # Truncate if needed

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.tensor(padded_encodings, device=device)


def main():
    embedding_types = ['one_hot', 'cvc']
    dataset_types = ['ms', 'article']
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_type', type=str, choices=embedding_types, default='cvc', help='Type of embeddings to use')
    parser.add_argument('--dataset_type', type=str, choices=dataset_types, default='ms', help='Type of the dataset to run on')
    args = parser.parse_args()

    embedding_type = args.embedding_type.lower()
    dataset_type = args.dataset_type.lower()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using embedding type: {embedding_type}")
    print(f"Using dataset type: {dataset_type}")
    print(f"Using device: {device}")

    np.random.seed(42)

    # Load data
    train_pos_seqs, valid_pos_seqs, test_pos_seqs, neg_seqs, valid_neg_seqs, test_neg_seqs = get_data(dataset_type)


    # Assume we have already trained the base model
    batch_size = 330
    ch_dropout = 0
    freeze_embed_model = False

    if embedding_type == 'one_hot':
        class OneHotModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = one_hot_encode_protein
        base_model = OneHotModel().to(device)
    else:
        base_model = CVCClassifierModel(batch_size=batch_size, ch_dropout=ch_dropout, freeze_embed_model=freeze_embed_model, device=device)
        model_save_path = "cache/models/cvc_loss-ce_entropy_dataset-ms_epochs-22_batch-330_ratio-10_weights-5.0_lr-0.0025_regcoef-0.3_freeze-False_criterion-True/model_epoch_21.pth"
        base_model.load_state_dict(torch.load(model_save_path, map_location=device))
        # freeze base_model.model parameters (the embedding model)
        for param in base_model.model.parameters():
            param.requires_grad = False
        base_model.model.eval()

    # Train VAE
    vae_model = train_vae(
        base_model=base_model,
        train_pos_seqs=train_pos_seqs,
        neg_seqs=neg_seqs,
        valid_pos_seqs=valid_pos_seqs,
        valid_neg_seqs=valid_neg_seqs,
        embedding_type=embedding_type,
        latent_dim=32,
        hidden_dims=[512, 396, 256, 128, 86],
        num_epochs=16,
        pos_per_batch = 30,
        neg_per_batch = 300,
        lr=1e-5,
        beta=1.0,
        device=device,
        checkpoint_path='../cache/vae/model_saves/vae_checkpoint.pt',
        positive_weights=3.0,
    )

    # Evaluate VAE
    results = evaluate_vae_distributions(
        vae_model=vae_model,
        base_model=base_model,
        pos_seqs=valid_pos_seqs,
        neg_seqs=valid_neg_seqs,
        embedding_type=embedding_type,
        batch_size=330,
        device=device
    )

    # Create combined model
    combined_model = CVCVAEModel(base_model, vae_model, device=device)

    # Save combined model
    torch.save(combined_model.state_dict(), '../cache/vae/model_saves/combined_model.pt')

    print("Training and evaluation complete!")


# TODO: Add weight to the loss of positives!
if __name__ == "__main__":
    main()
