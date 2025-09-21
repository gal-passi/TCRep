import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from utils.cache_handler import get_model_config_str
import pickle

CACHE_DIR = 'cache/vae_cache'
PLOTS_DIR = 'plots/vae'

class ControlVAE(nn.Module):
    """
    ControlVAE implementation for sequence generation
    Based on "ControlVAE: Model-Based Learning of Generative Controllers for Physics-Based Characters"
    """

    def __init__(self, input_dim, trained_model, model, latent_dim=64, hidden_dim=256, control_dim=32, max_seq_len=30, embedding_type='onehot'):
        super(ControlVAE, self).__init__()
        self.input_dim = input_dim
        self.trained_model = trained_model
        self.model = model
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.max_seq_len = max_seq_len
        self.embedding_type = embedding_type

        # Freeze the trained model parameters
        self.trained_model.eval()
        self.model.eval()

        # Amino acid vocabulary (20 standard amino acids + padding)
        self.vocab_size = 21
        self.aa_to_idx = {aa: i for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWYX')}  # X for padding
        self.idx_to_aa = {i: aa for aa, i in self.aa_to_idx.items()}

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )

        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Control network (maps latent + control to controlled latent)
        self.control_net = nn.Sequential(
            nn.Linear(latent_dim + control_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode_sequences(self, sequences):
        """Convert amino acid sequences to one-hot encoded tensors"""
        if self.embedding_type == 'onehot':
            batch_size = len(sequences)
            encoded = torch.zeros(batch_size, self.max_seq_len, self.vocab_size)

            for i, seq in enumerate(sequences):
                seq = seq[:self.max_seq_len]  # Truncate if too long
                for j, aa in enumerate(seq):
                    if aa in self.aa_to_idx:
                        encoded[i, j, self.aa_to_idx[aa]] = 1.0
                    else:
                        encoded[i, j, self.aa_to_idx['X']] = 1.0  # Use padding for unknown AA

            return encoded.view(batch_size, -1)  # Flatten
        elif self.embedding_type == 'trained_cvc':
            with torch.no_grad():
                embeddings = self.trained_model.get_embeddings(sequences).to(torch.float32)
            return embeddings
        elif self.embedding_type == 'untrained_cvc':
            with torch.no_grad():
                embeddings = self.model.get_embeddings(sequences).to(torch.float32)
            return embeddings
        else:
            raise ValueError(f"Unknown embedding type: {self.embedding_type}")

    def decode_to_sequences(self, x):
        """Convert one-hot encoded tensors back to amino acid sequences"""
        batch_size = x.shape[0]
        x = x.view(batch_size, self.max_seq_len, self.vocab_size)
        sequences = []

        for i in range(batch_size):
            seq = ""
            for j in range(self.max_seq_len):
                aa_idx = torch.argmax(x[i, j]).item()
                if aa_idx < len(self.idx_to_aa) and self.idx_to_aa[aa_idx] != 'X':
                    seq += self.idx_to_aa[aa_idx]
                else:
                    break  # Stop at padding
            sequences.append(seq)

        return sequences

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def control(self, z, c):
        """Apply control signal to latent representation"""
        z_c = torch.cat([z, c], dim=1)
        return self.control_net(z_c)

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x, c=None):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)

        if c is not None:
            z = self.control(z, c)

        x_recon = self.decode(z)
        return x_recon, mu, logvar, z


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss function with KL divergence"""
    # BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # return BCE + beta * KLD, BCE, KLD
    return recon_loss + beta * KLD, recon_loss, KLD


def train_controlvae(positive_sequences, valid_pos_seqs, valid_neg_seqs, args, device, trained_model, untrained_model, model_config_str):
    """
    Train ControlVAE on positive sequences
    """
    print("Starting ControlVAE training...")

    # Hyperparameters
    max_seq_len = max(len(seq) for seq in positive_sequences)
    max_seq_len = min(max_seq_len, 21)  # Cap at reasonable length
    latent_dim = 64
    control_dim = 32
    hidden_dim = 256
    batch_size = 512
    epochs = 20
    learning_rate = 0.00001
    beta = 0.5  # KL divergence weight
    embedding_type = ['onehot', 'trained_cvc', 'untrained_cvc'][1]  # Choose embedding type

    # Initialize model
    vocab_size = 21
    input_dim = max_seq_len * vocab_size if embedding_type == 'onehot' else 768
    model = ControlVAE(input_dim, trained_model, untrained_model, latent_dim, hidden_dim, control_dim, max_seq_len, embedding_type).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"Model parameters: input_dim={input_dim}, latent_dim={latent_dim}, max_seq_len={max_seq_len}")
    print(f"Training on {len(positive_sequences)} positive sequences")

    embedding_cache = os.path.join(CACHE_DIR, f'controlvae_cache/model_{model_config_str}/embedding_{embedding_type}.pkl')
    if os.path.exists(embedding_cache):
        print(f"Loading cached embeddings from {embedding_cache}")
        with open(embedding_cache, 'rb') as f:
            cached_data = pickle.load(f)
            positive_sequences = cached_data['positive_sequences']
            train_encoded = cached_data['train_encoded']
            valid_pos_seqs = cached_data['valid_pos_seqs']
            valid_encoded = cached_data['valid_encoded']
            valid_neg_seqs = cached_data['valid_neg_seqs']
            valid_neg_encoded = cached_data['valid_neg_encoded']
    else:
        # Prepare data
        train_encoded = model.encode_sequences(positive_sequences).to(device)
        valid_encoded = model.encode_sequences(valid_pos_seqs).to(device)
        valid_neg_seqs = np.random.choice(valid_neg_seqs, size=min(len(valid_neg_seqs), len(valid_pos_seqs)), replace=False)
        valid_neg_encoded = model.encode_sequences(valid_neg_seqs).to(device)

        os.makedirs(os.path.dirname(embedding_cache), exist_ok=True)
        with open(embedding_cache, 'wb') as f:
            pickle.dump({
                'positive_sequences': positive_sequences,
                'train_encoded': train_encoded,
                'valid_pos_seqs': valid_pos_seqs,
                'valid_encoded': valid_encoded,
                'valid_neg_seqs': valid_neg_seqs,
                'valid_neg_encoded': valid_neg_encoded
            }, f)

    train_dataset = TensorDataset(train_encoded)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    train_losses = []
    valid_losses = []
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_bce = 0
        epoch_kld = 0
        epoch_num_samples = 0

        for batch_idx, (data,) in enumerate(train_loader):
            optimizer.zero_grad()

            # Forward pass
            recon_batch, mu, logvar, z = model(data)

            # Loss computation
            loss, bce, kld = vae_loss_function(recon_batch, data, mu, logvar, beta)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_num_samples += data.size(0)
            epoch_bce += bce.item()
            epoch_kld += kld.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_recon, val_mu, val_logvar, _ = model(valid_encoded)
            val_loss, val_bce, val_kld = vae_loss_function(val_recon, valid_encoded, val_mu, val_logvar, beta)
            # normalize val_loss  # TODO: Check if this effects the scheduler
            val_loss = val_loss / len(valid_encoded)
            valid_losses.append(val_loss.item())
        model.train()

        # Record losses
        avg_loss = epoch_loss / epoch_num_samples
        avg_bce = epoch_bce / epoch_num_samples
        avg_kld = epoch_kld / epoch_num_samples
        train_losses.append(avg_loss)

        scheduler.step(val_loss.item())

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(
                f'Epoch {epoch:3d}/{epochs}: Loss={avg_loss:.4f} (BCE={avg_bce:.4f}, KLD={avg_kld:.4f}), Val Loss={val_loss.item() / len(valid_encoded):.4f}')

    # Save model
    model_path = os.path.join(CACHE_DIR, f'controlvae_model_{args.dataset_type}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'args': args,
        'max_seq_len': max_seq_len,
        'latent_dim': latent_dim,
        'control_dim': control_dim,
        'train_losses': train_losses,
        'valid_losses': valid_losses
    }, model_path)
    print(f"Model saved to {model_path}")

    # Plot losses
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('ControlVAE Training Loss')
    plt.legend()
    plt.grid(True)

    # Generate and analyze latent representations
    model.eval()
    with torch.no_grad():
        # sample from train the same amount as there is in validation
        train_encoded = train_encoded[torch.randperm(train_encoded.size(0))[:len(valid_pos_seqs)]]
        train_mu, train_logvar = model.encode(train_encoded)
        valid_mu, valid_logvar = model.encode(valid_encoded)
        neg_mu, neg_logvar = model.encode(valid_neg_encoded)

        # Combine for visualization
        all_mu = torch.cat([train_mu, valid_mu, neg_mu], dim=0)
        labels = ['Train'] * len(train_mu) + ['Validation'] * len(valid_mu) + ['Negative'] * len(neg_mu)

        # PCA visualization
        plt.subplot(1, 2, 2)
        if latent_dim > 2:
            pca = PCA(n_components=2)
            mu_2d = pca.fit_transform(all_mu.cpu().numpy())
        else:
            mu_2d = all_mu.cpu().numpy()

        colors = []
        for l in labels:
            if l == 'Train':
                colors.append('blue')
            elif l == 'Validation':
                colors.append('green')
            else:
                colors.append('red')
        plt.scatter(mu_2d[:, 0], mu_2d[:, 1], c=colors, alpha=0.4, s=20)
        plt.xlabel('Latent Dim 1')
        plt.ylabel('Latent Dim 2')
        plt.title('Latent Space (PCA)')
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', label='Train', markerfacecolor='blue', markersize=8, alpha=0.6),
            Line2D([0], [0], marker='o', color='w', label='Validation', markerfacecolor='green', markersize=8, alpha=0.6),
            Line2D([0], [0], marker='o', color='w', label='Negative', markerfacecolor='red', markersize=8, alpha=0.6)
        ]
        plt.legend(handles=legend_elements, loc='best')
        # plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f'controlvae_training_and_pca_model_{model_config_str}_{args.dataset_type}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # TODO: Turned off inference for faster testing
    # Inference and generation
    # print("\nPerforming VAE inference...")
    # perform_vae_inference(model, positive_sequences, valid_pos_seqs, args, device)

    return model


def perform_vae_inference(model, train_pos_seqs, valid_pos_seqs, args, device):
    """
    Perform inference with trained ControlVAE
    """
    model.eval()

    with torch.no_grad():
        # Encode validation sequences
        valid_encoded = model.encode_sequences(valid_pos_seqs).to(device)
        valid_mu, valid_logvar = model.encode(valid_encoded)

        # 1. Reconstruction quality
        recon_valid, _, _, _ = model(valid_encoded)
        recon_seqs = model.decode_to_sequences(recon_valid.cpu())

        print("\nReconstruction Examples:")
        for i in range(min(5, len(valid_pos_seqs))):
            print(f"Original:      {valid_pos_seqs[i]}")
            print(f"Reconstructed: {recon_seqs[i]}")
            print()

        # 2. Generate new sequences by sampling from latent space
        print("Generating new sequences...")
        n_samples = 20
        z_samples = torch.randn(n_samples, model.latent_dim).to(device)
        generated = model.decode(z_samples)
        generated_seqs = model.decode_to_sequences(generated.cpu())

        print("\nGenerated Sequences:")
        for i, seq in enumerate(generated_seqs[:10]):
            print(f"Generated {i + 1}: {seq}")

        # 3. Controlled generation with different control signals
        print("\nControlled Generation:")
        base_z = torch.randn(5, model.latent_dim).to(device)

        # Try different control signals
        control_signals = [
            torch.zeros(5, model.control_dim).to(device),  # No control
            torch.ones(5, model.control_dim).to(device) * 0.5,  # Mild control
            torch.ones(5, model.control_dim).to(device) * 1.0,  # Strong control
        ]

        for c_idx, control in enumerate(control_signals):
            controlled_z = model.control(base_z, control)
            controlled_gen = model.decode(controlled_z)
            controlled_seqs = model.decode_to_sequences(controlled_gen.cpu())

            print(f"\nControl Signal {c_idx} (strength: {control[0, 0].item():.1f}):")
            for i, seq in enumerate(controlled_seqs):
                print(f"  {seq}")

        # 4. Latent space analysis
        print("\nAnalyzing latent space...")

        # Get latent representations for all training sequences
        train_encoded = model.encode_sequences(train_pos_seqs).to(device)
        train_mu, train_logvar = model.encode(train_encoded)

        # Combine train and validation
        all_mu = torch.cat([train_mu, valid_mu], dim=0)
        all_sequences = np.concatenate([train_pos_seqs, valid_pos_seqs])  #  train_pos_seqs + valid_pos_seqs
        labels = ['Train'] * len(train_pos_seqs) + ['Valid'] * len(valid_pos_seqs)

        # t-SNE visualization if we have enough samples
        if len(all_sequences) > 50:
            print("Creating t-SNE visualization...")
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_sequences) // 4))
            mu_tsne = tsne.fit_transform(all_mu.cpu().numpy())

            plt.figure(figsize=(12, 5))

            # t-SNE plot
            plt.subplot(1, 2, 1)
            colors = ['blue' if l == 'Train' else 'red' for l in labels]
            plt.scatter(mu_tsne[:, 0], mu_tsne[:, 1], c=colors, alpha=0.6, s=30)
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.title('Latent Space (t-SNE)')
            plt.legend(['Training', 'Validation'])
            plt.grid(True)

            # Sequence length distribution
            plt.subplot(1, 2, 2)
            train_lengths = [len(seq) for seq in train_pos_seqs]
            valid_lengths = [len(seq) for seq in valid_pos_seqs]

            plt.hist(train_lengths, alpha=0.7, label='Training', bins=20, color='blue')
            plt.hist(valid_lengths, alpha=0.7, label='Validation', bins=20, color='red')
            plt.xlabel('Sequence Length')
            plt.ylabel('Count')
            plt.title('Sequence Length Distribution')
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f'controlvae_inference_{args.dataset_type}.png'), dpi=300, bbox_inches='tight')
            plt.show()

        # 5. Sequence diversity analysis
        unique_generated = set(generated_seqs)
        unique_original = set(train_pos_seqs + valid_pos_seqs)
        overlap = unique_generated.intersection(unique_original)

        print(f"\nSequence Diversity Analysis:")
        print(f"Generated unique sequences: {len(unique_generated)}")
        print(f"Original unique sequences: {len(unique_original)}")
        print(f"Overlap: {len(overlap)} sequences")
        print(f"Novel sequences: {len(unique_generated) - len(overlap)}")

        # Length statistics
        gen_lengths = [len(seq) for seq in generated_seqs if seq]
        if gen_lengths:
            print(f"Generated sequence lengths: mean={np.mean(gen_lengths):.1f}, std={np.std(gen_lengths):.1f}")

        orig_lengths = [len(seq) for seq in train_pos_seqs]
        print(f"Original sequence lengths: mean={np.mean(orig_lengths):.1f}, std={np.std(orig_lengths):.1f}")


def run_vae_training_and_inference(args, dataset_loader, device, trained_model, model):
    """
    Main function to run VAE training and inference
    """
    model_config_str = get_model_config_str(args)
    print("=== Starting ControlVAE Training and Inference ===")

    # Get positive sequences from dataset loader
    train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs = dataset_loader.get_seqs()

    print(f"Dataset: {args.dataset_type}")
    print(f"Training positive sequences: {len(train_pos_seqs)}")
    print(f"Validation positive sequences: {len(valid_pos_seqs)}")
    print(f"Test positive sequences: {len(test_pos_seqs)}")

    # Train ControlVAE
    trained_vae = train_controlvae(train_pos_seqs, valid_pos_seqs, valid_neg_seqs, args, device, trained_model, model, model_config_str)

    print("=== ControlVAE Training and Inference Completed ===")
    exit(0)
