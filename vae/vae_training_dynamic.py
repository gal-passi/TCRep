import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
import os


class DynamicVAE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden_dims=None):
        super(DynamicVAE, self).__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        decoder_layers.append(nn.Sigmoid())  # Assuming normalized input data

        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


def vae_loss_function(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss with beta parameter for KL divergence weighting"""
    BCE = nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD, BCE, KLD


def sequences_to_onehot(sequences, max_length=None):
    """Convert amino acid sequences to one-hot encoding"""
    # Standard amino acids
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    aa_to_idx = {aa: idx for idx, aa in enumerate(amino_acids)}

    if max_length is None:
        max_length = max(len(seq) for seq in sequences)

    encoded_seqs = []
    for seq in sequences:
        # Pad or truncate sequence
        seq = seq[:max_length]
        seq = seq + 'A' * (max_length - len(seq))  # Pad with 'A'

        # One-hot encoding
        onehot = np.zeros((max_length, len(amino_acids)))
        for i, aa in enumerate(seq):
            if aa in aa_to_idx:
                onehot[i, aa_to_idx[aa]] = 1

        encoded_seqs.append(onehot.flatten())

    return np.array(encoded_seqs)


def plot_latent_space(vae, data_loader, device, save_path=None):
    """Plot the latent space representation"""
    vae.eval()
    latent_vectors = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch[0].to(device)
            mu, _ = vae.encode(x)
            latent_vectors.append(mu.cpu().numpy())

    latent_vectors = np.concatenate(latent_vectors, axis=0)

    # Use t-SNE for visualization if latent_dim > 2
    if latent_vectors.shape[1] > 2:
        tsne = TSNE(n_components=2, random_state=42)
        latent_2d = tsne.fit_transform(latent_vectors)
    else:
        latent_2d = latent_vectors

    plt.figure(figsize=(10, 8))
    plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.6, s=1)
    plt.title('VAE Latent Space Representation (t-SNE)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.colorbar()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_reconstruction_examples(vae, data_loader, device, num_examples=5, save_path=None):
    """Plot original vs reconstructed sequences"""
    vae.eval()

    with torch.no_grad():
        for batch in data_loader:
            x = batch[0][:num_examples].to(device)
            recon_x, _, _ = vae(x)

            # Convert back to sequence format for comparison
            x_np = x.cpu().numpy()
            recon_np = recon_x.cpu().numpy()

            fig, axes = plt.subplots(num_examples, 2, figsize=(15, 3 * num_examples))

            for i in range(num_examples):
                # Original
                axes[i, 0].imshow(x_np[i].reshape(-1, 20), aspect='auto', cmap='viridis')
                axes[i, 0].set_title(f'Original {i + 1}')
                axes[i, 0].set_ylabel('Position')

                # Reconstructed
                axes[i, 1].imshow(recon_np[i].reshape(-1, 20), aspect='auto', cmap='viridis')
                axes[i, 1].set_title(f'Reconstructed {i + 1}')
                axes[i, 1].set_ylabel('Position')

            plt.tight_layout()
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            break


def plot_loss_curves(train_losses, val_losses, save_path=None):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses['total'], label='Train Total')
    plt.plot(val_losses['total'], label='Val Total')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_losses['kld'], label='Train KLD')
    plt.plot(val_losses['kld'], label='Val KLD')
    plt.title('KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('KLD')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_new_sequences(vae, num_samples, device, max_length=None):
    """Generate new sequences from the VAE"""
    vae.eval()

    with torch.no_grad():
        # Sample from latent space
        z = torch.randn(num_samples, vae.latent_dim).to(device)
        generated = vae.decode(z).cpu().numpy()

        # Convert back to sequences (simplified - you might want to improve this)
        amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        sequences = []

        for sample in generated:
            if max_length is None:
                seq_len = len(sample) // 20
            else:
                seq_len = max_length

            onehot = sample.reshape(seq_len, 20)
            seq = ''
            for pos in onehot:
                aa_idx = np.argmax(pos)
                seq += amino_acids[aa_idx]
            sequences.append(seq.rstrip('A'))  # Remove padding

        return sequences


def train_and_inference_vae(args, dataset_loader, device):
    """Main function to train and run inference on VAE"""
    print("Starting VAE training and inference...")

    # Get positive training sequences
    train_pos_seqs, _, valid_pos_seqs, _, _, _ = dataset_loader.get_seqs()

    print(f"Training on {len(train_pos_seqs)} positive sequences")
    print(f"Validation on {len(valid_pos_seqs)} positive sequences")

    # Convert sequences to one-hot encoding
    max_seq_len = max(len(seq) for seq in train_pos_seqs + valid_pos_seqs)
    print(f"Maximum sequence length: {max_seq_len}")

    train_encoded = sequences_to_onehot(train_pos_seqs, max_seq_len)
    val_encoded = sequences_to_onehot(valid_pos_seqs, max_seq_len)

    # Normalize to [0, 1] range
    train_encoded = train_encoded.astype(np.float32)
    val_encoded = val_encoded.astype(np.float32)

    # Create data loaders
    train_dataset = TensorDataset(torch.tensor(train_encoded))
    val_dataset = TensorDataset(torch.tensor(val_encoded))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize VAE
    input_dim = train_encoded.shape[1]
    vae = DynamicVAE(
        input_dim=input_dim,
        latent_dim=args.vae_latent_dim,
        hidden_dims=args.vae_hidden_dims
    ).to(device)

    print(f"VAE architecture:")
    print(f"Input dim: {input_dim}")
    print(f"Latent dim: {args.vae_latent_dim}")
    print(f"Hidden dims: {args.vae_hidden_dims}")

    # Optimizer
    optimizer = optim.Adam(vae.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    # Training loop
    train_losses = {'total': [], 'bce': [], 'kld': []}
    val_losses = {'total': [], 'bce': [], 'kld': []}

    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 20

    for epoch in range(args.epochs):
        # Training
        vae.train()
        train_loss_epoch = {'total': 0, 'bce': 0, 'kld': 0}

        for batch in train_loader:
            x = batch[0].to(device)

            optimizer.zero_grad()
            recon_x, mu, logvar = vae(x)

            loss, bce, kld = vae_loss_function(recon_x, x, mu, logvar, beta=args.vae_beta)
            loss.backward()
            optimizer.step()

            train_loss_epoch['total'] += loss.item()
            train_loss_epoch['bce'] += bce.item()
            train_loss_epoch['kld'] += kld.item()

        # Validation
        vae.eval()
        val_loss_epoch = {'total': 0, 'bce': 0, 'kld': 0}

        with torch.no_grad():
            for batch in val_loader:
                x = batch[0].to(device)
                recon_x, mu, logvar = vae(x)

                loss, bce, kld = vae_loss_function(recon_x, x, mu, logvar, beta=args.vae_beta)

                val_loss_epoch['total'] += loss.item()
                val_loss_epoch['bce'] += bce.item()
                val_loss_epoch['kld'] += kld.item()

        # Average losses
        for key in train_loss_epoch:
            train_loss_epoch[key] /= len(train_loader)
            val_loss_epoch[key] /= len(val_loader)

        train_losses['total'].append(train_loss_epoch['total'])
        train_losses['bce'].append(train_loss_epoch['bce'])
        train_losses['kld'].append(train_loss_epoch['kld'])

        val_losses['total'].append(val_loss_epoch['total'])
        val_losses['bce'].append(val_loss_epoch['bce'])
        val_losses['kld'].append(val_loss_epoch['kld'])

        scheduler.step(val_loss_epoch['total'])

        # Early stopping
        if val_loss_epoch['total'] < best_val_loss:
            best_val_loss = val_loss_epoch['total']
            patience_counter = 0
            # Save best model
            torch.save(vae.state_dict(), f'cache/best_vae_{args.dataset_type}.pth')
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Train Loss: {train_loss_epoch["total"]:.4f}, '
                  f'Val Loss: {val_loss_epoch["total"]:.4f}, '
                  f'BCE: {val_loss_epoch["bce"]:.4f}, KLD: {val_loss_epoch["kld"]:.4f}')

        if patience_counter >= early_stopping_patience:
            print(f'Early stopping at epoch {epoch}')
            break

    # Load best model
    vae.load_state_dict(torch.load(f'cache/best_vae_{args.dataset_type}.pth'))

    # Create output directory
    output_dir = f'vae_results_{args.dataset_type}'
    os.makedirs(output_dir, exist_ok=True)

    print("\nTraining completed! Running inference and generating plots...")

    # Plot loss curves
    plot_loss_curves(train_losses, val_losses,
                     save_path=f'{output_dir}/loss_curves.png')

    # Plot latent space
    plot_latent_space(vae, val_loader, device,
                      save_path=f'{output_dir}/latent_space.png')

    # Plot reconstruction examples
    plot_reconstruction_examples(vae, val_loader, device, num_examples=5,
                                 save_path=f'{output_dir}/reconstructions.png')

    # Generate new sequences
    print("\nGenerating new sequences...")
    generated_seqs = generate_new_sequences(vae, num_samples=100, device=device,
                                            max_length=max_seq_len)

    print("Sample generated sequences:")
    for i, seq in enumerate(generated_seqs[:10]):
        print(f"{i + 1}: {seq}")

    # Save generated sequences
    with open(f'{output_dir}/generated_sequences.txt', 'w') as f:
        for i, seq in enumerate(generated_seqs):
            f.write(f">{i + 1}\n{seq}\n")

    # Compute and plot sequence length distributions
    train_lengths = [len(seq) for seq in train_pos_seqs]
    val_lengths = [len(seq) for seq in valid_pos_seqs]
    gen_lengths = [len(seq) for seq in generated_seqs]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.hist(train_lengths, bins=20, alpha=0.7, label='Training')
    plt.title('Training Sequence Lengths')
    plt.xlabel('Length')
    plt.ylabel('Count')

    plt.subplot(1, 3, 2)
    plt.hist(val_lengths, bins=20, alpha=0.7, label='Validation', color='orange')
    plt.title('Validation Sequence Lengths')
    plt.xlabel('Length')
    plt.ylabel('Count')

    plt.subplot(1, 3, 3)
    plt.hist(gen_lengths, bins=20, alpha=0.7, label='Generated', color='green')
    plt.title('Generated Sequence Lengths')
    plt.xlabel('Length')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/sequence_length_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nVAE training and inference completed!")
    print(f"Results saved in '{output_dir}/' directory")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Exit to avoid running the rest of the main script
    exit(0)