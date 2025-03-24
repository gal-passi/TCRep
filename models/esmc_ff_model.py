import torch
import torch.nn as nn
import numpy as np
from embedding.embedding import embed_esmc


class ESMCFeedForwardClassifier(nn.Module):
    def __init__(self, num_classes=2, device="cpu"):
        super(ESMCFeedForwardClassifier, self).__init__()

        self.device = device

        self.embed_len = 960
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
        self.num_amino_acids = len(self.amino_acids)  # 20 unique amino acids

        # Define a simple feedforward network
        input_dim = self.embed_len
        hidden_dim = 128  # Hidden layer size (can be adjusted)

        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)  # Output dim = 2 for binary classification
        )

    def forward(self, sequences):
        """
        sequences: numpy array of strings (protein sequences)
        """
        x = torch.stack(embed_esmc(sequences, to_tqdm=False, use_pre_loaded=True)).to(self.device)  # Convert sequences to one-hot)
        logits = self.fc_layers(x)  # Pass through the network
        return logits  # No softmax, since we'll use CrossEntropyLoss
