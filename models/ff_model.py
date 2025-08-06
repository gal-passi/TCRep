import torch
import torch.nn as nn
import numpy as np


class FeedForwardClassifier(nn.Module):
    def __init__(self, max_seq_len=18, num_classes=2):
        super(FeedForwardClassifier, self).__init__()

        self.max_seq_len = max_seq_len
        self.amino_acids = "ACDEFGHIKLMNPQRSTVWY"  # 20 standard amino acids
        self.num_amino_acids = len(self.amino_acids)  # 20 unique amino acids

        # Define a simple feedforward network
        input_dim = self.num_amino_acids * self.max_seq_len  # One-hot encoding size
        hidden_dim = 128  # Hidden layer size (can be adjusted)

        self.fc_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)  # Output dim = 2 for binary classification
        )

    def one_hot_encode(self, sequences):
        """ Convert list/ndarray of protein sequences to one-hot encoding tensor """
        batch_size = len(sequences)
        one_hot = np.zeros((batch_size, self.max_seq_len, self.num_amino_acids), dtype=np.float32)

        # Fill the one-hot encoded array
        for i, seq in enumerate(sequences):
            for j, aa in enumerate(seq[:self.max_seq_len]):  # Truncate if longer
                if aa in self.amino_acids:
                    one_hot[i, j, self.amino_acids.index(aa)] = 1.0

        # Flatten the one-hot encoding for the linear layer
        device = self.fc_layers[0].weight.device
        return torch.tensor(one_hot.reshape(batch_size, -1), dtype=torch.float32, device=device)

    def forward(self, sequences):
        """
        sequences: numpy array of strings (protein sequences)
        """
        x = self.one_hot_encode(sequences)  # Convert sequences to one-hot
        logits = self.fc_layers(x)  # Pass through the network
        return logits  # No softmax, since we'll use CrossEntropyLoss
