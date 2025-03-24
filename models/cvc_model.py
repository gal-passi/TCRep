import sys
sys.path.append('other_models/CVC')
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from itertools import zip_longest
from typing import *
from tqdm.auto import tqdm
from transformers import BertModel
import cvc.data_loader as dl
import cvc.featurization as ft
import warnings
from lab_notebooks.utils import TRANSFORMER
warnings.simplefilter("ignore", category=FutureWarning)


class CVCModel(nn.Module):
    def __init__(self, model_dir: str = TRANSFORMER, method: str = 'mean', device: str = 'cuda', batch_size: int = 1024):
        super().__init__()
        self.device = device
        self.model = BertModel.from_pretrained(model_dir, add_pooling_layer=method == "pool", output_hidden_states=True).to(device)
        self.tok = ft.get_pretrained_bert_tokenizer(model_dir)
        self.freeze_bert_layers()
        self.method = method  # Options: "mean", "max", "attn_mean", "cls", "pool"
        self.batch_size = batch_size

    def get_transformer_embeddings(
            self,
            seqs: Iterable[str],
            seq_pair: Optional[Iterable[str]] = None,
            *,
            layers: List[int] = [-1],
            batch_size: int = 256,
            pbar=False,
            max_len: int = 64,
    ):
        """
        Get the embeddings for the given sequences from the given layers.
        - `model`: Preloaded transformer model.
        - `tok`: Tokenizer corresponding to the model.
        - `seqs`: List of input sequences.
        - `seq_pair`: Optional second sequence (e.g., for paired embeddings).
        - `layers`: Layers to extract embeddings from.
        - `method`: How to aggregate embeddings.
        - `batch_size`: Number of sequences per batch.
        - `pbar`: Show progress bar.
        - `max_len`: Maximum sequence length.
        Returns:
        - NumPy array of shape (num_seqs, hidden_dim * len(layers)).
        """

        model = self.model
        tok = self.tok
        method = self.method
        device = next(model.parameters()).device  # Get device from model
        seqs = [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seqs]

        chunks = dl.chunkify(seqs, batch_size)
        chunks_pair = [None]

        if seq_pair is not None:
            assert len(seq_pair) == len(seqs)
            chunks_pair = dl.chunkify(
                [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seq_pair],
                batch_size,
            )

        chunks_zipped = list(zip_longest(chunks, chunks_pair))
        embeddings = []

        for seq_chunk in tqdm(chunks_zipped, disable=not pbar):
            encoded = tok(
                *seq_chunk, padding="max_length", max_length=max_len, return_tensors="pt"
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}

            x = model.forward(**encoded, output_hidden_states=True, output_attentions=True)

            if method == "pool":
                embeddings.append(x.pooler_output.type(torch.float64))
                continue

            batch_embeddings = []

            for i in range(len(seq_chunk[0])):
                e = []
                for l in layers:
                    h = x.hidden_states[l][i].type(torch.float64)

                    if method == "cls":
                        e.append(h[0])
                        continue

                    if seq_chunk[1] is None:
                        seq_len = len(seq_chunk[0][i].split())
                    else:
                        seq_len = (
                                len(seq_chunk[0][i].split())
                                + len(seq_chunk[1][i].split())
                                + 1  # For the sep token
                        )

                    seq_hidden = h[1: 1 + seq_len]

                    if method == "mean":
                        e.append(seq_hidden.mean(axis=0))
                    elif method == "max":
                        e.append(seq_hidden.max(axis=0))
                    elif method == "attn_mean":
                        attn = x.attentions[l][i, :, :, : seq_len + 2]
                        print(attn.sum(axis=-1))
                        raise NotImplementedError
                    else:
                        raise ValueError(f"Unrecognized method: {method}")

                e = torch.cat(e)
                batch_embeddings.append(e)
            embeddings.append(torch.stack(batch_embeddings))  # added
        embeddings = torch.cat(embeddings)
        return embeddings

    def freeze_bert_layers(self):
        # Freeze the embeddings (word, position, token type)
        for param in self.model.embeddings.parameters():
            param.requires_grad = False
        # Freeze all Bert layers except the last 4
        for i, layer in enumerate(self.model.encoder.layer):
            if i < 8:  # Freeze first 8 layers (0 to 7)
                for param in layer.parameters():
                    param.requires_grad = False
        # Keep the last 4 layers (8 to 11) trainable
        for i in range(8, 12):
            for param in self.model.encoder.layer[i].parameters():
                param.requires_grad = True

    def __call__(self, seqs: List[str]):
        return self.get_transformer_embeddings(seqs, batch_size=self.batch_size)


class CVCClassifierModel(nn.Module):
    def __init__(self, model_dir: str = TRANSFORMER, method: str = 'mean', device: str = 'cuda', batch_size: int = 1024):
        super().__init__()
        self.device = device
        self.model = CVCModel(model_dir, method, device, batch_size)
        self.batch_size = batch_size
        self.method = method

        # Add linear layers
        # self.linear = nn.Linear(768, 2).to(device)
        hidden_dim, num_classes = (768 // 2), 2
        self.linear = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)  # Output dim = 2 for binary classification
        ).to(device)

    def __call__(self, seqs: List[str]):
        embeddings = self.model(seqs)  # Get transformer embeddings
        logits = self.linear(embeddings.to(torch.float32))  # Pass through linear layer
        return logits

    def predict(self, seqs: List[str]):
        logits = self(seqs)  # Get logits
        probs = torch.nn.functional.softmax(logits, dim=-1)  # Convert to probabilities
        preds = torch.argmax(probs, dim=-1)  # Get class predictions
        return preds