import sys
sys.path.append('other_models/CVC')
import numpy as np
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from itertools import zip_longest
from typing import *
from tqdm.auto import tqdm
from transformers import BertModel
import cvc.data_loader as dl
import cvc.featurization as ft
import warnings
from lab_notebooks.utils import TRANSFORMER
from peft import LoraConfig, get_peft_model, TaskType
warnings.simplefilter("ignore", category=FutureWarning)


class CVCModel(nn.Module):
    def __init__(self, model_dir: str = TRANSFORMER, method: str = 'mean', device: str = 'cuda', batch_size: int = 256, dropout_rate: float = 0.0, freeze_embed_model: bool = False, cvc_layers_to_train: int = 3, lora: bool = False):
        super().__init__()
        self.device = device
        self.model = BertModel.from_pretrained(model_dir, add_pooling_layer=method == "pool", output_hidden_states=True).to(device)

        # TODO: Add option to go do with / without PEFT !!! (And any other options if needed, like lora_dropout, etc.)
        #  Also: Changed lora_dropout to 0.0 in the original code!
        if lora:
            # Define PEFT configuration
            peft_config_esmc = LoraConfig(
                r=8,
                lora_alpha=32,
                lora_dropout=dropout_rate,
                bias='none',
                layers_to_transform=list(range(11, 11-cvc_layers_to_train, -1)),
                task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=['attention.self.query', 'attention.self.value'],  # TODO: Made the following changes to LoraConfig due to suggestions
                # target_modules=['attention.self.query', 'attention.self.key', 'attention.self.value',
                #                 'attention.output.dense', 'intermediate.dense', 'output.dense'],
            )
            # Load pre-trained model
            self.model = get_peft_model(self.model, peft_config_esmc).to(device)

        for module in self.model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = dropout_rate

        self.tok = ft.get_pretrained_bert_tokenizer(model_dir)
        self.freeze_bert_layers(freeze_embed_model, cvc_layers_to_train)
        self.method = method  # Options: "mean", "max", "attn_mean", "cls", "pool"
        self.batch_size = batch_size
        self.mask_tokens = False  # Set to True if you want to mask tokens during training

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

        # Apply masking at the string level if in training mode and mask_tokens is True
        if model.training and self.mask_tokens:
            masked_seqs = []
            for seq in seqs:
                # First make sure the sequence has whitespace
                if not ft.is_whitespaced(seq):
                    seq = ft.insert_whitespace(seq)

                # Split the sequence into tokens
                tokens = seq.split()

                # Randomly select 15% of the tokens for potential masking
                num_tokens = len(tokens)
                num_to_mask = max(1, int(0.15 * num_tokens))  # Ensure at least one token is considered
                mask_indices = random.sample(range(num_tokens), num_to_mask)

                # For 80% of those tokens, replace with the mask token
                for idx in mask_indices:
                    if random.random() < 0.8:  # 80% chance to mask
                        tokens[idx] = ft.MASK

                # Rejoin the tokens
                masked_seqs.append(' '.join(tokens))

            seqs = masked_seqs

            # If there's a second sequence, apply the same masking logic
            if seq_pair is not None:
                masked_seq_pair = []
                for seq in seq_pair:
                    if not ft.is_whitespaced(seq):
                        seq = ft.insert_whitespace(seq)

                    tokens = seq.split()
                    num_tokens = len(tokens)
                    num_to_mask = max(1, int(0.15 * num_tokens))
                    mask_indices = random.sample(range(num_tokens), num_to_mask)

                    for idx in mask_indices:
                        if random.random() < 0.8:
                            tokens[idx] = ft.MASK

                    masked_seq_pair.append(' '.join(tokens))

                seq_pair = masked_seq_pair
        else:
            # If no masking, just ensure whitespace
            seqs = [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seqs]
            if seq_pair is not None:
                seq_pair = [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seq_pair]

        chunks = dl.chunkify(seqs, batch_size)
        chunks_pair = [None]

        if seq_pair is not None:
            assert len(seq_pair) == len(seqs)
            chunks_pair = dl.chunkify(seq_pair, batch_size)

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

            embeddings.append(torch.stack(batch_embeddings))

        embeddings = torch.cat(embeddings)
        return embeddings

    def freeze_bert_layers(self, freeze_embed_model: bool, cvc_layers_to_train: int):
        if freeze_embed_model:
            # Freeze the embeddings (word, position, token type)
            for param in self.model.parameters():
                param.requires_grad = False
            return

        # Freeze the embeddings (word, position, token type)
        for param in self.model.embeddings.parameters():
            param.requires_grad = False

        # Freeze all Bert layers except the last cvc_layers_to_train layers
        for i, layer in enumerate(self.model.encoder.layer):
            if i < 12 - cvc_layers_to_train:  # Freeze first 9 layers (0 to 8)
                for param in layer.parameters():
                    param.requires_grad = False

        # Keep the last cvc_layers_to_train layers (12-cvc_layers_to_train to 11) trainable
        for i in range(12 - cvc_layers_to_train, 12):
            for param in self.model.encoder.layer[i].parameters():
                param.requires_grad = True

    def forward(self, seqs: List[str]):
        return self.get_transformer_embeddings(seqs, batch_size=self.batch_size)


class CVCClassifierModel(nn.Module):
    def __init__(self, model_dir: str = TRANSFORMER, method: str = 'mean', ch_dropout: float = 0.0, device: str = 'cuda',
                 batch_size: int = 256, freeze_embed_model: bool = False, cvc_layers_to_train: int = 3, lora: bool = False, ch_type: str = 'none'):
        super().__init__()
        self.device = device
        self.model = CVCModel(model_dir, method, device, batch_size, ch_dropout, freeze_embed_model, cvc_layers_to_train, lora)
        self.batch_size = batch_size
        self.method = method
        self.dropout_rate = ch_dropout

        # Add linear layers
        # self.linear = nn.Linear(768, 2).to(device)
        hidden_dim, num_classes = (768 // 2), 2
        if ch_type == 'none':
            self.linear = nn.Sequential(
                nn.Linear(768, hidden_dim),
                nn.ReLU(),
                # nn.Dropout(self.dropout_rate),  # Add dropout after first layer
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                # nn.Dropout(self.dropout_rate),  # Add another dropout layer
                nn.Linear(hidden_dim // 2, num_classes)  # Output dim = 2 for binary classification
            ).to(device)
        elif ch_type == 'v1':  # larger version
            self.linear = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Linear(768, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, num_classes)  # Output dim = 2 for binary classification
            ).to(device)
        elif ch_type == 'v2':
            self.linear = nn.Sequential(
                nn.Conv1d(in_channels=768, out_channels=256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1),  # Global max pooling across sequence length
                nn.Flatten(),  # shape: (batch_size, 128)
                nn.Linear(128, num_classes)
            ).to(device)
        else:
            raise ValueError(f"Unknown ch_type: {ch_type}")

    def forward(self, seqs: List[str]):
        embeddings = self.model(seqs)  # Get transformer embeddings
        logits = self.linear(embeddings.to(torch.float32))  # Pass through linear layer

        # from transformers import BertConfig, BertForMaskedLM
        # kwargs = {'hidden_size': 768, 'num_hidden_layers': 12, 'num_attention_heads': 12, 'intermediate_size': 3072, 'hidden_act': 'gelu', 'hidden_dropout_prob': 0.1, 'attention_probs_dropout_prob': 0.1, 'max_position_embeddings': 512, 'type_vocab_size': 2, 'initializer_range': 0.02, 'position_embedding_type': 'absolute'}
        # config = BertConfig(
        #     **kwargs,
        #     vocab_size=len(ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL),
        #     pad_token_id=ft.AMINO_ACIDS_WITH_ALL_ADDITIONAL_TO_IDX[ft.PAD],
        # )
        # model = BertForMaskedLM(config)

        # set the state of the model to the state of the pretrained model
        # model.load_state_dict(self.model.model.state_dict())
        # self.model.model = model
        # embeddings2 = self.model(seqs[:10])  # Get transformer embeddings
        # logits2 = self.linear(embeddings.to(torch.float32))  # Pass through linear layer

        # max_len: int = 64
        # seqs2 = [s if ft.is_whitespaced(s) else ft.insert_whitespace(s) for s in seqs]
        # encoded = self.model.tok(
        #     *seqs2, padding="max_length", max_length=max_len, return_tensors="pt"
        # )
        # encoded = {k: v.to(self.device) for k, v in encoded.items()}
        #
        # x = model.forward(**encoded, output_hidden_states=True, output_attentions=True)
        # out = model(seqs[:10])

        return logits

    def predict(self, seqs: List[str]):
        logits = self(seqs)  # Get logits
        probs = torch.nn.functional.softmax(logits, dim=-1)  # Convert to probabilities
        preds = torch.argmax(probs, dim=-1)  # Get class predictions
        return preds

    def set_mask_on(self):
        self.model.mask_tokens = True

    def set_mask_off(self):
        self.model.mask_tokens = False