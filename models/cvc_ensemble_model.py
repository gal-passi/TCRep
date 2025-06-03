import os
import json
import torch
import numpy as np
import torch.nn as nn
import random
from tqdm import tqdm
from models.cvc_model import CVCClassifierModel
from cache_handler import load_model_state
from hashlib import md5


NUM_OF_MODELS = 5


class CVCEnsembleModel(nn.Module):
    def __init__(self, args, device, models_weights=None, cache_dir=None, default_to_return='weighted_sum', verbose=False):
        super(CVCEnsembleModel, self).__init__()

        self.models = []
        self.device = device
        self.default_to_return = default_to_return
        self.verbose = verbose
        args.to_ensemble = False
        for i in range(1, NUM_OF_MODELS + 1):
            args.negative_partition = i
            model = CVCClassifierModel(batch_size=args.batch_size, ch_dropout=args.classification_dropout,
                                       cvc_layers_to_train=args.cvc_layers_to_train,
                                       freeze_embed_model=args.freeze_embed_model,
                                       lora=args.lora, ch_type=args.ch_type, device=device)
            trained_model = load_model_state(model, args, args.epochs - 1, device)
            if trained_model is not None:
                self.models.append(trained_model)
            else:
                raise ValueError(
                    f"Model for negative partition {i} could not be loaded. Check the model path or training process.")
        args.to_ensemble = True
        self.weights = models_weights if models_weights is not None else [1.0 / NUM_OF_MODELS] * len(self.models)

        # Cache setup
        self.cache_dir = cache_dir
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        self.cache = {}
        self._load_cache()

    def _load_cache(self):
        """Load pre-computed outputs from cache file if it exists"""
        if self.cache_dir:
            cache_file = os.path.join(self.cache_dir, "model_cache_ensemble.json")
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        cache_data = json.load(f)

                    # Convert lists back to tensors
                    for key, values in cache_data.items():
                        self.cache[key] = [torch.tensor(v, device=self.device) for v in values]

                    print(f"Loaded cache with {len(self.cache)} entries")
                except Exception as e:
                    print(f"Error loading cache: {e}")
                    self.cache = {}

    def _save_cache(self, new_entries):
        """Save only new computed outputs to cache file, preserving existing entries"""
        if self.cache_dir and new_entries:
            cache_file = os.path.join(self.cache_dir, "model_cache_ensemble.json")

            # Load existing cache from disk to ensure we have the latest version
            existing_cache = {}
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, 'r') as f:
                        existing_cache = json.load(f)
                except Exception as e:
                    print(f"Error reading existing cache: {e}")

            # Update with new entries
            for key, values in new_entries.items():
                existing_cache[key] = [v.detach().cpu().numpy().tolist() for v in values]

            # Write back the updated cache
            try:
                with open(cache_file, 'w') as f:
                    json.dump(existing_cache, f)
                if self.verbose:
                    print(f"Updated cache with {len(new_entries)} new entries, total entries: {len(existing_cache)}")
            except Exception as e:
                print(f"Error saving cache: {e}")

    def _get_cache_key(self, sequence):
        """Generate a unique key for caching purposes"""
        if isinstance(sequence, str):
            return md5(sequence.encode()).hexdigest()
        else:
            # If it's not a string, convert to string and hash
            return md5(str(sequence).encode()).hexdigest()

    def forward(self, x, to_return=None):
        """Forward pass with caching and additional statistics"""
        if to_return is None:
            to_return = self.default_to_return
        if to_return not in ['weighted_sum', 'min', 'max', 'median', 'all_models'] and not isinstance(to_return, int):
            raise ValueError("to_return must be one of ['weighted_sum', 'min', 'max', 'median', 'all_models']")

        batch_outputs = []
        cache_hits = 0
        new_cache_entries = {}

        # First, separate sequences into cached and uncached
        uncached_indices = []
        uncached_sequences = []
        uncached_keys = []

        for i, sequence in enumerate(x):
            cache_key = self._get_cache_key(sequence)

            if cache_key in self.cache:
                # Cache hit - use stored model outputs
                batch_outputs.append(self.cache[cache_key])
                cache_hits += 1
            else:
                # Cache miss - mark for batch processing
                uncached_indices.append(i)
                uncached_sequences.append(sequence)
                uncached_keys.append(cache_key)

        # Process all uncached sequences in a single batch if any exist
        if uncached_sequences:
            # Convert to appropriate format for model input
            uncached_batch = np.array(uncached_sequences)

            # Process batch through each model
            for model_idx, model in enumerate(self.models):
                with torch.no_grad():
                    outputs = model(uncached_batch)

                    # Store each output in its corresponding sequence's results
                    for seq_idx, cache_key in enumerate(uncached_keys):
                        # Create entry in new_cache_entries if this is the first model for this sequence
                        if cache_key not in new_cache_entries:
                            new_cache_entries[cache_key] = []

                        # Store this model's output for this sequence
                        output = outputs[seq_idx]
                        new_cache_entries[cache_key].append(output)

            # Update in-memory cache with new entries
            self.cache.update(new_cache_entries)

            # Place cached results in correct order
            for i, idx in enumerate(uncached_indices):
                # Get results for this sequence
                cache_key = uncached_keys[i]
                batch_outputs.insert(idx, self.cache[cache_key])

            # Save new entries to disk
            self._save_cache(new_cache_entries)

        # Print cache stats
        if len(x) > 0 and self.verbose:
            print(f"Cache hits: {cache_hits}/{len(x)} ({cache_hits/len(x)*100:.1f}%)")

        # Stack and organize results
        stacked_outputs = []
        for model_idx in range(len(self.models)):
            model_preds = [batch_output[model_idx] for batch_output in batch_outputs]
            stacked_outputs.append(torch.stack(model_preds))

        # Calculate ensemble statistics (batch_size, output_dim)
        weighted_sum = sum(w * out for w, out in zip(self.weights, stacked_outputs))

        # Transpose to get shape (batch_size, num_models, output_dim)
        all_outputs = torch.stack(stacked_outputs, dim=1)

        # Calculate min, max, median across models dimension (dim=1)
        min_values, _ = torch.min(all_outputs, dim=1)
        max_values, _ = torch.max(all_outputs, dim=1)
        median_values = torch.median(all_outputs, dim=1).values

        result = {
            'weighted_sum': weighted_sum,
            'min': min_values,
            'max': max_values,
            'median': median_values,
            'all_models': all_outputs  # Include all models' outputs if needed
        }

        if isinstance(to_return, int):  # return outputs of only model to_return
            to_return = to_return % NUM_OF_MODELS
            return result['all_models'][:, to_return, :]
        return result[to_return]
