import os
import json
import torch
import numpy as np
import torch.nn as nn
from hashlib import md5
from cache_handler import get_model_dir


class CVCCachingModel(nn.Module):
    def __init__(self, model, args, device, verbose=False):
        super(CVCCachingModel, self).__init__()
        self.model = model
        self.device = device
        self.verbose = verbose

        # Cache setup
        self.cache_dir = os.path.join(get_model_dir(args, False), "output_cache.json")
        self.cache = {}
        self._load_cache()

    def _load_cache(self):
        """Load pre-computed outputs from cache file if it exists"""
        if self.cache_dir:
            if os.path.exists(self.cache_dir):
                try:
                    with open(self.cache_dir, 'r') as f:
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
            # Load existing cache from disk to ensure we have the latest version
            existing_cache = {}
            if os.path.exists(self.cache_dir):
                try:
                    with open(self.cache_dir, 'r') as f:
                        existing_cache = json.load(f)
                except Exception as e:
                    print(f"Error reading existing cache: {e}")

            # Update with new entries
            for key, values in new_entries.items():
                existing_cache[key] = [v.detach().cpu().numpy().tolist() for v in values]

            # Write back the updated cache
            try:
                with open(self.cache_dir, 'w') as f:
                    json.dump(existing_cache, f)
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

    def forward(self, x):
        """Forward pass with caching and additional statistics"""
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
            with torch.no_grad():
                outputs = self.model(uncached_batch)

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
        model_preds = torch.stack([batch_output[0] for batch_output in batch_outputs])
        return model_preds
