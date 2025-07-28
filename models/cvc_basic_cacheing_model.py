import os
import json
import torch
import numpy as np
import torch.nn as nn
from hashlib import md5
from cache_handler import get_model_dir


class CVCBasicCachingModel(nn.Module):
    def __init__(self, model, args, device, verbose=False):
        super(CVCBasicCachingModel, self).__init__()
        self.model = model
        self.device = device
        self.verbose = verbose

        # Cache setup
        self.cache_dir = os.path.join(get_model_dir(args, False), "output_cache.json")
        self.cache = {}
        self._load_cache()
        self._has_new_entries = False

    def _load_cache(self):
        """Load pre-computed outputs from cache file if it exists (only once)"""
        if self.cache_dir and os.path.exists(self.cache_dir):
            try:
                with open(self.cache_dir, 'r') as f:
                    cache_data = json.load(f)

                # Convert lists back to tensors
                for key, values in cache_data.items():
                    self.cache[key] = [torch.tensor(v, device=self.device) for v in values]

                if self.verbose:
                    print(f"Loaded cache with {len(self.cache)} entries")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.cache = {}

    def _save_cache(self):
        """Save entire cache to file (only when there are new entries)"""
        if not self._has_new_entries or not self.cache_dir:
            return

        try:
            # Convert tensors to lists for JSON serialization
            cache_data = {}
            for key, values in self.cache.items():
                cache_data[key] = [v.detach().cpu().numpy().tolist() for v in values]

            # Write entire cache (replacing previous file)
            with open(self.cache_dir, 'w') as f:
                json.dump(cache_data, f)

            if self.verbose:
                print(f"Saved cache with {len(self.cache)} total entries")

            self._has_new_entries = False
        except Exception as e:
            print(f"Error saving cache: {e}")

    def _get_cache_key(self, sequence):
        """Generate a unique key for caching purposes"""
        if isinstance(sequence, str):
            return md5(sequence.encode()).hexdigest()
        else:
            return md5(str(sequence).encode()).hexdigest()

    def forward(self, x):
        """Forward pass with optimized caching"""
        batch_size = len(x)

        # Generate all cache keys at once (vectorized operation)
        cache_keys = [self._get_cache_key(seq) for seq in x]

        # Use dict.get() with default None to check cache existence efficiently
        # This avoids the for-loop with individual key checks
        cached_results = [self.cache.get(key) for key in cache_keys]

        # Identify uncached sequences using list comprehension (faster than loops)
        uncached_mask = [result is None for result in cached_results]
        uncached_indices = [i for i, is_uncached in enumerate(uncached_mask) if is_uncached]

        cache_hits = batch_size - len(uncached_indices)

        # Process uncached sequences if any exist
        if uncached_indices:
            # Extract uncached sequences efficiently
            uncached_sequences = [x[i] for i in uncached_indices]
            uncached_keys = [cache_keys[i] for i in uncached_indices]

            # Process batch through model
            uncached_batch = np.array(uncached_sequences)
            with torch.no_grad():
                outputs = self.model(uncached_batch)

                # Update cache with new results
                for seq_idx, cache_key in enumerate(uncached_keys):
                    output = outputs[seq_idx]
                    self.cache[cache_key] = [output]
                    # Update the cached_results for final assembly
                    original_idx = uncached_indices[seq_idx]
                    cached_results[original_idx] = [output]

            self._has_new_entries = True

        # Print cache stats
        if batch_size > 0 and self.verbose:
            print(f"Cache hits: {cache_hits}/{batch_size} ({cache_hits / batch_size * 100:.1f}%)")

        # Assemble final results (all positions now have valid cached_results)
        model_preds = torch.stack([result[0] for result in cached_results])

        # Save cache only if there were new entries
        if self._has_new_entries:
            self._save_cache()

        return model_preds

    def __del__(self):
        """Ensure cache is saved when object is destroyed"""
        if hasattr(self, '_has_new_entries') and self._has_new_entries:
            self._save_cache()
