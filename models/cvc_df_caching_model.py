import os
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
from hashlib import md5
from cache_handler import get_model_dir


class CVCDFCachingModel(nn.Module):
    def __init__(self, model, args, device, verbose=False):
        super(CVCDFCachingModel, self).__init__()
        self.model = model
        self.device = device
        self.verbose = verbose

        # Cache setup - using pickle for single file storage
        self.cache_dir = os.path.join(get_model_dir(args, False), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_file = os.path.join(self.cache_dir, "output_cache_df.pkl")

        # Load existing cache
        self.cache_df = self._load_cache()

    def _load_cache(self):
        """Load pre-computed outputs cache as pandas DataFrame"""
        if os.path.exists(self.cache_file):
            try:
                cache_df = pd.read_pickle(self.cache_file)
                if self.verbose:
                    print(f"Loaded cache with {len(cache_df)} entries")
                return cache_df
            except Exception as e:
                print(f"Error loading cache: {e}")

        # Create empty DataFrame with required columns
        return pd.DataFrame(columns=['sequence', 'output_tensor'])

    def _save_cache(self):
        """Save the updated cache DataFrame to disk"""
        try:
            self.cache_df.to_pickle(self.cache_file)
            if self.verbose:
                print(f"Saved cache with {len(self.cache_df)} entries")
        except Exception as e:
            print(f"Error saving cache: {e}")

    def forward(self, x):
        """Forward pass with efficient pandas-based caching"""
        # Convert input to numpy array of strings if needed
        if not isinstance(x, (list, np.ndarray)):
            x = [x]

        sequences = np.array([str(seq) for seq in x])
        batch_size = len(sequences)

        # Create a DataFrame for current batch to enable efficient lookups
        batch_df = pd.DataFrame({
            'sequence': sequences,
            'batch_idx': range(batch_size)
        })

        # Perform efficient merge to find cached entries
        merged_df = batch_df.merge(
            self.cache_df[['sequence', 'output_tensor']],
            on='sequence',
            how='left'
        )

        # Separate cached and uncached sequences
        cached_mask = merged_df['output_tensor'].notna()
        cached_df = merged_df[cached_mask]
        uncached_df = merged_df[~cached_mask]

        cache_hits = len(cached_df)

        # Initialize results array
        batch_outputs = [None] * batch_size

        # Load cached results
        for _, row in cached_df.iterrows():
            batch_idx = int(row['batch_idx'])
            output_tensor = row['output_tensor'].to(self.device)
            batch_outputs[batch_idx] = output_tensor

        # Process uncached sequences if any exist
        if len(uncached_df) > 0:
            uncached_sequences = uncached_df['sequence'].values
            uncached_indices = uncached_df['batch_idx'].values

            # Convert to appropriate format for model input
            uncached_batch = np.array(uncached_sequences)

            # Process batch through model
            with torch.no_grad():
                outputs = self.model(uncached_batch)

                # Handle both single output and batch outputs
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(0)

                # Store results and update cache
                new_cache_entries = []

                for i, (batch_idx, sequence) in enumerate(zip(uncached_indices, uncached_sequences)):
                    output_tensor = outputs[i]
                    batch_outputs[batch_idx] = output_tensor

                    # Prepare cache entry
                    new_cache_entries.append({
                        'sequence': sequence,
                        'output_tensor': output_tensor.detach().cpu()
                    })

                # Update cache DataFrame efficiently
                if new_cache_entries:
                    new_entries_df = pd.DataFrame(new_cache_entries)
                    self.cache_df = pd.concat([self.cache_df, new_entries_df], ignore_index=True)

                    # Remove duplicates if any (keep last occurrence)
                    self.cache_df = self.cache_df.drop_duplicates(subset=['sequence'], keep='last')

                    # Save updated cache
                    self._save_cache()

        # Print cache statistics
        if batch_size > 0 and self.verbose:
            hit_rate = cache_hits / batch_size * 100
            print(f"Cache hits: {cache_hits}/{batch_size} ({hit_rate:.1f}%)")

        # Stack results into final tensor
        model_preds = torch.stack(batch_outputs)
        return model_preds

    def get_cache_stats(self):
        """Return cache statistics"""
        return {
            'total_entries': len(self.cache_df),
            'cache_file_size': os.path.getsize(self.cache_file) if os.path.exists(self.cache_file) else 0
        }

    def clear_cache(self):
        """Clear all cached data"""
        import shutil
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)
        os.makedirs(self.cache_dir, exist_ok=True)
        self.cache_df = pd.DataFrame(columns=['sequence', 'output_tensor'])
        if self.verbose:
            print("Cache cleared")


# Cloud Version 2.0 (still older - with hash):
# import os
# import pandas as pd
# import torch
# import numpy as np
# import torch.nn as nn
# from hashlib import md5
# from cache_handler import get_model_dir
#
#
# class CVCCachingModel(nn.Module):
#     def __init__(self, model, args, device, verbose=False):
#         super(CVCCachingModel, self).__init__()
#         self.model = model
#         self.device = device
#         self.verbose = verbose
#
#         # Cache setup - using pickle for single file storage
#         self.cache_dir = os.path.join(get_model_dir(args, False), "cache")
#         os.makedirs(self.cache_dir, exist_ok=True)
#         self.cache_file = os.path.join(self.cache_dir, "output_cache.pkl")
#
#         # Load existing cache
#         self.cache_df = self._load_cache()
#
#     def _load_cache(self):
#         """Load pre-computed outputs cache as pandas DataFrame"""
#         if os.path.exists(self.cache_file):
#             try:
#                 cache_df = pd.read_pickle(self.cache_file)
#                 if self.verbose:
#                     print(f"Loaded cache with {len(cache_df)} entries")
#                 return cache_df
#             except Exception as e:
#                 print(f"Error loading cache: {e}")
#
#         # Create empty DataFrame with required columns
#         return pd.DataFrame(columns=['sequence_hash', 'sequence', 'output_tensor'])
#
#     def _save_cache(self):
#         """Save the updated cache DataFrame to disk"""
#         try:
#             self.cache_df.to_pickle(self.cache_file)
#             if self.verbose:
#                 print(f"Saved cache with {len(self.cache_df)} entries")
#         except Exception as e:
#             print(f"Error saving cache: {e}")
#
#     def _get_cache_key(self, sequence):
#         """Generate a unique key for caching purposes"""
#         if isinstance(sequence, str):
#             return md5(sequence.encode()).hexdigest()
#         else:
#             return md5(str(sequence).encode()).hexdigest()
#
#     def forward(self, x):
#         """Forward pass with efficient pandas-based caching"""
#         # Convert input to numpy array of strings if needed
#         if not isinstance(x, (list, np.ndarray)):
#             x = [x]
#
#         sequences = np.array([str(seq) for seq in x])
#         batch_size = len(sequences)
#
#         # Generate cache keys for all sequences
#         cache_keys = [self._get_cache_key(seq) for seq in sequences]
#
#         # Create a DataFrame for current batch to enable efficient lookups
#         batch_df = pd.DataFrame({
#             'sequence_hash': cache_keys,
#             'sequence': sequences,
#             'batch_idx': range(batch_size)
#         })
#
#         # Perform efficient merge to find cached entries
#         merged_df = batch_df.merge(
#             self.cache_df[['sequence_hash', 'output_tensor']],
#             on='sequence_hash',
#             how='left'
#         )
#
#         # Separate cached and uncached sequences
#         cached_mask = merged_df['output_tensor'].notna()
#         cached_df = merged_df[cached_mask]
#         uncached_df = merged_df[~cached_mask]
#
#         cache_hits = len(cached_df)
#
#         # Initialize results array
#         batch_outputs = [None] * batch_size
#
#         # Load cached results
#         for _, row in cached_df.iterrows():
#             batch_idx = int(row['batch_idx'])
#             output_tensor = row['output_tensor'].to(self.device)
#             batch_outputs[batch_idx] = output_tensor
#
#         # Process uncached sequences if any exist
#         if len(uncached_df) > 0:
#             uncached_sequences = uncached_df['sequence'].values
#             uncached_indices = uncached_df['batch_idx'].values
#             uncached_keys = uncached_df['sequence_hash'].values
#
#             # Convert to appropriate format for model input
#             uncached_batch = np.array(uncached_sequences)
#
#             # Process batch through model
#             with torch.no_grad():
#                 outputs = self.model(uncached_batch)
#
#                 # Handle both single output and batch outputs
#                 if outputs.dim() == 1:
#                     outputs = outputs.unsqueeze(0)
#
#                 # Store results and update cache
#                 new_cache_entries = []
#
#                 for i, (batch_idx, cache_key, sequence) in enumerate(
#                         zip(uncached_indices, uncached_keys, uncached_sequences)):
#                     output_tensor = outputs[i]
#                     batch_outputs[batch_idx] = output_tensor
#
#                     # Prepare cache entry
#                     new_cache_entries.append({
#                         'sequence_hash': cache_key,
#                         'sequence': sequence,
#                         'output_tensor': output_tensor.detach().cpu()
#                     })
#
#                 # Update cache DataFrame efficiently
#                 if new_cache_entries:
#                     new_entries_df = pd.DataFrame(new_cache_entries)
#                     self.cache_df = pd.concat([self.cache_df, new_entries_df], ignore_index=True)
#
#                     # Remove duplicates if any (keep last occurrence)
#                     self.cache_df = self.cache_df.drop_duplicates(subset=['sequence_hash'], keep='last')
#
#                     # Save updated cache
#                     self._save_cache()
#
#         # Print cache statistics
#         if batch_size > 0 and self.verbose:
#             hit_rate = cache_hits / batch_size * 100
#             print(f"Cache hits: {cache_hits}/{batch_size} ({hit_rate:.1f}%)")
#
#         # Stack results into final tensor
#         model_preds = torch.stack(batch_outputs)
#         return model_preds
#
#     def get_cache_stats(self):
#         """Return cache statistics"""
#         return {
#             'total_entries': len(self.cache_df),
#             'cache_file_size': os.path.getsize(self.cache_file) if os.path.exists(self.cache_file) else 0
#         }
#
#     def clear_cache(self):
#         """Clear all cached data"""
#         import shutil
#         if os.path.exists(self.cache_dir):
#             shutil.rmtree(self.cache_dir)
#         os.makedirs(self.cache_dir, exist_ok=True)
#         self.cache_df = pd.DataFrame(columns=['sequence_hash', 'sequence', 'output_tensor'])
#         if self.verbose:
#             print("Cache cleared")