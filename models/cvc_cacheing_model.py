import os
import json
import torch
import numpy as np
import torch.nn as nn
from hashlib import md5
from cache_handler import get_model_dir
import tempfile
import shutil
import fcntl
import torch
from contextlib import contextmanager
import time


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

    @contextmanager
    def _file_lock(self, filepath):
        """Context manager for file locking to prevent concurrent access"""
        lock_file = filepath + ".lock"
        try:
            with open(lock_file, 'w') as lock:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                yield
        finally:
            try:
                os.remove(lock_file)
            except FileNotFoundError:
                pass

    def _atomic_write(self, filepath, data):
        """Atomically write data to file using temporary file + rename"""
        # Create temporary file in same directory to ensure same filesystem
        temp_dir = os.path.dirname(filepath) if os.path.dirname(filepath) else '.'

        with tempfile.NamedTemporaryFile(
                mode='w',
                dir=temp_dir,
                delete=False,
                suffix='.tmp'
        ) as temp_file:
            json.dump(data, temp_file, indent=2)
            temp_file.flush()
            os.fsync(temp_file.fileno())  # Force write to disk
            temp_path = temp_file.name

        # Atomic rename (works on most filesystems)
        try:
            shutil.move(temp_path, filepath)
        except Exception as e:
            # Clean up temp file if rename fails
            try:
                os.remove(temp_path)
            except FileNotFoundError:
                pass
            raise e

    def _create_backup(self, filepath):
        """Create backup of existing cache file"""
        if os.path.exists(filepath):
            backup_path = filepath + ".backup"
            try:
                shutil.copy2(filepath, backup_path)
                return backup_path
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Could not create backup: {e}")
                return None
        return None

    def _restore_from_backup(self, filepath):
        """Restore cache from backup if main file is corrupted"""
        backup_path = filepath + ".backup"
        if os.path.exists(backup_path):
            try:
                shutil.copy2(backup_path, filepath)
                if self.verbose:
                    print("Restored cache from backup")
                return True
            except Exception as e:
                if self.verbose:
                    print(f"Failed to restore from backup: {e}")
        return False

    def _validate_cache_data(self, data):
        """Validate cache data structure"""
        if not isinstance(data, dict):
            raise ValueError("Cache data must be a dictionary")

        for key, values in data.items():
            if not isinstance(key, str):
                raise ValueError(f"Cache key must be string, got {type(key)}")
            if not isinstance(values, list):
                raise ValueError(f"Cache values must be list, got {type(values)}")

    def _load_cache(self):
        """Load pre-computed outputs from cache file if it exists"""
        if not self.cache_dir:
            return

        if not os.path.exists(self.cache_dir):
            return

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._file_lock(self.cache_dir):
                    with open(self.cache_dir, 'r') as f:
                        cache_data = json.load(f)

                    # Validate data structure
                    self._validate_cache_data(cache_data)

                    # Convert lists back to tensors
                    for key, values in cache_data.items():
                        try:
                            self.cache[key] = [
                                torch.tensor(v, device=self.device) for v in values
                            ]
                        except Exception as e:
                            if self.verbose:
                                print(f"Warning: Skipping corrupted entry {key}: {e}")
                            continue

                    if self.verbose:
                        print(f"Loaded cache with {len(self.cache)} entries")
                    return

            except json.JSONDecodeError as e:
                if self.verbose:
                    print(f"Cache file corrupted (attempt {attempt + 1}): {e}")

                # Try to restore from backup
                if attempt == 0 and self._restore_from_backup(self.cache_dir):
                    continue

            except Exception as e:
                if self.verbose:
                    print(f"Error loading cache (attempt {attempt + 1}): {e}")

            if attempt < max_retries - 1:
                time.sleep(0.1 * (2 ** attempt))  # Exponential backoff

        # If all attempts failed, start with empty cache
        if self.verbose:
            print("Failed to load cache after all attempts, starting fresh")
        self.cache = {}

    def _save_cache(self, new_entries):
        """Save only new computed outputs to cache file, preserving existing entries"""
        if not self.cache_dir or not new_entries:
            return

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._file_lock(self.cache_dir):
                    # Create backup before modifying
                    backup_path = self._create_backup(self.cache_dir)

                    # Load existing cache from disk
                    existing_cache = {}
                    if os.path.exists(self.cache_dir):
                        try:
                            with open(self.cache_dir, 'r') as f:
                                existing_cache = json.load(f)
                            self._validate_cache_data(existing_cache)
                        except Exception as e:
                            if self.verbose:
                                print(f"Error reading existing cache: {e}")
                            existing_cache = {}

                    # Update with new entries
                    for key, values in new_entries.items():
                        try:
                            existing_cache[key] = [
                                v.detach().cpu().numpy().tolist() for v in values
                            ]
                        except Exception as e:
                            if self.verbose:
                                print(f"Warning: Failed to serialize entry {key}: {e}")
                            continue

                    # Validate final data before writing
                    self._validate_cache_data(existing_cache)

                    # Atomic write
                    self._atomic_write(self.cache_dir, existing_cache)

                    if self.verbose:
                        print(f"Updated cache with {len(new_entries)} new entries, "
                              f"total entries: {len(existing_cache)}")

                    # Clean up backup on success
                    if backup_path and os.path.exists(backup_path):
                        try:
                            os.remove(backup_path)
                        except Exception:
                            pass  # Not critical if backup cleanup fails

                    return

            except Exception as e:
                if self.verbose:
                    print(f"Error saving cache (attempt {attempt + 1}): {e}")

                # Try to restore from backup if we corrupted the file
                if attempt == 0 and os.path.exists(self.cache_dir):
                    backup_path = self.cache_dir + ".backup"
                    if os.path.exists(backup_path):
                        try:
                            shutil.copy2(backup_path, self.cache_dir)
                            if self.verbose:
                                print("Restored cache from backup after write failure")
                        except Exception:
                            pass

                if attempt < max_retries - 1:
                    time.sleep(0.1 * (2 ** attempt))  # Exponential backoff

        if self.verbose:
            print("Failed to save cache after all attempts")

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
