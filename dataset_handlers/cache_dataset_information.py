import json
import os
import time
import random
import uuid
import tempfile
import shutil
from datetime import datetime
from fileinput import filename
from pathlib import Path


def safe_append_dataset_info(data, filename="dataset_information.jsonl", max_retries=20):
    """
    Safely append dataset information to a file using atomic operations.
    Works across different computers accessing shared network storage.

    Args:
        data (dict): Dictionary containing the dataset information
        filename (str): Name of the file to append to
        max_retries (int): Maximum number of retry attempts
    """
    filename = Path(filename)

    # Ensure the directory exists
    filename.parent.mkdir(parents=True, exist_ok=True)

    # Generate unique identifiers for this write operation
    machine_id = os.environ.get('HOSTNAME', os.environ.get('COMPUTERNAME', 'unknown'))
    process_id = os.getpid()
    unique_id = str(uuid.uuid4())[:8]

    # Create a unique temporary filename in the same directory
    temp_filename = filename.parent / f".tmp_{machine_id}_{process_id}_{unique_id}_{filename.name}"

    for attempt in range(max_retries):
        try:
            # Step 1: Write to temporary file first
            with open(temp_filename, 'w') as temp_file:
                json.dump(data, temp_file)
                temp_file.write('\n')
                temp_file.flush()
                os.fsync(temp_file.fileno())  # Force write to disk

            # Step 2: Atomically append the temp file content to main file
            success = atomic_append_file(temp_filename, filename)

            if success:
                print(f"Successfully wrote dataset info to {filename}")
                return True
            else:
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    wait_time = random.uniform(0.1, 0.5) * (2 ** min(attempt, 6))
                    print(f"Append attempt {attempt + 1} failed, retrying in {wait_time:.2f}s")
                    time.sleep(wait_time)

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                wait_time = random.uniform(0.1, 0.5) * (2 ** min(attempt, 6))
                time.sleep(wait_time)

        finally:
            # Clean up temporary file if it exists
            try:
                if temp_filename.exists():
                    temp_filename.unlink()
            except:
                pass  # Ignore cleanup errors

    print(f"Failed to write after {max_retries} attempts")
    return False


def atomic_append_file(source_file, target_file, max_attempts=10):
    """
    Atomically append content from source_file to target_file using file operations
    that work across network file systems.
    """
    source_file = Path(source_file)
    target_file = Path(target_file)

    if not source_file.exists():
        return False

    # Read the content to append
    try:
        with open(source_file, 'r') as f:
            content_to_append = f.read()
    except Exception as e:
        print(f"Failed to read temporary file: {e}")
        return False

    for attempt in range(max_attempts):
        try:
            # Method 1: Try direct append (fastest when it works)
            try:
                with open(target_file, 'a') as f:
                    f.write(content_to_append)
                    f.flush()
                    os.fsync(f.fileno())
                return True
            except (OSError, IOError) as e:
                # If direct append fails, try the copy-and-replace method
                pass

            # Method 2: Copy-and-replace approach (more reliable across network FS)
            unique_temp = target_file.parent / f".tmp_merge_{uuid.uuid4().hex[:8]}_{target_file.name}"

            try:
                # Copy existing content if file exists
                if target_file.exists():
                    shutil.copy2(target_file, unique_temp)

                # Append new content
                with open(unique_temp, 'a') as f:
                    f.write(content_to_append)
                    f.flush()
                    os.fsync(f.fileno())

                # Atomic rename (this is the key atomic operation)
                if os.name == 'nt':  # Windows
                    # On Windows, need to remove target first
                    if target_file.exists():
                        target_file.unlink()
                    unique_temp.rename(target_file)
                else:  # Unix/Linux
                    unique_temp.rename(target_file)

                return True

            except Exception as inner_e:
                # Clean up temp file
                try:
                    if unique_temp.exists():
                        unique_temp.unlink()
                except:
                    pass

                if attempt < max_attempts - 1:
                    wait_time = random.uniform(0.05, 0.2) * (2 ** min(attempt, 4))
                    time.sleep(wait_time)
                else:
                    print(f"Atomic append failed after {max_attempts} attempts: {inner_e}")

        except Exception as e:
            print(f"Atomic append attempt {attempt + 1} failed: {e}")
            if attempt < max_attempts - 1:
                time.sleep(random.uniform(0.1, 0.3))

    return False


def safe_append_with_backup(data, filename="dataset_information.jsonl", backup_dir="backup_logs"):
    """
    Wrapper function that also creates individual backup files as fallback.
    """
    # Try the main atomic append first
    success = safe_append_dataset_info(data, filename)

    # Always create a backup file as well (for extra safety)
    try:
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
        machine_id = os.environ.get('HOSTNAME', os.environ.get('COMPUTERNAME', 'unknown'))
        backup_file = backup_path / f"dataset_info_{machine_id}_{timestamp}.json"

        with open(backup_file, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"Backup saved to {backup_file}")

    except Exception as e:
        print(f"Warning: Could not create backup file: {e}")

    return success


# Replace the extract_dataset_information section with this:
def extract_dataset_information_function(dataset_type, filter_num_of_patients, filter_num_of_healthy,
                                         filter_to_inflate, top_n_seqs, train_pos_seqs, valid_pos_seqs, test_pos_seqs,
                                         neg_seqs, valid_neg_seqs, test_neg_seqs, train_patient_ids, valid_patient_ids, test_patient_ids,
                                         df_bld, df_hlt):
    print("Extracting dataset information...")
    # Collect all the information
    dataset_info = {
        # Timestamp and machine info for tracking
        "timestamp": datetime.now().isoformat(),
        "machine_id": os.environ.get('HOSTNAME', os.environ.get('COMPUTERNAME', 'unknown')),
        "process_id": os.getpid(),
        "run_id": str(uuid.uuid4()),

        # 1. Dataset configuration parameters
        "dataset_type": dataset_type,
        "filter_num_of_patients": filter_num_of_patients,
        "filter_num_of_healthy": filter_num_of_healthy,
        "filter_to_inflate": filter_to_inflate,
        "top_n_seqs": top_n_seqs,

        # 2. Sequence lengths
        "train_pos_seqs_count": len(train_pos_seqs),
        "valid_pos_seqs_count": len(valid_pos_seqs),
        "test_pos_seqs_count": len(test_pos_seqs),
        "neg_seqs_count": len(neg_seqs),
        "valid_neg_seqs_count": len(valid_neg_seqs),
        "test_neg_seqs_count": len(test_neg_seqs),

        # 3. Patient counts
        "train_patients_count": len(train_patient_ids),
        "valid_patients_count": len(valid_patient_ids),
        "test_patients_count": len(test_patient_ids),
        "total_disease_patients": df_bld["patient_id"].nunique(),
        "total_healthy_patients": df_hlt["patient_id"].nunique(),
    }

    # Safely append to file with backup
    safe_append_with_backup(dataset_info,
                            filename="cache/datasets_information/dataset_runs_info.jsonl",
                            backup_dir="cache/datasets_information/backup_logs")
    print("Dataset information extracted and saved.")
    exit(0)