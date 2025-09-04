import numpy as np
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from itertools import combinations
from utils.utils import pairwise_scores, levenshtein_dist_non_bin


def helper_function_common_aaseq_analysis(df, lev_dist_accept, only_valid=False, verbose=True):
    # Step 1: Create patient-wise groups
    seqs_by_patient = df.groupby('patient_id')['AASeq'].unique().to_dict()

    # Step 2: Compare sequences across different patients
    valid_sequences = set()
    iterator = list(combinations(seqs_by_patient.items(), 2))
    if verbose and len(iterator) > 1:
        iterator = tqdm(iterator, total=len(iterator))
    for (pid1, seqs1), (pid2, seqs2) in iterator:
        # Sort seqs1 by length
        seqs1_by_len = {}
        for seq in seqs1:
            seq_len = len(seq)
            if seq_len not in seqs1_by_len:
                seqs1_by_len[seq_len] = []
            seqs1_by_len[seq_len].append(seq)

        # Convert lists to numpy arrays for efficiency
        seqs1_by_len = {k: np.array(v, dtype=object) for k, v in seqs1_by_len.items()}

        # Sort seqs2 by length for efficient filtering
        seqs2_by_len = np.array(sorted(seqs2, key=len), dtype=object)
        seqs2_lens = np.array([len(seq) for seq in seqs2_by_len])

        # Iterate over length groups in seqs1
        inner_itter = list(seqs1_by_len.items())
        if verbose:
            inner_itter = tqdm(inner_itter, total=len(inner_itter))
        for length, group1 in inner_itter:
            # Select seqs2 that are in the range [length-1, length+1] (or +- lev_dist_accept)
            min_len, max_len = length - lev_dist_accept, length + lev_dist_accept
            mask = (seqs2_lens >= min_len) & (seqs2_lens <= max_len)
            group2 = seqs2_by_len[mask]

            if len(group2) > 0:
                # Compute pairwise identity matrix
                pwc_mat = pairwise_scores(group1, group2, score=levenshtein_dist_non_bin)
                sim_inxs = np.where(pwc_mat <= lev_dist_accept)

                # Add matching sequences with patient_id to valid set
                for x, y in zip(sim_inxs[0], sim_inxs[1]):
                    if only_valid:
                        valid_sequences.add(group1[x])
                        valid_sequences.add(group2[y])
                    else:
                        valid_sequences.add((group1[x], pid1, pid2))  # Tuple (seq, originating pid x2)
                        valid_sequences.add((group2[y], pid1, pid2))  # Tuple (seq, originating pid x2)

    if only_valid:
        return valid_sequences

    # Step 3: Calculate the mean number of common sequences and percentages
    unique_patient_ids = df["patient_id"].unique()
    masks = []
    for patient in unique_patient_ids:
        # Get sequences that belong to the current patient
        patient_seqs = set(df.loc[df["patient_id"] == patient, "AASeq"])

        # Create a mask for sequences
        mask = np.array([1 if seq[0] in patient_seqs else 0 for seq in valid_sequences])
        masks.append(mask)

    # Convert to ndarray
    masks = np.array(masks)  # Shape: (num_unique_patients, len(sequences))
    return masks


def process_combination(combination, patient_sequences, df, lev_dist_accept, helper_function):
    selected_sequences = [patient_sequences[pid] for pid in combination]

    if lev_dist_accept >= 1:
        temp_df = df[df['patient_id'].isin(combination)]
        masks = helper_function(temp_df, lev_dist_accept)
        num_common = np.sum(np.any(masks == 1, axis=0))
        print(f"Using lev_dist_accept >= 1 for combination {combination}, there might be a problem!")
    else:
        common_sequences = set.intersection(*selected_sequences)
        num_common = len(common_sequences)

    total_sequences = sum(len(seqs) for seqs in selected_sequences)
    min_sequences = min(len(seqs) for seqs in selected_sequences)
    max_sequences = max(len(seqs) for seqs in selected_sequences)

    return {
        'num_total_seqs': total_sequences,
        'num_common': num_common,
        'percent_of_total': (num_common / total_sequences) * 100 if total_sequences > 0 else 0,
        'percent_of_min': (num_common / min_sequences) * 100 if min_sequences > 0 else 0,
        'percent_of_max': (num_common / max_sequences) * 100 if max_sequences > 0 else 0
    }


def run_multiprocess_analysis(patient_combinations, patient_sequences, df, lev_dist_accept, helper_function, num_processes=None):
    process_func = partial(process_combination, patient_sequences=patient_sequences,
                           df=df, lev_dist_accept=lev_dist_accept, helper_function=helper_function)

    with mp.Pool(processes=num_processes) as pool:
        results = pool.map(process_func, patient_combinations)

    percent_of_total_values = [r['percent_of_total'] for r in results]
    return results, percent_of_total_values

