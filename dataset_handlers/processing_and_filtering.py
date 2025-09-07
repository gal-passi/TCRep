import numpy as np
import pandas as pd
import pickle
from collections import Counter
from dataset_handlers.constants import *


def apply_extra_filter(df_bld, df_hlt, top_n_seqs):
    # Filtering sequences by length
    df_bld = df_bld[df_bld['AASeq'].str.len() > 10]  # remove sequences that are too short
    df_bld = df_bld[df_bld['AASeq'].str.len() < 20]  # remove sequences that are too long
    df_hlt = df_hlt[df_hlt['AASeq'].str.len() > 10]  # remove sequences that are too short
    df_hlt = df_hlt[df_hlt['AASeq'].str.len() < 20]  # remove sequences that are too long

    # Filtering patients with 5k sequences less than top_n_seqs (if it exists)
    if top_n_seqs is not None:
        # Each patient with unique AASeqs less than 1000 * top_n_seqs should be completely removed:
        min_seq_count = 1000 * top_n_seqs - 5000
        df_bld = df_bld.groupby('patient_id').filter(lambda x: len(x['AASeq'].unique()) >= min_seq_count)
        df_hlt = df_hlt.groupby('patient_id').filter(lambda x: len(x['AASeq'].unique()) >= min_seq_count)
    return df_bld, df_hlt


def validate_required_columns(df_bld, df_hlt):
    required_cols = ['AASeq', 'cloneFraction', 'Vregion', 'Dregion', 'Jregion',
                     'RunId', 'patient_id', 'tissue', 'cell_type', 'condition', 'study_id']
    for col in required_cols:
        if col not in df_bld.columns or col not in df_hlt.columns:
            raise ValueError(f"Column '{col}' is missing from one of the dataframes!")
    df_bld = df_bld[required_cols]
    df_hlt = df_hlt[required_cols]
    return df_bld, df_hlt


def split_patients(df_bld, num_test_patients, unique_patient_ids=None, run_on_full_data=False):
    if unique_patient_ids is None:
        unique_patient_ids = df_bld["patient_id"].unique()
        unique_patient_ids = np.random.permutation(unique_patient_ids)
    test_patient_ids = unique_patient_ids[:num_test_patients // 2]
    valid_patient_ids = unique_patient_ids[num_test_patients // 2:num_test_patients]
    train_patient_ids = unique_patient_ids[num_test_patients:]

    if run_on_full_data:
        # Add to the self.df_bld the valid and test patient ids with added ending to their patient_ids
        valid_df = df_bld[df_bld['patient_id'].isin(valid_patient_ids)].copy()
        valid_df['patient_id'] = valid_df['patient_id'].astype(str) + '_full'
        test_df = df_bld[df_bld['patient_id'].isin(test_patient_ids)].copy()
        test_df['patient_id'] = test_df['patient_id'].astype(str) + '_full'
        df_bld = pd.concat([df_bld, valid_df, test_df], axis=0, ignore_index=True)

    return train_patient_ids, valid_patient_ids, test_patient_ids, unique_patient_ids, df_bld


def find_all_common_sequences(df, num_of_patients=3):
    # Step 1: Group by 'patient_id' and get unique AASeqs
    grouped = df.groupby('patient_id')['AASeq'].unique()

    # Step 2: Count occurrences of each AASeq across different patient groups
    aa_seq_counter = Counter()
    for aa_seqs in grouped:
        aa_seq_counter.update(aa_seqs)

    # Step 3: Filter AASeqs that appear in at least num_of_patients different patients
    valid_aa_seqs = {aa_seq for aa_seq, count in aa_seq_counter.items() if count >= num_of_patients}

    return valid_aa_seqs


def generate_neighbors(sequences, valid_letters):
    valid_letters = set(valid_letters)  # Ensure valid letters are a set for quick lookup
    neighbor_set = set(sequences)  # Start with the original sequences

    for seq in sequences:
        seq_len = len(seq)

        # Generate substitutions
        for i in range(seq_len):
            for letter in valid_letters:
                if seq[i] != letter:  # Avoid replacing with the same letter
                    neighbor_set.add(seq[:i] + letter + seq[i + 1:])

        # Generate insertions
        for i in range(seq_len + 1):
            for letter in valid_letters:
                neighbor_set.add(seq[:i] + letter + seq[i:])

        # Generate deletions
        if seq_len > 1:  # Ensure we don't delete the only character
            for i in range(seq_len):
                neighbor_set.add(seq[:i] + seq[i + 1:])

    return neighbor_set


def generate_full_neighbors(seqs, valid_letters):
    all_neighbors = set()
    length_groups = {}
    for seq in seqs:
        length_groups.setdefault(len(seq), set()).add(seq)
    # Process each length group separately
    for seq_len, seq_group in length_groups.items():
        # Generate neighbors for this group
        neighbors = generate_neighbors(seq_group, valid_letters)
        all_neighbors.update(neighbors)
    return all_neighbors


# Loading all valid sequences for disease and healthy samples
def get_positive_negative(df_bld, df_hlt, df_name, dataset_type, cell_type, num_of_patients=3, num_of_healthy=3, verbose=True):
    all_common_seqs = find_all_common_sequences(df_bld, num_of_patients=num_of_patients)
    valid_seqs_healthy = find_all_common_sequences(df_hlt, num_of_patients=num_of_healthy)
    all_common_seqs = all_common_seqs - valid_seqs_healthy

    # TODO: The following code was added to try to speed up the process of finding valid sequences!
    #  Check that it works the same!
    # Faster way to get the same results (ONLY WHEN LEV DISTANCE IS 1):
    valid_seqs_disease = set(all_common_seqs)  # D*
    valid_seqs_healthy = set(valid_seqs_healthy)  # H*

    seqs_disease_neighbours = generate_full_neighbors(valid_seqs_disease, valid_letters=set(''.join(valid_seqs_disease)))  # Id (unfiltered)
    seqs_disease_neighbours = set(seqs_disease_neighbours).intersection(set(df_bld['AASeq'].unique()))  # Id
    seqs_healthy_neighbours = generate_full_neighbors(valid_seqs_healthy, valid_letters=set(''.join(valid_seqs_healthy)))  # Ih

    # return (D* \ H*) U (Id \ Ih), H*
    positive_seqs = set(valid_seqs_disease) - set(valid_seqs_healthy)
    positive_seqs = positive_seqs.union(seqs_disease_neighbours - seqs_healthy_neighbours)
    negative_seqs = set(valid_seqs_healthy)
    if verbose:
        print(f"Valid Disease Sequence (num of common = {num_of_patients}): {len(positive_seqs)}")

    return positive_seqs, negative_seqs


def calculate_pos_neg_sequences(df_bld, df_hlt, cache_sequences_path, df_name, patient_ids, dataset_type, cell_type, top_percent, top_n_seqs,
                                num_of_patients=3, num_of_healthy=3, filter_to_inflate=True, verbose=True):
    if filter_to_inflate:
        lev_dist_accept = 1  # for now its always lev distance 1
        save_folder = cache_sequences_path
        save_name = f"{df_name}_disease_{dataset_type}_{cell_type}_neighbours{num_of_patients}"
        if num_of_healthy != 3:
            save_name += f"_healthy{num_of_healthy}"
        if top_percent is not None:
            save_name += f"_top_{top_percent}"
        if top_n_seqs is not None:
            save_name += f"_top_n_{top_n_seqs}"
        save_file = os.path.join(save_folder, f"{save_name}_valid_seqs_dist_{lev_dist_accept}.pkl")
        # Check if the file already exists
        if os.path.exists(save_file):
            with open(save_file, 'rb') as f:
                positive_seqs, negative_seqs = pickle.load(f)
            if verbose:
                print(f"Loaded valid sequences from {save_file}")
            return positive_seqs, negative_seqs
        else:
            df_bld = df_bld[df_bld['patient_id'].isin(patient_ids)]
            positive_seqs, negative_seqs = get_positive_negative(df_bld, df_hlt, df_name, dataset_type, cell_type, num_of_patients=num_of_patients, num_of_healthy=num_of_healthy, verbose=verbose)
            # Save the positive and negative sequences to a file
            os.makedirs(save_folder, exist_ok=True)
            with open(save_file, 'wb') as f:
                pickle.dump((positive_seqs, negative_seqs), f)
            if not os.path.exists(save_file):
                print(f"Failed to save valid sequences to {save_file}")
        return positive_seqs, negative_seqs
    else:
        temp_df = df_bld[df_bld['patient_id'].isin(patient_ids)]
        grouped = temp_df.groupby('patient_id')['AASeq'].unique()
        aa_seq_counter = Counter()
        for aa_seqs in grouped:
            aa_seq_counter.update(aa_seqs)
        valid_aa_seqs = {aa_seq for aa_seq, count in aa_seq_counter.items() if count >= num_of_patients}

        temp_df_h = df_hlt
        grouped_h = temp_df_h.groupby('patient_id')['AASeq'].unique()
        aa_seq_counter_h = Counter()
        for aa_seqs in grouped_h:
            aa_seq_counter_h.update(aa_seqs)
        valid_aa_seqs_h = {aa_seq for aa_seq, count in aa_seq_counter_h.items() if count >= num_of_healthy}

        positive_seqs = np.array(list(set(valid_aa_seqs) - set(valid_aa_seqs_h)))
        negative_seqs = np.array(list(set(valid_aa_seqs_h) - set(valid_aa_seqs)))
        return positive_seqs, negative_seqs


def calculate_positive_sequences(df_bld, df_hlt, dataset_type, cache_sequences_path, train_ids, valid_ids, test_ids,
                                 filter_num_of_patients, filter_num_of_healthy, filter_to_inflate,
                                 extra_filter, k_fold, top_percent, top_n_seqs, verbose):
    """Compute train/valid/test pos and neg sequences with overlaps resolved."""
    if k_fold > 0:
        name_metadata = f"_fold_{k_fold}"
    else:
        name_metadata = ""
    if 'article_sle' in dataset_type or 't1d' in dataset_type:
        num_of_patients = filter_num_of_patients
        num_of_healthy = filter_num_of_healthy
    else:
        num_of_patients = filter_num_of_patients
        num_of_healthy = filter_num_of_healthy
    if not filter_to_inflate:
        name_metadata += "_no_inflate"
    if extra_filter:
        name_metadata += "_extra_filter"
    if verbose:
        print('Calculating positive and negative sequences...')

    train_pos_seqs, _ = calculate_pos_neg_sequences(df_bld, df_hlt, cache_sequences_path, "train" + name_metadata, train_ids,
                                                         dataset_type, "ALL", top_percent, top_n_seqs, num_of_patients=num_of_patients,
                                                         num_of_healthy=num_of_healthy,
                                                         filter_to_inflate=filter_to_inflate, verbose=verbose)
    # Calculate positive valid sequences
    train_and_valid_ids = np.concatenate((train_ids, valid_ids))
    valid_pos_seqs, _ = calculate_pos_neg_sequences(df_bld, df_hlt, cache_sequences_path, "valid" + name_metadata, train_and_valid_ids,
                                                         dataset_type, "ALL", top_percent, top_n_seqs, num_of_patients=num_of_patients,
                                                         num_of_healthy=num_of_healthy,
                                                         filter_to_inflate=filter_to_inflate, verbose=verbose)
    valid_bld_seqs = df_bld[df_bld['patient_id'].isin(valid_ids)]["AASeq"].unique()
    valid_pos_seqs = np.array(list(set(valid_pos_seqs) & set(valid_bld_seqs)))
    # Calculate positive test sequences
    train_and_test_ids = np.concatenate((train_ids, test_ids))
    test_pos_seqs, _ = calculate_pos_neg_sequences(df_bld, df_hlt, cache_sequences_path, "test" + name_metadata, train_and_test_ids,
                                                        dataset_type, "ALL", top_percent, top_n_seqs, num_of_patients=num_of_patients,
                                                        num_of_healthy=num_of_healthy,
                                                        filter_to_inflate=filter_to_inflate, verbose=verbose)
    test_bld_seqs = df_bld[df_bld['patient_id'].isin(test_ids)]["AASeq"].unique()
    test_pos_seqs = np.array(list(set(test_pos_seqs) & set(test_bld_seqs)))

    return train_pos_seqs, valid_pos_seqs, test_pos_seqs


def build_patient_masks(df_bld, unique_patient_ids, positive_seqs, train_ids, valid_ids, test_ids):
    masks = []
    for patient in unique_patient_ids:
        # Get sequences that belong to the current patient
        patient_seqs = set(df_bld.loc[df_bld["patient_id"] == patient, "AASeq"])
        # Create a mask for sequences
        mask = np.array([1 if seq in patient_seqs else 0 for seq in positive_seqs])
        if 1 in mask:
            masks.append(mask)
        else:
            print(f"Patient {patient} has NO positive sequences! Look into this case!")
            masks.append(mask)
    # Convert to ndarray
    patient_id_masks = np.array(masks)  # Shape: (num_unique_patients, len(positive_seqs))

    # translate back to the inds according to unique_patient_ids
    test_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in test_ids])
    valid_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in valid_ids])
    train_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in train_ids])

    # Get the masks for the test and train patients
    test_masks = patient_id_masks[test_patient_inds]
    valid_masks = patient_id_masks[valid_patient_inds]
    train_masks = patient_id_masks[train_patient_inds]
    test_inds = test_masks.any(axis=0)
    valid_inds = valid_masks.any(axis=0)
    valid_test_inds = test_inds & valid_inds
    if sum(valid_inds) > sum(test_inds):
        test_inds = test_inds | valid_test_inds
        valid_inds = valid_inds & ~valid_test_inds
    else:
        valid_inds = valid_inds | valid_test_inds
        test_inds = test_inds & ~valid_test_inds
    train_inds = (~test_inds) & (~valid_inds)

    return train_inds, valid_inds, test_inds, train_masks, valid_masks, test_masks, patient_id_masks, test_patient_inds, valid_patient_inds, train_patient_inds


def finalize_pos_neg_seqs(df_bld, train_patient_ids, valid_patient_ids, test_patient_ids,
                          positive_seqs, train_inds, valid_inds, test_inds):
    # Get the positive sequences for the test and train sets
    test_pos_seqs = np.array(positive_seqs)[test_inds]
    valid_pos_seqs = np.array(positive_seqs)[valid_inds]
    train_pos_seqs = np.array(positive_seqs)[train_inds]
    # Get the negative sequences
    neg_seqs = df_bld[df_bld['patient_id'].isin(train_patient_ids)]['AASeq'].unique()
    test_neg_seqs = df_bld[df_bld['patient_id'].isin(test_patient_ids)]['AASeq'].unique()
    valid_neg_seqs = df_bld[df_bld['patient_id'].isin(valid_patient_ids)]['AASeq'].unique()
    # Remove positive sequences from the negative sequences
    neg_seqs = np.array(list(set(neg_seqs) - set(positive_seqs)))
    test_neg_seqs = np.array(list(set(test_neg_seqs) - set(positive_seqs)))
    valid_neg_seqs = np.array(list(set(valid_neg_seqs) - set(positive_seqs)))
    # Get all sequences that are in valid_neg_seqs and valid_neg_seqs
    valid_test_neg_seqs = np.array(list(set(test_neg_seqs) & set(valid_neg_seqs)))
    if len(test_neg_seqs) < len(valid_neg_seqs):
        # remove valid_test_neg_seqs from valid_neg_seqs
        valid_neg_seqs = valid_neg_seqs[~np.isin(valid_neg_seqs, valid_test_neg_seqs)]
    else:
        # remove valid_test_neg_seqs from test_neg_seqs
        test_neg_seqs = test_neg_seqs[~np.isin(test_neg_seqs, valid_test_neg_seqs)]
    # Remove all valid_neg_seqs and test_neg_seqs sequences from the negative sequences
    neg_seqs = np.array(list(set(neg_seqs) - set(np.concatenate((valid_neg_seqs, test_neg_seqs)))))

    return train_pos_seqs, valid_pos_seqs, test_pos_seqs, neg_seqs, valid_neg_seqs, test_neg_seqs
