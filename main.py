import warnings
warnings.simplefilter("ignore", category=FutureWarning)
from Curation import Study
import pandas as pd
import numpy as np
import torch
import os
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from itertools import product
from Levenshtein import ratio  # pip install python-Levenshtein
from utils import seq_identity, calculate_distance_matrix, pairwise_scores, levenshtein_dist
import pickle
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import accuracy_score
from itertools import combinations


STUDY_ID = 'PRJNA393498'
STUDY_ID2 = 'immunoSEQ47'
STUDY_ID3 = 'immunoSEQ77'
HEALTHY_STUDY_ID = STUDY_ID3
STUDIES = [STUDY_ID, STUDY_ID2, STUDY_ID3]
VALID_SEQ_CACHE = "cache/valid_sequences"


def get_valid_seqs_original(df, df_ind, name_opt, study_name):
    if df_ind <= 1:
        valid_seqs = df.groupby('AASeq')['patient_id'].nunique()
        valid_seqs = valid_seqs[valid_seqs >= 2]
        valid_seqs = set(valid_seqs.index)
    else:
        valid_seqs = set(df['AASeq'].unique())

    base_path = os.path.join(VALID_SEQ_CACHE, study_name)
    os.makedirs(base_path, exist_ok=True)
    cache_pkl = os.path.join(base_path, f"valid_sequences_{df_ind}{name_opt}.pkl")
    with open(cache_pkl, 'wb') as f:
        pickle.dump(valid_seqs, f)
    return valid_seqs


def embed(seqs):
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embeds = list()
    # The size of the embedding is 960 for this model
    client = ESMC.from_pretrained("esmc_300m").to(device)
    for seq in tqdm(seqs):
        protein = ESMProtein(sequence=seq)
        protein_tensor = client.encode(protein)
        logits_output = client.logits(
            protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
        )
        embeds.append(logits_output.embeddings.cpu())  # can also output: logits_output.logits
    return embeds


def pad_and_flatten(tensor_list, pad):
    # Find max length
    max_len = max(max(tensor.shape[1] for tensor in tensor_list), pad)

    # Pad and flatten each tensor
    padded_data = []
    for tensor in tensor_list:
        pad_width = ((0, 0), (0, max_len - tensor.shape[1]), (0, 0))
        padded = np.pad(tensor, pad_width, mode='constant', constant_values=0)
        padded_data.append(padded.reshape(-1))

    return np.array(padded_data)


def process_and_evaluate(syn, healthy, bld, syn_mask, bld_mask, k_fold_type, study_name,
                         pad, to_plot=True, cd_type='4', name_opt=''):
    # Prepare data
    X_syn = torch.cat([x.sum(dim=1) for x in syn])
    X_bld = torch.cat([x.sum(dim=1) for x in bld])
    X_hlt = torch.cat([x.sum(dim=1) for x in healthy])
    # X_syn = pad_and_flatten(syn, pad)
    # X_bld = pad_and_flatten(bld, pad)
    # X_hlt = pad_and_flatten(healthy, pad)

    X = X_syn
    y = np.array([0] * len(syn))

    # Apply t-SNE
    # tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    # X_tsne = tsne.fit_transform(X)

    # Plotting if needed
    # if to_plot:
    #     plt.figure(figsize=(10, 6))
    #     plt.scatter(X_tsne[:len(syn), 0], X_tsne[:len(syn), 1],
    #                 c='blue', label='Synovial Fluid')
    #     plt.scatter(X_tsne[len(bld):, 0], X_tsne[len(bld):, 1],
    #                 c='red', label='Blood')
    #     plt.legend()
    #     plt.title(f't-SNE Visualization of CD{cd_type} Data')
    #     plt.xlabel('t-SNE 1')
    #     plt.ylabel('t-SNE 2')
    #     # save plot
    #     plots_folder = f"plots/{study_name}"
    #     os.makedirs(plots_folder, exist_ok=True)
    #     plt.savefig(os.path.join(plots_folder, f"tsne_cd{cd_type}{name_opt}.png"))
    #     plt.show()

    # K-fold cross validation
    if k_fold_type == 0:  # random k-fold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds_indices = kf.split(X)
        blood_indices = [x[0] for x in kf.split(X_bld)]
    else:  # patient k-fold
        folds_indices = [((np.delete(syn_mask, i, axis=0) == 1).any(axis=0), syn_mask[i] == 1)
                         for i in range(len(syn_mask))]
        blood_indices = [mask == 1 for mask in bld_mask]

    scores = []
    for i, (train_idx, test_idx) in enumerate(folds_indices):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Adding the same healthy data to the training set (for each fold)
        # X_hlt_rnd = X_hlt
        X_hlt_rnd = X_hlt[np.random.choice(len(X_hlt), len(X_train) * 2, replace=False)]
        X_train = np.vstack([X_train, X_hlt_rnd])
        y_train = np.hstack([y_train, np.array([1] * len(X_hlt_rnd))])

        # Adding random samples from blood to the test set (of the same size as positive samples in the test set)
        X_bld_rnd = X_bld[blood_indices[i]]
        # X_bld_rnd = X_bld
        # X_bld_rnd = X_bld[np.random.choice(len(X_bld), len(y_test) * 1, replace=False)]
        X_test = np.vstack([X_test, X_bld_rnd])
        y_test = np.hstack([y_test, np.array([1] * len(X_bld_rnd))])

        # X_stack = np.vstack([X_train, X_test])
        # tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        # X_tsne = tsne.fit_transform(X_stack)
        # X_train, X_test = X_tsne[:len(X_train)], X_tsne[len(X_train):]

        # Train and evaluate KNN

        # print((y_train == 0).sum(), (y_train == 1).sum(), (y_test == 0).sum(), (y_test == 1).sum())
        knn = KNeighborsClassifier(n_neighbors=3, metric="minkowski")
        knn.fit(X_train, y_train)
        score = accuracy_score(y_test, knn.predict(X_test))
        scores.append(score)

    return np.mean(scores), np.std(scores)


def get_cached_embeddings(sequences, study_name, name='', cache_dir="cache/esm_c/", embed_fn=None):
    """
    Load cached embeddings if available, otherwise compute and cache them.

    Args:
        sequences: List of sequences to embed
        cache_dir: Directory to store cached embeddings
        embed_fn: Function to compute embeddings if not cached

    Returns:
        numpy array of embeddings
    """
    cache_dir = os.path.join(cache_dir, study_name)
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"embeds_{name}.pkl")

    # Load with pickle
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    if embed_fn is None:
        raise ValueError("embed_fn must be provided if embeddings are not cached")

    # Compute embeddings
    embeddings = embed_fn(sequences)

    # saving with pickle
    with open(cache_file, 'wb') as f:
        pickle.dump(embeddings, f)
    return embeddings


def get_valid_seqs(df, df_ind, name_opt, study_name):
    if df_ind >= 2:
        valid_seqs = set(df['AASeq'].unique())
        return valid_seqs

    base_path = os.path.join(VALID_SEQ_CACHE, study_name)
    os.makedirs(base_path, exist_ok=True)
    cache_pkl = os.path.join(base_path, f"valid_sequences_{df_ind}{name_opt}.pkl")
    if os.path.exists(cache_pkl):
        with open(cache_pkl, 'rb') as f:
            valid_sequences = pickle.load(f)
        return df[df['AASeq'].isin(valid_sequences)]['AASeq'].unique()
    else:
        return None


def calculate_valid_seqs(df, df_ind, name_opt, study_name):
    # Step 1: Create patient-wise groups
    seqs_by_patient = df.groupby('patient_id')['AASeq'].unique().to_dict()

    # Step 2: Compare sequences across different patients
    valid_sequences = set()
    for (pid1, seqs1), (pid2, seqs2) in tqdm(combinations(seqs_by_patient.items(), 2),
                                             total=len(seqs_by_patient) * (len(seqs_by_patient) - 1) // 2):
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
        for length, group1 in seqs1_by_len.items():
            # Select seqs2 that are in the range [length-2, length+2]
            min_len, max_len = length - 1, length + 1
            mask = (seqs2_lens >= min_len) & (seqs2_lens <= max_len)
            group2 = seqs2_by_len[mask]

            if len(group2) > 0:
                # Compute pairwise identity matrix
                pwc_mat = pairwise_scores(group1, group2, score=levenshtein_dist)
                # pwc_mat = pairwise_scores(group1, group2, score=seq_identity)
                sim_inxs = np.where(pwc_mat >= 0.9)

                # Add matching sequences to valid set
                for x, y in zip(sim_inxs[0], sim_inxs[1]):
                    valid_sequences.add(group1[x])
                    valid_sequences.add(group2[y])

    base_path = os.path.join(VALID_SEQ_CACHE, study_name)
    os.makedirs(base_path, exist_ok=True)
    cache_pkl = os.path.join(base_path, f"valid_sequences_{df_ind}{name_opt}.pkl")
    with open(cache_pkl, 'wb') as f:
        pickle.dump(valid_sequences, f)
    return valid_sequences


def plot_length_histogram(arr, title_text="sequences"):
    """
    Plots a histogram of string lengths in a given ndarray.

    Parameters:
        arr (numpy.ndarray): Array of strings.
        title_text (str): Custom text to insert in the title.
    """
    lengths = np.vectorize(len)(arr)

    plt.hist(lengths, bins=range(min(lengths), max(lengths) + 2), edgecolor='black', alpha=0.7)
    plt.xlabel("String Length")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {title_text} lengths")
    plt.xticks(range(min(lengths), max(lengths) + 1))
    plt.show()


def bound_and_sample_blood(syn, blood):
    lengths = set([len(x) for x in syn])
    min_len, max_len = min(lengths), max(lengths)
    blood = np.array(sorted(list(blood)))
    mask = np.vectorize(lambda s: min_len <= len(s) <= max_len)(blood)
    blood = blood[mask]
    blood = np.random.choice(blood, len(syn) * 2, replace=False)
    return blood


def get_patient_ids_masks(df, sequences):
    unique_patient_ids = sorted(df["patient_id"].unique())
    masks = []
    for patient in unique_patient_ids:
        # Get sequences that belong to the current patient
        patient_seqs = set(df.loc[df["patient_id"] == patient, "AASeq"])
        # Create a mask for sequences
        mask = np.array([1 if seq in patient_seqs else 0 for seq in sequences])
        masks.append(mask)
    # Convert to ndarray
    masks = np.array(masks)  # Shape: (num_unique_patients, len(sequences))
    return masks


if __name__ == '__main__':
    # return numpy array of unique sequences - see docstring
    # sequence = study.build_train_representations(samples=None, save=False, path=None)
    # print(sequence)

    # study = Study(STUDY_ID2)
    # # return numpy array of unique sequences - see docstring
    # sequence = study.build_train_representations(samples=None, save=False, path=None)
    # print(sequence)

    # use esm_2 to extract representation
    # model_checkpoint = "facebook/esm2_t4815B__UR50D"
    # pad to length of the longest sequence - can be done through esm parameters (no need to change sequences)
    # save a dictionary of {seq : representation} - make sure to save on lab folder may be heavy


    # get df_ind from program arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_ind', type=int, default=-1)
    parser.add_argument('--dist_option', type=int, default=0)
    parser.add_argument('--study_ind', type=int, default=0)
    parser.add_argument('--k_fold_type', type=int, default=0)
    args = parser.parse_args()
    df_ind = args.df_ind
    dist_option = args.dist_option
    study_ind = args.study_ind
    k_fold_type = args.k_fold_type
    print("RUN CONFIGURATION:")
    print(f"\tdf_ind: {df_ind}")
    print(f"\tdist_option: {dist_option}")
    print(f"\tstudy_ind: {study_ind}")
    print(f"\tk_fold_type: {k_fold_type}")  # 0 - random, 1 - patient
    print("\n")

    name_opt = '_opt1' if dist_option == 1 else ''

    study = Study(STUDIES[study_ind])
    study_healthy = Study(HEALTHY_STUDY_ID)

    # reading synovial samples
    samples_syn = study._samples['usable']
    df_syn = study.read_sample(samples_syn)
    df_cd4_syn = df_syn[df_syn['cell_type'] == 'CD4']
    df_cd8_syn = df_syn[df_syn['cell_type'] == 'CD8']
    # reading blood samples
    samples_bld = study._samples['uncertain']
    df_bld = study.read_sample(samples_bld)
    df_cd4_bld = df_bld[df_bld['cell_type'] == 'CD4']
    df_cd4_bld = df_cd4_bld[df_cd4_bld['patient_id'].isin(df_cd4_syn['patient_id'].unique())]
    df_cd8_bld = df_bld[df_bld['cell_type'] == 'CD8']
    df_cd8_bld = df_cd8_bld[df_cd8_bld['patient_id'].isin(df_cd8_syn['patient_id'].unique())]

    # reading healthy study:
    samples_h = study_healthy._samples['usable']
    df_h = study_healthy.read_sample(samples_h)
    df_h_cd8 = df_h[df_h['cell_type'] == 'CD8']
    valid_seqs_cd8_h = df_h_cd8["AASeq"].unique()

    # keep only sequences that appear at least twice between different patients (AASeq that appear in different patient_ids)
    all_dfs = [df_cd4_syn, df_cd8_syn, df_cd4_bld, df_cd8_bld]
    all_valid_seqs = [get_valid_seqs(df, i, name_opt, study.name) for i, df in enumerate(all_dfs)]

    if any(item is None for item in all_valid_seqs):
        if df_ind != -1 and all_valid_seqs[df_ind] is None:
            df = all_dfs[df_ind]
            print(f"Calculating Valid Sequences of DF: {df_ind}!")
            if dist_option == 0:
                valid_sequences = get_valid_seqs_original(df, df_ind, name_opt, study.name)
            else:
                valid_sequences = calculate_valid_seqs(df, df_ind, name_opt, study.name)
            print(f"Done calculating valid sequences of df: {df_ind}!")
            all_valid_seqs[df_ind] = valid_sequences
        else:
            for i, df in enumerate(all_dfs):
                if all_valid_seqs[i] is None:
                    print(f"Calculating Valid Sequences of DF: {i}!")
                    if dist_option == 0:
                        valid_sequences = get_valid_seqs_original(df, i, name_opt, study.name)
                    else:
                        valid_sequences = calculate_valid_seqs(df, i, name_opt, study.name)
                    print(f"Done calculating valid sequences of df: {i}!")
                    all_valid_seqs[i] = valid_sequences

    sr_cd4_syn_vld, sr_cd8_syn_vld, sr_cd4_bld_vld, sr_cd8_bld_vld = all_valid_seqs

    # bounding the length of sequences to be min and max of synovial samples,
    # then sampling *2 samples from blood to match *2 the number of synovial samples
    np.random.seed(42)
    sr_cd4_bld_vld = bound_and_sample_blood(sr_cd4_syn_vld, sr_cd4_bld_vld)
    sr_cd8_bld_vld = bound_and_sample_blood(sr_cd8_syn_vld, sr_cd8_bld_vld)
    valid_seqs_cd8_h = bound_and_sample_blood(sr_cd8_syn_vld, valid_seqs_cd8_h)

    # getting patient id masks in order to do k-fold by patient (according to synovial samples)
    cd4_syn_patient_id_masks = get_patient_ids_masks(df_cd4_syn, sr_cd4_syn_vld)
    cd4_bld_patient_id_masks = get_patient_ids_masks(df_cd4_bld, sr_cd4_bld_vld)
    cd8_syn_patient_id_masks = get_patient_ids_masks(df_cd8_syn, sr_cd8_syn_vld)
    cd8_bld_patient_id_masks = get_patient_ids_masks(df_cd8_bld, sr_cd8_bld_vld)

    # Calculating embeddings (or loading if it is available)
    cd4_syn = get_cached_embeddings(list(sr_cd4_syn_vld), study.name, name='cd4_syn' + name_opt, embed_fn=embed)
    cd8_syn = get_cached_embeddings(list(sr_cd8_syn_vld), study.name, name='cd8_syn' + name_opt, embed_fn=embed)
    cd4_bld = get_cached_embeddings(list(sr_cd4_bld_vld), study.name, name='cd4_bld' + name_opt, embed_fn=embed)
    cd8_bld = get_cached_embeddings(list(sr_cd8_bld_vld), study.name, name='cd8_bld' + name_opt, embed_fn=embed)

    cd8_h = get_cached_embeddings(list(valid_seqs_cd8_h), study.name, name='cd8_h' + name_opt, embed_fn=embed)

    # cd4_syn = [x for x in cd4_syn if x.shape[1] <= 20]
    # cd4_bld = [x for x in cd4_bld if x.shape[1] <= 20]
    # print(f"Samples CD4 Synovial: {len(cd4_syn)}, CD4 Blood: {len(cd4_bld)}")
    # mean_acc, std_acc = process_and_evaluate(cd4_syn, cd4_bld, study_name=study.name, pad=20, cd_type="4", name_opt=name_opt)  # 20 is the max size for all seqs
    # print(f"CD4 - KNN 3 neighbours: Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    # print()


    # cd8_syn = [x for x in cd8_syn if x.shape[1] <= 17]
    # cd8_bld = [x for x in cd8_bld if x.shape[1] <= 17]
    # cd8_h = [x for x in cd8_h if x.shape[1] <= 17]
    print(f"Samples CD8 Synovial: {len(cd8_syn)}, CD8 Blood: {len(cd8_bld)}")
    mean_acc, std_acc = process_and_evaluate(cd8_syn, cd8_h, cd8_bld,
                                             cd8_syn_patient_id_masks, cd8_bld_patient_id_masks,
                                             k_fold_type,
                                             study_name=study.name, pad=17, cd_type="8", name_opt=name_opt)  # 20 is the max size for all seqs
    # mean_acc, std_acc = process_and_evaluate(cd8_syn, cd8_bld, study_name=study.name, pad=20, cd_type="8", name_opt=name_opt)  # 20 is the max size for all seqs
    print(f"CD8 - KNN 3 neighbours: Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
