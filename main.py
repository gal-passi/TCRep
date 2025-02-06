from Curation import Study
import pandas as pd
import numpy as np
import torch
import os
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import KFold
from tqdm import tqdm
from itertools import product
from Levenshtein import ratio  # pip install python-Levenshtein
from utils import seq_identity, calculate_distance_matrix, pairwise_scores
import pickle
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


STUDY_ID = 'PRJNA393498'
STUDY_ID2 = 'immunoSEQ47'
STUDIES = [STUDY_ID, STUDY_ID2]
VALID_SEQ_CACHE = "cache/valid_sequences"


def get_valid_seqs_original(df):
    valid_seqs = df.groupby('AASeq')['patient_id'].nunique()
    return valid_seqs[valid_seqs >= 2]


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
        embeds.append(logits_output.embeddings)  # can also output: logits_output.logits
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


def process_and_evaluate(syn, bld, pad, to_plot=True, cd_type='4'):
    # Prepare data
    X_syn = pad_and_flatten(syn, pad)
    X_bld = pad_and_flatten(bld, pad)

    X = np.vstack([X_syn, X_bld])
    y = np.array([0] * len(syn) + [1] * len(bld))

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Plotting if needed
    if to_plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(X_tsne[:len(syn), 0], X_tsne[:len(syn), 1],
                    c='blue', label='Synovial Fluid')
        plt.scatter(X_tsne[len(bld):, 0], X_tsne[len(bld):, 1],
                    c='red', label='Blood')
        plt.legend()
        plt.title(f't-SNE Visualization of CD{cd_type} Data')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.show()

    # K-fold cross validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, test_idx in kf.split(X_tsne):
        X_train, X_test = X_tsne[train_idx], X_tsne[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Train and evaluate KNN
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(X_train, y_train)
        score = accuracy_score(y_test, knn.predict(X_test))
        scores.append(score)

    return np.mean(scores), np.std(scores)


def get_cached_embeddings(sequences, name='', cache_dir="cache/esm_c/", embed_fn=None):
    """
    Load cached embeddings if available, otherwise compute and cache them.

    Args:
        sequences: List of sequences to embed
        cache_dir: Directory to store cached embeddings
        embed_fn: Function to compute embeddings if not cached

    Returns:
        numpy array of embeddings
    """
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


def get_valid_seqs(df, df_ind, name_opt):
    cache_pkl = os.path.join(VALID_SEQ_CACHE, f"valid_sequences_{df_ind}{name_opt}.pkl")
    if os.path.exists(cache_pkl):
        with open(cache_pkl, 'rb') as f:
            valid_sequences = pickle.load(f)
        return df[df['AASeq'].isin(valid_sequences)]['AASeq'].unique()
    else:
        return None


def calculate_valid_seqs(df, df_ind):
    # Step 1: Create patient-wise groups
    seqs_by_patient = df.groupby('patient_id')['AASeq'].unique().to_dict()
    # Step 2: Compare sequences across different patients
    valid_sequences = set()
    for (pid1, seqs1), (pid2, seqs2) in tqdm(product(seqs_by_patient.items(), repeat=2),
                                             total=len(seqs_by_patient) ** 2):
        if pid1 == pid2:
            continue  # Skip same patient

        # Calculate pairwise identity matrix
        pwc_mat = pairwise_scores(seqs1, seqs2, score=seq_identity)
        sim_inxs = np.where(pwc_mat >= 0.9)

        # Add matching sequences to valid set
        for x, y in zip(sim_inxs[0], sim_inxs[1]):
            valid_sequences.add(seqs1[x])
            valid_sequences.add(seqs2[y])

    cache_pkl = os.path.join(VALID_SEQ_CACHE, f"valid_sequences_{df_ind}_opt1.pkl")
    with open(cache_pkl, 'wb') as f:
        pickle.dump(valid_sequences, f)
    return valid_sequences


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
    parser.add_argument('--df_ind', type=int, default=0)
    parser.add_argument('--dist_option', type=int, default=0)
    parser.add_argument('--study_ind', type=int, default=0)
    args = parser.parse_args()
    df_ind = args.df_ind
    dist_option = args.dist_option
    study_ind = args.study_ind
    print("RUN CONFIGURATION:")
    print(f"df_ind: {df_ind}")
    print(f"dist_option: {dist_option}")
    print(f"study_ind: {study_ind}")
    print()

    if dist_option == 1:
        name_opt = '_opt1'
    else:
        name_opt = ''

    study = Study(STUDIES[study_ind])

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

    # keep only sequences that appear at least twice between different patients (AASeq that appear in different patient_ids)
    all_dfs = [df_cd4_syn, df_cd8_syn, df_cd4_bld, df_cd8_bld]
    all_valid_seqs = [get_valid_seqs(df, i, name_opt) for i, df in enumerate(all_dfs)]

    if any(item is None for item in all_valid_seqs):
        if all_valid_seqs[df_ind] is None:
            df = all_dfs[df_ind]
            print(f"Calculating Valid Sequences of DF: {df_ind}!")
            valid_sequences = calculate_valid_seqs(df, df_ind)
            print(f"Done calculating valid sequences of df: {df_ind}!")
            all_valid_seqs[df_ind] = valid_sequences
        else:
            for i, df in enumerate(all_dfs):
                if all_valid_seqs[i] is None:
                    print(f"Calculating Valid Sequences of DF: {i}!")
                    if dist_option == 0:
                        valid_sequences = get_valid_seqs_original(df)
                    else:
                        valid_sequences = calculate_valid_seqs(df, i)
                    print(f"Done calculating valid sequences of df: {i}!")
                    all_valid_seqs[i] = valid_sequences
    # OTHER OPTION: Use exact sequences that appear in two or more patients in the same Tissue
    #     # Filter sequences that appear in at least two different patients
    #     # TODO: Find out why are there less samples in my case than in Sapir's case
    #     out = [get_valid_seqs_original(d) for d in all_dfs]

    sr_cd4_syn_vld, sr_cd8_syn_vld, sr_cd4_bld_vld, sr_cd8_bld_vld = all_valid_seqs

    # TODO: Maybe need to remove samples that are only in blood but not in synovial!
    # Modified embedding code
    cd4_syn = get_cached_embeddings(list(sr_cd4_syn_vld), name='cd4_syn' + name_opt, embed_fn=embed)
    cd8_syn = get_cached_embeddings(list(sr_cd8_syn_vld), name='cd8_syn' + name_opt, embed_fn=embed)
    cd4_bld = get_cached_embeddings(list(sr_cd4_bld_vld), name='cd4_bld' + name_opt, embed_fn=embed)
    cd8_bld = get_cached_embeddings(list(sr_cd8_bld_vld), name='cd8_bld' + name_opt, embed_fn=embed)

    mean_acc, std_acc = process_and_evaluate(cd4_syn, cd4_bld, pad=22, cd_type="4")  # 20 is the max size for all seqs
    print(f"CD4 - KNN 3 neighbours: Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

    mean_acc, std_acc = process_and_evaluate(cd8_syn, cd8_bld, pad=22, cd_type="8")  # 20 is the max size for all seqs
    print(f"CD8 - KNN 3 neighbours: Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

    # seqs_cd4 = df_cd4["AASeq"].drop_duplicates().reset_index(drop=True)
    # # seqs_cd8 = df_cd8["AASeq"].drop_duplicates().reset_index(drop=True)
    #
    # # splitting randomly with 5 folds
    # kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # fold_inxs = []
    # for i, (train_index, test_index) in enumerate(kf.split(seqs_cd4)):
    #     fold_inxs.append((train_index, test_index))
    #
    # seqs_cd4[fold_inxs[0][1]]

    # do_k_fold(seqs_cd4)
