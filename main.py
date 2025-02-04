from Curation import Study
import pandas as pd
import numpy as np
import torch
import os
from Bio import pairwise2
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


# Function to compute global sequence identity
def sequence_identity(seq1, seq2):
    alignments = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)
    score = alignments[0].score  # Number of matching characters
    return score / max(len(seq1), len(seq2))  # Normalize by longest sequence


# TODO: Too slow because there are too many sequences
def do_k_fold(seqs):
    # Compute pairwise identity matrix
    n = len(seqs)
    identity_matrix = np.zeros((n, n))

    for i in tqdm(range(n)):
        for j in range(i + 1, n):
            identity_matrix[i, j] = identity_matrix[j, i] = sequence_identity(seqs[i], seqs[j])

    # Cluster sequences based on identity (adjust threshold as needed)
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.9, affinity='precomputed',
                                         linkage='complete')
    clusters = clustering.fit_predict(1 - identity_matrix)  # Convert similarity to distance (1 - identity)

    # Assign clusters to folds
    unique_clusters = np.unique(clusters)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = {i: [] for i in range(5)}

    for fold_idx, cluster_idx in enumerate(kf.split(unique_clusters)):
        for cluster in unique_clusters[cluster_idx[1]]:
            folds[fold_idx].extend(np.where(clusters == cluster)[0])

    # Display fold assignments
    for fold, indices in folds.items():
        print(f"Fold {fold + 1}: {[seqs[i] for i in indices]}")


def get_valid_seqs(df):
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


def process_and_evaluate(cd4_syn, cd4_bld, pad, to_plot=True):
    # Prepare data
    X_syn = pad_and_flatten(cd4_syn, pad)
    X_bld = pad_and_flatten(cd4_bld, pad)

    X = np.vstack([X_syn, X_bld])
    y = np.array([0] * len(cd4_syn) + [1] * len(cd4_bld))

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    X_tsne = tsne.fit_transform(X)

    # Plotting if needed
    if to_plot:
        plt.figure(figsize=(10, 6))
        plt.scatter(X_tsne[:len(cd4_syn), 0], X_tsne[:len(cd4_syn), 1],
                    c='blue', label='Synovial Fluid')
        plt.scatter(X_tsne[len(cd4_syn):, 0], X_tsne[len(cd4_syn):, 1],
                    c='red', label='Blood')
        plt.legend()
        plt.title('t-SNE Visualization of CD4 Data')
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


def get_cached_embeddings(sequences, cache_dir="cache/esm_c/", embed_fn=None):
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
    cache_file = os.path.join(cache_dir, f"embeddings_{hash(tuple(sequences))}.npy")

    if os.path.exists(cache_file):
        return np.load(cache_file)

    if embed_fn is None:
        raise ValueError("embed_fn must be provided if embeddings are not cached")

    embeddings = embed_fn(sequences)
    np.save(cache_file, embeddings)
    return embeddings


if __name__ == '__main__':
    study = Study(STUDY_ID)
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
    args = parser.parse_args()
    df_ind = args.df_ind
    dist_option = args.dist_option
    print(f"df_ind: {df_ind}")
    print(f"dist_option: {dist_option}")

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

    df = all_dfs[df_ind]
    valid_sequences = set()

    # TODO: Method 0 is probably very slow when compared to Method 1!
    if dist_option == 0:
        # Step 1: Create patient-wise groups
        seqs_by_patient = df.groupby('patient_id')['AASeq'].unique().to_dict()

        # Step 2: Compare sequences across different patients
        valid_sequences = set()

        for (pid1, seqs1), (pid2, seqs2) in tqdm(product(seqs_by_patient.items(), repeat=2),
                                                 total=len(seqs_by_patient) ** 2):
            if pid1 == pid2:
                continue  # Skip same patient

            for seq1 in tqdm(seqs1):
                for seq2 in seqs2:
                    if sequence_identity(seq1, seq2) >= 0.9:  # 90% sequence identity threshold
                    # if ratio(seq1, seq2) >= 0.9:  # 90% sequence identity
                        valid_sequences.add(seq1)
                        valid_sequences.add(seq2)

        # save valid sequences as pkl
        with open(f'valid_sequences_{df_ind}.pkl', 'wb') as f:
            pickle.dump(valid_sequences, f)
        print('done!')
    elif dist_option == 1:
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

        with open(f'valid_sequences_{df_ind}_opt1.pkl', 'wb') as f:
            pickle.dump(valid_sequences, f)
    else:
        # Filter sequences that appear in at least two different patients
        out = [get_valid_seqs(d) for d in all_dfs]
        sr_cd4_syn_vld, sr_cd8_syn_vld, sr_cd4_bld_vld, sr_cd8_bld_vld = out
        # df_cd4_syn_vld, df_cd8_syn_vld, df_cd4_bld_vld, df_cd8_bld_vld = df_cd4_syn[df_cd4_syn['AASeq'].isin(sr_cd4_syn_vld.index)], \
        #                                                                   df_cd8_syn[df_cd8_syn['AASeq'].isin(sr_cd8_syn_vld.index)], \
        #                                                                     df_cd4_bld[df_cd4_bld['AASeq'].isin(sr_cd4_bld_vld.index)], \
        #                                                                     df_cd8_bld[df_cd8_bld['AASeq'].isin(sr_cd8_bld_vld.index)]

    # Step 3: Keep only matching sequences
    # filtered_df = df[df['AASeq'].isin(valid_sequences)]

    # TODO: Maybe need to remove samples that are only in blood but not in synovial!

    # cd4_syn = embed(list(sr_cd4_syn_vld.index))
    # cd8_syn = embed(list(sr_cd8_syn_vld.index))
    # cd4_bld = embed(list(sr_cd4_bld_vld.index))
    # cd8_bld = embed(list(sr_cd8_bld_vld.index))
    # Modified embedding code
    cd4_syn = get_cached_embeddings(list(sr_cd4_syn_vld.index), embed_fn=embed)
    cd8_syn = get_cached_embeddings(list(sr_cd8_syn_vld.index), embed_fn=embed)
    cd4_bld = get_cached_embeddings(list(sr_cd4_bld_vld.index), embed_fn=embed)
    cd8_bld = get_cached_embeddings(list(sr_cd8_bld_vld.index), embed_fn=embed)

    mean_acc, std_acc = process_and_evaluate(cd4_syn, cd4_bld, pad=20)  # 20 is the max size for all seqs
    print(f"CD4 - KNN 3 neighbours: Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

    mean_acc, std_acc = process_and_evaluate(cd8_syn, cd8_bld, pad=20)  # 20 is the max size for all seqs
    print(f"CD8 - KNN 3 neighbours: Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

    pass



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


#
# from esm.models.esmc import ESMC
# from esm.sdk.api import ESMProtein, LogitsConfig
#
# protein = ESMProtein(sequence="AAAAA")
# client = ESMC.from_pretrained("esmc_300m").to("cpu") # or "cuda"
# protein_tensor = client.encode(protein)
# logits_output = client.logits(
#    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
# )
# print(logits_output.logits, logits_output.embeddings)
#
