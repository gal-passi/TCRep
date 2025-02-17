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
STUDY_ID4 = 'PRJNA258001'
HEALTHY_STUDY_ID = STUDY_ID3
HEALTHY_STUDY_ID2 = STUDY_ID4
STUDIES = [STUDY_ID, STUDY_ID2, STUDY_ID3, STUDY_ID4]
VALID_SEQ_CACHE = "cache/valid_sequences"
TO_DISPLAY_LENGTHS_HIST = False


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
        # SAVING MEAN OF EMBEDDINGS! To save the full embedding (1, seq_len, 960) remove the mean function
        embeds.append(logits_output.embeddings.mean(dim=1).cpu())  # can also output: logits_output.logits
    return embeds


# TODO: Add this function in each fold, so that we can display more clearly the results (instead of using it on the full data)
def t_sne_display(X_syn, X_bld, study_name, cd_type, name_opt=''):
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    X_tsne = tsne.fit_transform(np.concatenate([X_syn, X_bld]))
    X_tsne_syn = X_tsne[:len(X_syn)]
    X_tsne_bld = X_tsne[len(X_syn):len(X_syn) + len(X_bld)]

    # plotting
    plt.figure(figsize=(12, 8))
    plt.scatter(X_tsne_bld[:, 0], X_tsne_bld[:, 1],
                c='red', label='Blood', alpha=0.4)
    plt.scatter(X_tsne_syn[:, 0], X_tsne_syn[:, 1],
                c='blue', label='Synovial Fluid', alpha=0.4)
    plt.legend()
    plt.title(f't-SNE Visualization of CD{cd_type} Data')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    # save plot
    plots_folder = f"plots/{study_name}"
    os.makedirs(plots_folder, exist_ok=True)
    plt.savefig(os.path.join(plots_folder, f"tsne_cd{cd_type}{name_opt}.png"))
    plt.show()


def process_and_evaluate(syn, healthy, bld, syn_mask, bld_mask, k_fold_type, study_name,
                         ratio=3, n_neighbors=9, to_plot=False, cd_type='4', name_opt=''):
    # Prepare data
    X_syn = torch.cat([x for x in syn])
    X_bld = torch.cat([x for x in bld])
    X_hlt = torch.cat([x for x in healthy])

    X = X_syn
    y = np.array([0] * len(syn))

    # Plotting if needed
    if to_plot:
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        X_tsne = tsne.fit_transform(np.concatenate([X_syn, X_bld]))
        # X_tsne = tsne.fit_transform(np.concatenate([X_syn, X_bld, X_hlt]))
        X_tsne_syn = X_tsne[:len(syn)]
        X_tsne_bld = X_tsne[len(syn):len(syn) + len(bld)]
        # X_tsne_hlt = X_tsne[len(syn) + len(bld):]

        # plotting
        plt.figure(figsize=(12, 8))
        # plt.scatter(X_tsne_hlt[:, 0], X_tsne_hlt[:, 1],
        #             c='green', label='Blood (healthy)', alpha=0.4)
        plt.scatter(X_tsne_bld[:, 0], X_tsne_bld[:, 1],
                    c='red', label='Blood', alpha=0.4)
        plt.scatter(X_tsne_syn[:, 0], X_tsne_syn[:, 1],
                    c='blue', label='Synovial Fluid', alpha=0.4)
        plt.legend()
        plt.title(f't-SNE Visualization of CD{cd_type} Data')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        # save plot
        plots_folder = f"plots/{study_name}"
        os.makedirs(plots_folder, exist_ok=True)
        plt.savefig(os.path.join(plots_folder, f"tsne_cd{cd_type}{name_opt}.png"))
        plt.show()

    # K-fold cross validation
    if k_fold_type == 0:  # random k-fold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds_indices = kf.split(X)
        blood_indices = [x[0] for x in kf.split(X_bld)]
    else:  # patient k-fold
        folds_indices = []
        for i in range(len(syn_mask)):
            train_dx = ((np.delete(syn_mask, i, axis=0) == 1).any(axis=0))
            test_dx = ~train_dx  # OR: test_dx = syn_mask[i] == 1
            folds_indices.append((train_dx, test_dx))
        blood_indices = [mask == 1 for mask in bld_mask]

    scores = []
    for i, (train_idx, test_idx) in enumerate(folds_indices):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Adding the same healthy data to the training set (for each fold)
        # X_hlt_rnd = X_hlt
        n = min(len(X_train) * ratio, len(X_hlt))
        X_hlt_rnd = X_hlt[np.random.choice(len(X_hlt), n, replace=False)]
        X_train = np.vstack([X_train, X_hlt_rnd])
        y_train = np.hstack([y_train, np.array([1] * len(X_hlt_rnd))])

        # Adding random samples from blood to the test set (of the same size as positive samples in the test set)
        # X_bld_rnd = X_bld[blood_indices[i]]
        X_bld_test = X_bld[blood_indices[i]]
        m = min(len(y_test) * ratio, len(X_bld_test))
        X_bld_rnd = X_bld_test[np.random.choice(len(X_bld_test), m, replace=False)]
        X_test = np.vstack([X_test, X_bld_rnd])
        y_test = np.hstack([y_test, np.array([1] * len(X_bld_rnd))])
        print(f"Ratio: Train {n/len(X[train_idx]):.2f}, Test {m/len(X[test_idx]):.2f}")

        # Train and evaluate KNN
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="minkowski")
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)

        # Displaying the t-SNE figure for each fold
        # t_sne_display(X[test_idx], X_bld_rnd, study_name, cd_type, name_opt)

        # Initialize variables
        # best_score = 0
        # best_n_neighbors = 0
        # n_neighbors_range = range(1, 11)  # Range of values to check for n_neighbors
        # for n_neighbors in n_neighbors_range:
        #     knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="minkowski")
        #     knn.fit(X_train, y_train)
        #     y_pred = knn.predict(X_test)
        #     score = accuracy_score(y_test, y_pred)
        #     scores.append(score)
        #
        #     if score > best_score:
        #         best_score = score
        #         best_n_neighbors = n_neighbors
        # scores.append(best_score)
        # print(f'Best number of neighbors: {best_n_neighbors} with a score of {best_score}')

        # TODO: Need to get the sequence identity matrix somehow!
        # # Compute sequence identities
        # if seq_id_matrix is not None:
        #     seq_id_test_train = seq_id_matrix[np.ix_(test_idx, train_idx)]
        # else:
        #     seq_id_test_train = np.array([
        #         [compute_sequence_identity(test_seq, train_seq) for train_seq in X_train]
        #         for test_seq in X_test
        #     ])
        #
        # # Get the highest sequence identity for each test sample
        # max_seq_id = seq_id_test_train.max(axis=1)
        #
        # # Define bins (0-10%, 10-20%, ..., 90-100%)
        # bins = np.linspace(0, 1, 11)
        # bin_indices = np.digitize(max_seq_id, bins) - 1
        #
        # # Calculate success and failure rate per bin
        # bin_success = np.zeros(len(bins) - 1)
        # bin_failure = np.zeros(len(bins) - 1)
        # bin_counts = np.zeros(len(bins) - 1)
        #
        # for j, bin_idx in enumerate(bin_indices):
        #     bin_counts[bin_idx] += 1
        #     if y_pred[j] == y_test[j]:
        #         bin_success[bin_idx] += 1
        #     else:
        #         bin_failure[bin_idx] += 1
        #
        # # Normalize to get success rates
        # bin_success_rate = bin_success / np.maximum(bin_counts, 1)
        # bin_failure_rate = bin_failure / np.maximum(bin_counts, 1)
        #
        # # Plot results
        # plt.figure(figsize=(10, 6))
        # plt.bar(bins[:-1] * 100, bin_success_rate, width=10, color='green', alpha=0.7, label="Success Rate")
        # plt.bar(bins[:-1] * 100, bin_failure_rate, width=10, bottom=bin_success_rate, color='red', alpha=0.7,
        #         label="Failure Rate")
        # plt.xlabel("Max Sequence Identity with Training Set (%)")
        # plt.ylabel("Prediction Rate")
        # plt.title(f"Success vs Failure Rate per Sequence Identity (Fold {i + 1})")
        # plt.legend()
        #
        # # Save plot
        # plots_folder = f"plots/{study_name}"
        # os.makedirs(plots_folder, exist_ok=True)
        # plt.savefig(os.path.join(plots_folder, f"seq_id_success_cd{cd_type}_fold{i + 1}{name_opt}.png"))
        # plt.show()

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


def plot_length_histogram(*arrays, labels=None, title_text="sequences"):
    """
    Plots histograms of string lengths for multiple ndarrays with bars side by side.

    Parameters:
        *arrays (numpy.ndarray): Multiple arrays of strings.
        labels (list, optional): List of labels corresponding to each array.
        title_text (str): Custom text to insert in the title.
    """
    if labels is None:
        labels = [f"Dataset {i + 1}" for i in range(len(arrays))]

    plt.figure(figsize=(12, 6))

    # Calculate the range for bins
    all_lengths = np.concatenate([np.vectorize(len)(arr) for arr in arrays])
    min_length = min(all_lengths)
    max_length = max(all_lengths)
    bins = range(min_length, max_length + 2)

    # Calculate bar width based on number of arrays
    width = 0.8 / len(arrays)

    # Plot histograms side by side
    for i, (arr, label) in enumerate(zip(arrays, labels)):
        lengths = np.vectorize(len)(arr)
        # Calculate histogram data manually
        counts, _ = np.histogram(lengths, bins=bins)
        # Calculate bar positions
        bar_positions = np.array(list(bins[:-1])) + (i * width)
        plt.bar(bar_positions, counts, width=width, alpha=0.8,
                label=label, edgecolor='black')

    plt.xlabel("String Length")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {title_text} lengths")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust x-axis to center the grouped bars
    plt.xticks(list(bins[:-1]), bins[:-1])

    plt.show()


def bound_and_sample_blood(syn, blood, ratio=2):
    lengths = set([len(x) for x in syn])
    min_len, max_len = min(lengths), max(lengths)
    blood = np.array(sorted(list(blood)))
    mask = np.vectorize(lambda s: min_len <= len(s) <= max_len)(blood)
    blood = blood[mask]
    n = min(len(syn) * ratio, len(blood))
    blood = np.random.choice(blood, n, replace=False)
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


# def plot_prediction_percentages(correct_percentages, incorrect_percentages, total_counts, bins, title):
#     bar_width = 0.35
#     x = np.arange(10)
#
#     plt.figure(figsize=(12, 7))
#     plt.bar(x - bar_width / 2, correct_percentages, width=bar_width, label='Correct (%)', color='green')
#     plt.bar(x + bar_width / 2, incorrect_percentages, width=bar_width, label='Incorrect (%)', color='red')
#
#     # Add text for the total number of samples in each bin
#     for i in range(10):
#         if total_counts[i] > 0:
#             plt.text(i, max(correct_percentages[i], incorrect_percentages[i]) + 2, f"n={total_counts[i]}", ha='center')
#
#     plt.xlabel('Max Sequence Identity Range')
#     plt.ylabel('Percentage (%)')
#     plt.title(title)
#     plt.xticks(x, [f"{bins[i] * 100:.1f}-{bins[i + 1] * 100:.1f}" for i in range(10)], rotation=30)
#     plt.ylim(0, 110)
#     plt.legend()
#     plt.tight_layout()
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.show()


def plot_prediction_percentages(correct_percentages, incorrect_percentages, total_counts, bins, title,
                                show_accuracy=True):
    """
    Plot the correct and incorrect prediction percentages per sequence identity bin.
    Optionally, display accuracy instead of percentages when `show_accuracy` is True.

    Parameters:
    - correct_percentages: Array of correct prediction percentages for each bin.
    - incorrect_percentages: Array of incorrect prediction percentages for each bin.
    - total_counts: Array of total sample counts per bin.
    - bins: The bin edges for sequence identity.
    - title: The title of the plot.
    - show_accuracy: Boolean flag to switch between showing percentages or accuracy in the plot.
    """
    # Calculate accuracy if show_accuracy is True
    if show_accuracy:
        # Calculate the accuracy for each bin: correct / total
        accuracies = np.divide(correct_percentages, 100, where=total_counts > 0)
    else:
        accuracies = None

    bar_width = 0.35
    x = np.arange(10)

    plt.figure(figsize=(12, 7))

    if show_accuracy:
        plt.bar(x, accuracies * 100, width=bar_width, label='Accuracy (%)', color='blue')
        plt.ylabel('Accuracy (%)')
    else:
        # Plot percentages (correct/incorrect)
        plt.bar(x - bar_width / 2, correct_percentages, width=bar_width, label='Correct (%)', color='green')
        plt.bar(x + bar_width / 2, incorrect_percentages, width=bar_width, label='Incorrect (%)', color='red')
        plt.ylabel('Percentage (%)')

    # Add text for the total number of samples in each bin
    for i in range(10):
        if total_counts[i] > 0:
            # Adjust text position: slightly above the bar
            y_position = max(correct_percentages[i], incorrect_percentages[i],
                             accuracies[i] if accuracies is not None else 0) + 2
            if show_accuracy:
                y_position = max(accuracies[i] * 100, 2) + 2  # Slightly above the accuracy bar
            plt.text(i, y_position, f"n={total_counts[i]}", ha='center')

    plt.xlabel('Max Sequence Identity Range')
    plt.title(title)
    plt.xticks(x, [f"{bins[i] * 100:.1f}-{bins[i + 1] * 100:.1f}" for i in range(10)], rotation=30)
    plt.ylim(0, 110)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def display_success_figure_per_patient(syn, hlt, bld, syn_mask, bld_mask, syn_seqs, k_fold_type, study_name,
                                       ratio=3, n_neighbors=9, cd_type='4', name_opt='', display_inner_figs=False, show_accuracy=True):
    X = torch.cat(syn)
    X_bld = torch.cat(bld)
    X_hlt = torch.cat(hlt)
    y = np.array([0] * len(syn))

    folds_indices = []
    for i in range(len(syn_mask)):
        train_dx = ((np.delete(syn_mask, i, axis=0) == 1).any(axis=0))
        test_dx = ~train_dx  # OR: test_dx = syn_mask[i] == 1
        folds_indices.append((train_dx, test_dx))
    blood_indices = [mask == 1 for mask in bld_mask]

    if show_accuracy:
        title_start = "Accuracy"
    else:
        title_start = "Prediction Percentage"

    # To accumulate the correct and incorrect counts for each bin across patients
    correct_counts_all_patients = []
    incorrect_counts_all_patients = []

    scores_per_patient = []
    for i, mask in enumerate(syn_mask):
        patient_idx = mask == 1
        X_patient = X[patient_idx]
        indices = np.arange(len(X_patient))
        np.random.shuffle(indices)
        split_idx = int(0.8 * len(X_patient))
        train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        X_train, X_test = X_patient[train_idx], X_patient[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Adding the same healthy data to the training set (for each fold)
        n = min(len(X_train) * ratio, len(X_hlt))
        X_hlt_rnd = X_hlt[np.random.choice(len(X_hlt), n, replace=False)]
        X_train = np.vstack([X_train, X_hlt_rnd])
        y_train = np.hstack([y_train, np.array([1] * len(X_hlt_rnd))])
        # print(f"Ratio: Train {n / len(X[train_idx]):.2f}")

        # # Adding random samples from blood to the test set (of the same size as positive samples in the test set)
        # X_bld_test = X_bld[blood_indices[i]]
        # m = min(len(y_test) * ratio, len(X_bld_test))
        # X_bld_rnd = X_bld_test[np.random.choice(len(X_bld_test), m, replace=False)]
        # X_test = np.vstack([X_test, X_bld_rnd])
        # y_test = np.hstack([y_test, np.array([1] * len(X_bld_rnd))])
        # # print(f"Ratio: Train {n / len(X[train_idx]):.2f}, Test {m / len(X[test_idx]):.2f}")

        # Getting patient train and test sequences (Of Synovial Fluid)
        syn_seqs_train = syn_seqs[patient_idx][train_idx]
        syn_seqs_test = syn_seqs[patient_idx][test_idx]

        # Compute pairwise identity matrix
        pwc_mat = pairwise_scores(syn_seqs_train, syn_seqs_test)  # OR: , score=levenshtein_dist

        # Step 1: Get max identity value for each test sample
        max_similarities = np.max(pwc_mat, axis=0)  # max similarity score for each test sample

        # Step 2: Create bins for the max similarity values (between lowest and highest value)
        min_sim = np.min(max_similarities)
        max_sim = np.max(max_similarities)
        bins = np.linspace(min_sim, max_sim, 11)  # 10 bins

        # Step 3: Assign each test sample to a bin based on its max similarity
        bin_indices = np.digitize(max_similarities, bins) - 1  # Subtract 1 to match bin index
        bin_indices[bin_indices == 10] = 9  # Fix edge case

        # Step 4: Train the KNN classifier
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="minkowski")
        knn.fit(X_train, y_train)  # Assuming X_train and y_train are available
        y_pred = knn.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores_per_patient.append(score)

        # Initialize counters for correct and incorrect samples in each bin
        correct_counts = np.zeros(10, dtype=int)
        incorrect_counts = np.zeros(10, dtype=int)

        # Loop over test samples
        for i in range(len(y_test)):
            bin_idx = bin_indices[i]
            if y_pred[i] == y_test[i]:
                correct_counts[bin_idx] += 1
            else:
                incorrect_counts[bin_idx] += 1

        # Calculate total samples and percentages
        total_counts = correct_counts + incorrect_counts
        correct_percentages = np.divide(correct_counts, total_counts, where=total_counts > 0) * 100
        incorrect_percentages = np.divide(incorrect_counts, total_counts, where=total_counts > 0) * 100

        # Store for aggregation later
        correct_counts_all_patients.append(correct_counts)
        incorrect_counts_all_patients.append(incorrect_counts)

        if display_inner_figs:
            plot_prediction_percentages(
                correct_percentages=correct_percentages,
                incorrect_percentages=incorrect_percentages,
                total_counts=total_counts,
                bins=bins,
                title=f'CD{cd_type} {title_start} Prediction Percentages per Sequence Identity Bin (n = Sample Count)',
                show_accuracy=show_accuracy
            )

        # title = f'CD{cd_type} Correct & Incorrect Prediction Percentages per Sequence Identity Bin (n = Sample Count)'
        # # Plotting
        # bar_width = 0.35
        # x = np.arange(10)
        #
        # plt.figure(figsize=(12, 7))
        # plt.bar(x - bar_width / 2, correct_percentages, width=bar_width, label='Correct (%)', color='green')
        # plt.bar(x + bar_width / 2, incorrect_percentages, width=bar_width, label='Incorrect (%)', color='red')
        #
        # # Add text for the total number of samples in each bin
        # for i in range(10):
        #     if total_counts[i] > 0:
        #         plt.text(i, max(correct_percentages[i], incorrect_percentages[i]) + 2, f"n={total_counts[i]}",
        #                  ha='center')
        #
        # plt.xlabel('Max Sequence Identity Range')
        # plt.ylabel('Percentage (%)')
        # plt.title(title)
        # plt.xticks(x, [f"{bins[i]*100:.1f}-{bins[i + 1]*100:.1f}" for i in range(10)], rotation=30)
        # plt.ylim(0, 110)
        # plt.legend()
        # plt.tight_layout()
        # plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.show()

    # Aggregate over patients
    correct_counts_all_patients = np.array(correct_counts_all_patients)
    incorrect_counts_all_patients = np.array(incorrect_counts_all_patients)

    # Sum counts across patients
    correct_counts_sum = np.sum(correct_counts_all_patients, axis=0)
    incorrect_counts_sum = np.sum(incorrect_counts_all_patients, axis=0)
    total_counts_sum = correct_counts_sum + incorrect_counts_sum

    # Mean percentage per bin
    correct_percentages_mean = np.divide(correct_counts_sum, total_counts_sum, where=total_counts_sum > 0) * 100
    incorrect_percentages_mean = np.divide(incorrect_counts_sum, total_counts_sum, where=total_counts_sum > 0) * 100

    plot_prediction_percentages(
        correct_percentages=correct_percentages_mean,
        incorrect_percentages=incorrect_percentages_mean,
        total_counts=total_counts_sum,
        bins=bins,
        title=f'CD{cd_type} Mean {title_start} Prediction Percentages per Sequence Identity Bin (n = Sample Count)',
        show_accuracy=show_accuracy
    )

    # title = f'CD{cd_type} Mean Correct & Incorrect Prediction Percentages per Sequence Identity Bin (n = Sample Count)'
    # # Plot final aggregated figure
    # bar_width = 0.35
    # x = np.arange(10)
    #
    # plt.figure(figsize=(12, 7))
    # plt.bar(x - bar_width / 2, correct_percentages_mean, width=bar_width, label='Correct (%)', color='green')
    # plt.bar(x + bar_width / 2, incorrect_percentages_mean, width=bar_width, label='Incorrect (%)', color='red')
    #
    # # Add text for total number of samples in each bin
    # for i in range(10):
    #     if total_counts_sum[i] > 0:
    #         plt.text(i, max(correct_percentages_mean[i], incorrect_percentages_mean[i]) + 2, f"n={total_counts_sum[i]}",
    #                  ha='center')
    #
    # plt.xlabel('Max Sequence Identity Range')
    # plt.ylabel('Percentage (%)')
    # plt.title(title)
    # plt.xticks(x, [f"{bins[i] * 100:.1f}-{bins[i + 1] * 100:.1f}" for i in range(10)], rotation=30)
    # plt.ylim(0, 110)
    # plt.legend()
    # plt.tight_layout()
    # plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()

    # print(f"Samples CD4 Synovial: {len(cd4_syn)}, CD4 Blood: {len(cd4_bld)}, CD4 Healthy: {len(cd4_h)}")
    mean_acc, std_acc = np.mean(scores_per_patient), np.std(scores_per_patient)
    print(f"CD{cd_type} - KNN {n_neighbors} neighbours: Accuracy: {mean_acc:.3f} ± {std_acc:.3f}. (Random split per patient!)")


if __name__ == '__main__':
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

    # Load study
    study = Study(STUDIES[study_ind])
    study_healthy = Study(HEALTHY_STUDY_ID2)
    # study_healthy = Study(HEALTHY_STUDY_ID)

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
    df_h_cd4 = df_h[df_h['cell_type'] == 'CD4']
    valid_seqs_cd4_h = df_h_cd4["AASeq"].unique()

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

    # Display histogram of lengths:
    if TO_DISPLAY_LENGTHS_HIST:
        print("Displaying Histograms of Lengths")
        plot_length_histogram(df_cd4_bld["AASeq"], df_cd4_syn["AASeq"],
                              labels=["Blood", "Synovial"],
                              title_text="CD4 Sequences")
        plot_length_histogram(df_cd8_bld["AASeq"], df_cd8_syn["AASeq"],
                              labels=["Blood", "Synovial"],
                              title_text="CD8 Sequences")

        plot_length_histogram(np.array(list(sr_cd4_bld_vld)), np.array(sr_cd4_syn_vld),
                              labels=["Blood", "Synovial"],
                              title_text="CD4 Filtered Sequences")
        plot_length_histogram(np.array(list(sr_cd8_bld_vld)), np.array(sr_cd8_syn_vld),
                              labels=["Blood", "Synovial"],
                              title_text="CD8 Filtered Sequences")

    # bounding the length of sequences to be min and max of synovial samples,
    # then sampling *ratio samples from blood to match *ratio the number of synovial samples
    np.random.seed(42)
    ratio = 10
    sr_cd4_bld_vld = bound_and_sample_blood(sr_cd4_syn_vld, sr_cd4_bld_vld, ratio)
    sr_cd8_bld_vld = bound_and_sample_blood(sr_cd8_syn_vld, sr_cd8_bld_vld, ratio)
    valid_seqs_cd4_h = bound_and_sample_blood(sr_cd4_syn_vld, valid_seqs_cd4_h, ratio)
    valid_seqs_cd8_h = bound_and_sample_blood(sr_cd8_syn_vld, valid_seqs_cd8_h, ratio)
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

    cd4_h = get_cached_embeddings(list(valid_seqs_cd4_h), study_healthy.name, name='cd4_h' + name_opt, embed_fn=embed)
    cd8_h = get_cached_embeddings(list(valid_seqs_cd8_h), study_healthy.name, name='cd8_h' + name_opt, embed_fn=embed)

    # Evaluating CD4 and CD8
    n_neighbors = 9
    neg_to_pos_ratio = 3
    to_plot = False

    display_success_figure_per_patient(cd4_syn, cd4_h, cd4_bld, cd4_syn_patient_id_masks, cd4_bld_patient_id_masks,
                                       sr_cd4_syn_vld,
                                       k_fold_type, study.name, ratio=neg_to_pos_ratio, n_neighbors=n_neighbors,
                                       cd_type='4', name_opt=name_opt)
    display_success_figure_per_patient(cd8_syn, cd8_h, cd8_bld, cd8_syn_patient_id_masks, cd8_bld_patient_id_masks,
                                       sr_cd8_syn_vld,
                                       k_fold_type, study.name, ratio=neg_to_pos_ratio, n_neighbors=n_neighbors,
                                       cd_type='8', name_opt=name_opt)
    exit(0)

    print(f"Samples CD4 Synovial: {len(cd4_syn)}, CD4 Blood: {len(cd4_bld)}, CD4 Healthy: {len(cd4_h)}")
    mean_acc, std_acc = process_and_evaluate(cd4_syn, cd4_h, cd4_bld,
                                             cd4_syn_patient_id_masks, cd4_bld_patient_id_masks, k_fold_type,
                                             ratio=neg_to_pos_ratio, n_neighbors=n_neighbors, to_plot=to_plot,
                                             study_name=study.name, cd_type="4", name_opt=name_opt)  # 20 is the max size for all seqs
    print(f"CD4 - KNN {n_neighbors} neighbours: Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    print()
    print(f"Samples CD8 Synovial: {len(cd8_syn)}, CD8 Blood: {len(cd8_bld)}, CD8 Healthy: {len(cd8_h)}")
    mean_acc, std_acc = process_and_evaluate(cd8_syn, cd8_h, cd8_bld,
                                             cd8_syn_patient_id_masks, cd8_bld_patient_id_masks, k_fold_type,
                                             ratio=neg_to_pos_ratio, n_neighbors=n_neighbors, to_plot=to_plot,
                                             study_name=study.name, cd_type="8", name_opt=name_opt)  # 20 is the max size for all seqs
    print(f"CD8 - KNN {n_neighbors} neighbours: Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
