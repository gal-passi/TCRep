import warnings
from gc import freeze

from triton.language.semantic import device_print
warnings.simplefilter("ignore", category=FutureWarning)
from Curation import Study
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import math
from tqdm import tqdm
from utils import pairwise_scores, levenshtein_dist, levenshtein_dist_non_bin
import pickle
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score
from itertools import combinations, chain
import multiprocessing as mp
from collections import Counter
from embedding.embedding import get_cached_embeddings
from embedding.esmc_finetuning import load_fine_tuned_esmc, fine_tune_esmc
from models.cvc_model import CVCClassifierModel
from models.ff_model import FeedForwardClassifier
from models.esmc_ff_model import ESMCFeedForwardClassifier
from torch.utils.data import Dataset, DataLoader
import time
import random
import seaborn as sns
from model_trainer import train_model, display_training_results
import wandb


STUDY_ID = 'PRJNA393498'  # Ankylosing Spondylitis study
STUDY_ID2 = 'immunoSEQ47'  # Hepatitis B virus study
STUDY_ID3 = 'immunoSEQ77'  # Rheumatoid arthritis study (plus healthy)
STUDY_ID4 = 'PRJNA258001'  # HIV study (plus healthy)
STUDY_ID5 = 'PRJNA390125'  # Only healthy study
STUDY_ID6 = 'PRJNA495603'  # Multiple sclerosis study (plus healthy)
STUDY_ID7 = 'PRJNA579190'  #  Multiple sclerosis study (plus healthy)
STUDY_ID8 = 'PRJNA280417'  #  Multiple sclerosis study
HEALTHY_STUDY_ID = STUDY_ID3  # ONLY CD8
HEALTHY_STUDY_ID2 = STUDY_ID4  # Both CD8 and CD4
HEALTHY_STUDY_ID3 = STUDY_ID5  # Larger both CD8 and CD4 (But fewer patients!)
HEALTHY_STUDY_ID4 = STUDY_ID6  # Other healthy study
HEALTHY_STUDY_ID5 = STUDY_ID7  # Other healthy study
STUDIES = [STUDY_ID, STUDY_ID2, STUDY_ID3, STUDY_ID4, STUDY_ID5, STUDY_ID6, STUDY_ID7]
VALID_SEQ_CACHE = "cache/valid_sequences"
TO_DISPLAY_LENGTHS_HIST = False
TO_DISPLAY_COMMON_SEQUENCES = False
TO_DISPLAY_ACCURACY_BIN_BY_DIST = False
TO_DISPLAY_RESULTS = False
TO_DISPLAY_RESULTS_PLOT_TSNE = False
TO_DISPLAY_NUMBER_OF_COMMON_SEQUENCES = False
TO_LOAD_FULL_SYNAPSE_DATA = False
TO_RETRAIN_CLASSIFIER_MODEL = True


def get_all_usable_healthy_data():
    healthy_study_ids = [HEALTHY_STUDY_ID, HEALTHY_STUDY_ID2, HEALTHY_STUDY_ID3, HEALTHY_STUDY_ID4, HEALTHY_STUDY_ID5]
    healthy_studies = []
    for study_id in healthy_study_ids:
        study = Study(study_id)
        usable_samples = study._samples['usable']
        df = study.read_sample(usable_samples)
        df = df[df['condition'] == 'Healthy']
        df['study_id'] = study_id
        healthy_studies.append(df)
    df_concat = pd.concat(healthy_studies, ignore_index=True)
    df_concat = df_concat.dropna(subset=['AASeq'])
    return df_concat


def get_all_usable_disease_data(disease='Multiple sclerosis'):
    studies = []
    if disease == 'Ankylosing spondylitis':
        study_ids = [STUDY_ID]
    elif disease == 'Hepatitis B virus':
        study_ids = [STUDY_ID2]
    elif disease == 'Rheumatoid arthritis':
        study_ids = [STUDY_ID3]
    elif disease == 'HIV':
        study_ids = [STUDY_ID4]
    elif disease == 'Multiple sclerosis':
        study_ids = [STUDY_ID6, STUDY_ID7, STUDY_ID8]
    else:
        raise ValueError(f"Invalid disease: {disease}")

    for study_id in study_ids:
        study = Study(study_id)
        usable_samples = study._samples['usable']
        df = study.read_sample(usable_samples)
        df = df[df['condition'] == disease]
        df['study_id'] = study_id
        studies.append(df)

    return pd.concat(studies, ignore_index=True)


def t_sne_display(X_bld, X_hlt, study_name, cell_type, name_opt=''):
    tsne = TSNE(n_components=2, perplexity=5, random_state=42)
    X_tsne = tsne.fit_transform(np.concatenate([X_bld, X_hlt]))
    X_tsne_syn = X_tsne[:len(X_bld)]
    X_tsne_bld = X_tsne[len(X_bld):len(X_bld) + len(X_hlt)]

    # plotting
    plt.figure(figsize=(12, 8))
    plt.scatter(X_tsne_bld[:, 0], X_tsne_bld[:, 1],
                c='red', label='Blood', alpha=0.4)
    plt.scatter(X_tsne_syn[:, 0], X_tsne_syn[:, 1],
                c='blue', label='Synovial Fluid', alpha=0.4)
    plt.legend()
    plt.title(f't-SNE Visualization of cell type {cell_type} Data')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    # save plot
    plots_folder = f"plots/{study_name}"
    os.makedirs(plots_folder, exist_ok=True)
    plt.savefig(os.path.join(plots_folder, f"tsne_{cell_type}{name_opt}.png"))
    plt.show()


# TODO: Think about the correct way to set split the train and test...
def get_fold_indices_by_patient(syn_mask):
    folds_indices = []
    for i in range(len(syn_mask)):
        train_dx = (syn_mask[i] == 0)
        # train_dx = ((np.delete(syn_mask, i, axis=0) == 1).any(axis=0))
        test_dx = ~train_dx  # OR: test_dx = syn_mask[i] == 1
        folds_indices.append((train_dx, test_dx))
    return folds_indices


def process_and_evaluate(bld, healthy, bld_mask, k_fold_type, study_name,
                         cell_type, ratio=3, n_neighbors=9, name_opt=''):
    # Prepare data
    X_bld = torch.stack(bld)
    X_hlt = torch.stack(healthy)

    X = X_bld
    y = np.array([0] * len(bld))

    # Plotting if needed
    if TO_DISPLAY_RESULTS_PLOT_TSNE:
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        X_tsne = tsne.fit_transform(np.concatenate([X_bld]))
        X_tsne_syn = X_tsne[:len(syn)]
        X_tsne_bld = X_tsne[len(syn):len(syn) + len(bld)]

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

    # K-fold cross validation
    if k_fold_type == 0:  # random k-fold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        folds_indices = kf.split(X)
    else:  # patient k-fold
        folds_indices = get_fold_indices_by_patient(bld_mask)

    scores = []
    for i, (train_idx, test_idx) in enumerate(folds_indices):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Adding the same healthy data to the training set (for each fold)
        # X_hlt_rnd = X_hlt
        n = min(len(X_train) * ratio, len(X_hlt))
        hlt_indices = np.random.choice(len(X_hlt), n, replace=False)
        X_hlt_rnd = X_hlt[hlt_indices]
        X_train = np.vstack([X_train, X_hlt_rnd])
        y_train = np.hstack([y_train, np.array([1] * len(X_hlt_rnd))])
        # print(f"Ratio: Train {n/len(X[train_idx]):.2f}, Test {m/len(X[test_idx]):.2f}")

        # Adding random samples from blood to the test set (of the same size as positive samples in the test set)
        X_hlt_rnd_test = X_hlt[~hlt_indices]
        m = min(len(y_test) * ratio, len(X_hlt_rnd_test))
        X_hlt_rnd_test = X_hlt_rnd_test[np.random.choice(len(X_hlt_rnd_test), m, replace=False)]
        X_test = np.vstack([X_test, X_hlt_rnd_test])
        y_test = np.hstack([y_test, np.array([1] * len(X_hlt_rnd_test))])

        # Train and evaluate KNN
        knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="minkowski")
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        scores.append(score)

        # Displaying the t-SNE figure for each fold
        if TO_DISPLAY_RESULTS_PLOT_TSNE:
            t_sne_display(X[test_idx], X_hlt_rnd_test, study_name, cell_type, name_opt)

    return np.mean(scores), np.std(scores)


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



def plot_prediction_percentages(correct_percentages, incorrect_percentages, total_counts, bins, title, plot_file,
                                num_of_bins, show_accuracy=True):
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
        accuracies[accuracies > 1] = 0  # Fix edge case
        accuracies[accuracies < 0] = 0  # Fix edge case
    else:
        accuracies = None

    bar_width = 0.35
    x = np.arange(num_of_bins)

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
    for i in range(num_of_bins):
        if total_counts[i] > 0:
            # Adjust text position: slightly above the bar
            y_position = max(correct_percentages[i], incorrect_percentages[i],
                             accuracies[i] if accuracies is not None else 0) + 2
            if show_accuracy:
                y_position = max(accuracies[i] * 100, 2) + 2  # Slightly above the accuracy bar
            plt.text(i, y_position, f"n={total_counts[i]}", ha='center')

    plt.xlabel('Max Sequence Identity Range')
    plt.title(title)
    plt.xticks(x, [f"{bins[i]}" for i in range(num_of_bins)])
    # plt.xticks(x, [f"{bins[i] * 100:.1f}-{bins[i + 1] * 100:.1f}" for i in range(num_of_bins)], rotation=30)
    plt.ylim(0, 110)
    plt.legend()
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plots_folder = f"plots/"
    os.makedirs(plots_folder, exist_ok=True)
    plt.savefig(os.path.join(plots_folder, f"bin_accuracy_by_dist_{plot_file}.png"))
    plt.show()


def display_accuracy_bin_by_dist_figure(syn, hlt, bld, syn_mask, bld_mask, syn_seqs, bld_seqs, hlt_seqs, k_fold_type, study_name,
                                       ratio=3, n_neighbors=9, cd_type='4', name_opt='', display_inner_figs=True, show_accuracy=True,
                                       num_of_bins=15):
    X = torch.cat(syn)
    X_bld = torch.cat(bld)
    X_hlt = torch.cat(hlt)
    y = np.array([0] * len(syn))

    folds_indices = get_fold_indices_by_patient(syn_mask)
    blood_indices = [mask == 1 for mask in bld_mask]

    if show_accuracy:
        title_start = "Accuracy"
    else:
        title_start = "Prediction Percentage"

    # To accumulate the correct and incorrect counts for each bin across patients
    correct_counts_all_patients = []
    incorrect_counts_all_patients = []

    bins = np.linspace(0, num_of_bins, num_of_bins+1)
    scores_per_patient = []
    # for i, mask in enumerate(syn_mask):
    for i, (train_idx, test_idx) in enumerate(folds_indices):
        # patient_idx = mask == 1
        # X_patient = X[patient_idx]
        # indices = np.arange(len(X_patient))
        # np.random.shuffle(indices)
        # split_idx = int(0.8 * len(X_patient))
        # train_idx, test_idx = indices[:split_idx], indices[split_idx:]
        # X_train, X_test = X_patient[train_idx], X_patient[test_idx]
        # y_train, y_test = y[train_idx], y[test_idx]
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Adding the same healthy data to the training set (for each fold)
        n = min(len(X_train) * ratio, len(X_hlt))
        hlt_idx = np.random.choice(len(X_hlt), n, replace=False)
        X_hlt_rnd = X_hlt[hlt_idx]
        X_train = np.vstack([X_train, X_hlt_rnd])
        y_train = np.hstack([y_train, np.array([1] * len(X_hlt_rnd))])
        # print(f"Ratio: Train {n / len(X[train_idx]):.2f}")

        # Adding random samples from blood to the test set (of the same size as positive samples in the test set)
        X_bld_test = X_bld[blood_indices[i]]
        m = min(len(y_test) * ratio, len(X_bld_test))
        bld_idx = np.random.choice(len(X_bld_test), m, replace=False)
        X_bld_rnd = X_bld_test[bld_idx]
        X_test = np.vstack([X_test, X_bld_rnd])
        y_test = np.hstack([y_test, np.array([1] * len(X_bld_rnd))])
        # print(f"Ratio: Train {n / len(X[train_idx]):.2f}, Test {m / len(X[test_idx]):.2f}")

        # Getting patient train and test sequences (Of Synovial Fluid)
        syn_seqs_train = syn_seqs[train_idx]
        syn_seqs_test = syn_seqs[test_idx]
        bld_seqs_train = hlt_seqs[hlt_idx]
        bld_seqs_test = bld_seqs[bld_idx]
        # syn_seqs_train = syn_seqs[patient_idx][train_idx]
        # syn_seqs_test = syn_seqs[patient_idx][test_idx]

        def calculate_correct_incorrect_bin(seqs_train, seqs_test, r):
            # Compute pairwise identity matrix
            pwc_mat = pairwise_scores(seqs_train, seqs_test, score=levenshtein_dist_non_bin)

            # Step 1: Get max\min identity value for each test sample
            max_similarities = np.min(pwc_mat, axis=0)  # max similarity score for each test sample
            # max_similarities = np.max(pwc_mat, axis=0)  # max similarity score for each test sample

            # Step 2: Create bins for the max similarity values (between lowest and highest value)
            min_sim = np.min(max_similarities)
            max_sim = np.max(max_similarities)
            print(f"L-dist Min and Max: {min_sim}, {max_sim}")

            # Step 3: Assign each test sample to a bin based on its max similarity
            bin_indices = np.digitize(max_similarities, bins) - 1  # Subtract 1 to match bin index
            bin_indices[bin_indices == num_of_bins] = num_of_bins - 1  # Fix edge case

            # Step 4: Train the KNN classifier
            knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric="minkowski")
            knn.fit(X_train, y_train)  # Assuming X_train and y_train are available
            y_pred = knn.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            scores_per_patient.append(score)

            # Initialize counters for correct and incorrect samples in each bin
            correct_counts = np.zeros(num_of_bins, dtype=int)
            incorrect_counts = np.zeros(num_of_bins, dtype=int)

            start_idx = next(iter(r))
            # Loop over test samples
            for i in r:
                bin_idx = bin_indices[i - start_idx]
                if y_pred[i] == y_test[i]:
                    correct_counts[bin_idx] += 1
                else:
                    incorrect_counts[bin_idx] += 1
            return correct_counts, incorrect_counts

        test_len = len(y[test_idx])
        print("Calculating correct/incorrect of SYN...")
        cc1, ci1 = calculate_correct_incorrect_bin(syn_seqs_train, syn_seqs_test, range(test_len))
        print("Calculating correct/incorrect of BLD...")
        cc2, ci2 = calculate_correct_incorrect_bin(bld_seqs_train, bld_seqs_test, range(test_len, test_len + m))
        print("Done.")

        # correct_counts, incorrect_counts = cc1, ci1
        correct_counts, incorrect_counts = cc1 + cc2, ci1 + ci2

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
                plot_file=f'CD{cd_type}_{title_start}_{i}',
                show_accuracy=show_accuracy,
                num_of_bins=num_of_bins
            )

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
        plot_file=f'CD{cd_type}_{title_start}',
        show_accuracy=show_accuracy,
        num_of_bins=num_of_bins
    )

    # print(f"Samples CD4 Synovial: {len(cd4_syn)}, CD4 Blood: {len(cd4_bld)}, CD4 Healthy: {len(cd4_h)}")
    mean_acc, std_acc = np.mean(scores_per_patient), np.std(scores_per_patient)
    print(f"CD{cd_type} - KNN {n_neighbors} neighbours: Accuracy: {mean_acc:.3f} ± {std_acc:.3f}. (Random split per patient!)")


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


def common_aaseq_analysis(df, num_of_patients, lev_dist_accept=0, mode=1):
    # Select the unique patients
    unique_patients = df['patient_id'].unique()

    if len(unique_patients) < num_of_patients:
        raise ValueError("Number of patients in the dataframe is less than num_of_patients")

    # Create a dictionary mapping each patient_id to their set of AASeq
    patient_sequences = {pid: set(df[df['patient_id'] == pid]['AASeq']) for pid in unique_patients}

    results = []
    if mode == 1:
        # Mode 1: All combinations
        patient_combinations = combinations(unique_patients, num_of_patients)
    elif mode == 2:
        # Mode 2: Always include the first patient
        patient_combinations = [tuple([unique_patients[0]] + list(comb)) for comb in combinations(unique_patients[1:], num_of_patients - 1)]
    else:
        raise ValueError("Mode must be 1 (All combinations) or 2 (Always include the first patient).")

    percent_of_total_values = []
    for combination in patient_combinations:
        selected_sequences = [patient_sequences[pid] for pid in combination]

        if lev_dist_accept >= 1:
            temp_df = df[df['patient_id'].isin(combination)]
            masks = helper_function_common_aaseq_analysis(temp_df, lev_dist_accept)
            num_common = np.sum(np.any(masks == 1, axis=0))  # TODO: This will always increase when we look at more patients... this isnt the calculation that we want here
        else:
            common_sequences = set.intersection(*selected_sequences)
            num_common = len(common_sequences)
        total_sequences = sum(len(seqs) for seqs in selected_sequences)
        min_sequences = min(len(seqs) for seqs in selected_sequences)
        max_sequences = max(len(seqs) for seqs in selected_sequences)
        percent_of_total = (num_common / total_sequences) * 100 if total_sequences > 0 else 0
        percent_of_min = (num_common / min_sequences) * 100 if min_sequences > 0 else 0
        percent_of_max = (num_common / max_sequences) * 100 if max_sequences > 0 else 0
        percent_of_total_values.append(percent_of_total)
        results.append({
            'num_total_seqs': total_sequences,
            'num_common': num_common,
            'percent_of_total': percent_of_total,
            'percent_of_min': percent_of_min,
            'percent_of_max': percent_of_max
        })

    # Calculate means
    mean_results = {
        'num_total_seqs': sum(r['num_total_seqs'] for r in results) / len(results),
        'num_common': sum(r['num_common'] for r in results) / len(results),
        'percent_of_total': sum(r['percent_of_total'] for r in results) / len(results),
        'percent_of_min': sum(r['percent_of_min'] for r in results) / len(results),
        'percent_of_max': sum(r['percent_of_max'] for r in results) / len(results)
    }

    # Calculate std for percent_of_total
    std_percent_of_total = np.std(percent_of_total_values)  # Calculate std for percent_of_total

    return mean_results, std_percent_of_total


def generate_patient_samples(df1, all_seqs_h: np.ndarray, patient_seqs_len: int) -> pd.DataFrame:
    """
    Generate a DataFrame where 'patient_id' ranges from H1 to H10, and 'AASeq' contains
    randomly sampled sequences from all_seqs_h for each patient.

    Parameters:
    - all_seqs_h (np.ndarray): Unique sequences.
    - patient_seqs_len (int): Number of sequences to sample per patient.

    Returns:
    - pd.DataFrame: DataFrame with 'patient_id' and 'AASeq' columns.
    """
    data = []
    for i in range(1, 15):  # Generate 10 samples
        sampled_seqs = np.random.choice(all_seqs_h, patient_seqs_len, replace=False)
        for seq in sampled_seqs:
            data.append((f"H{i}", seq))

    dfh = pd.DataFrame(data, columns=["patient_id", "AASeq"])
    return pd.concat([df1, dfh], ignore_index=True)


# This function can be applied on x_healthy_list_all in the function below to get average values across all differently picked patients
from collections import defaultdict
def average_dicts(outer_list):
    # The number of inner lists
    num_inner_lists = len(outer_list)

    # Initialize a list to store the averaged dictionaries
    averaged_list = []

    # Iterate through each inner list
    for i in range(len(outer_list[0])):  # assuming all inner lists have the same length
        # Initialize a defaultdict to accumulate values for each key
        accumulator = defaultdict(int)

        # Iterate through the outer list and accumulate the sum for each key in each dict
        for inner_list in outer_list:
            accumulator_dict = inner_list[i][0]  # Get the dict at index i
            for key, value in accumulator_dict.items():
                accumulator[key] += value

        # Now average the values by dividing by the number of inner lists
        averaged_dict = {key: value / num_inner_lists for key, value in accumulator.items()}

        # Add the averaged dict to the result list
        averaged_list.append(averaged_dict)

    return averaged_list


def display_common_sequences_figure(df, df_h, l=8, log_space=True):
    # find max len of uniques patient_id
    if l == None:
        l = min(1 + len(df_h['patient_id'].unique()), len(df['patient_id'].unique())) + 1

    study_groups = df.groupby('study_id')['patient_id'].unique().apply(list)
    rand_patients = [np.random.choice(x, size=5, replace=False) for x in study_groups]
    rand_patients = list(chain(*rand_patients))
    df = df[df['patient_id'].isin(rand_patients)]

    # calculate common sequences in disease and healthy samples
    value_to_take = "percent_of_total"  # "percent_of_total" or "num_common"
    x_disease_list = [common_aaseq_analysis(df, num_of_patients=i, mode=1) for i in range(2, l)]
    x_disease = np.array([x[0][value_to_take] for x in x_disease_list])
    x_disease_std = np.array([x[1] for x in x_disease_list])

    def calculate_common_healthy(patient_id_bld, option=1):
        df1 = df[df["patient_id"] == patient_id_bld]
        if option == 1:
            # First Option: Adding all healthy samples to the df as is (samples stays the same for each patient)
            random_patients = np.random.choice(df_h['patient_id'].unique(), size=15, replace=False)
            df_h_temp = df_h[df_h['patient_id'].isin(random_patients)]
            df_h_comb = pd.concat([df1, df_h_temp], ignore_index=True)
        else:
            # Second Option: Adding random samples from healthy to the df (of the same length as the patient with disease samples)
            patient_seqs_len = len(df1)
            all_seqs_h = df_h["AASeq"]
            df_h_comb = generate_patient_samples(df1, all_seqs_h, patient_seqs_len)
        x_healthy = [common_aaseq_analysis(df_h_comb, num_of_patients=i, mode=2) for i in range(2, l)]
        return x_healthy

    # Average the results of all patients with disease
    x_healthy_list_all = [calculate_common_healthy(patient_id_bld) for patient_id_bld in df['patient_id'].unique()]
    x_healthy_list = [[y[0][value_to_take] for y in x] for x in x_healthy_list_all]
    x_healthy_list_std = [[y[1] for y in x] for x in x_healthy_list_all]
    x_healthy = np.array(x_healthy_list).mean(axis=0)
    x_healthy_std = np.array(x_healthy_list_std).mean(axis=0)

    if log_space:
        x_disease = np.log(x_disease)
        x_healthy = np.log(x_healthy)

    # plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(range(2, len(x_disease) + 2), x_disease, label="Disease")
    plt.plot(range(2, len(x_healthy) + 2), x_healthy, label="Healthy")
    plt.fill_between(range(2, len(x_disease) + 2), x_disease - x_disease_std, x_disease + x_disease_std, alpha=0.2)
    plt.fill_between(range(2, len(x_healthy) + 2), x_healthy - x_healthy_std, x_healthy + x_healthy_std, alpha=0.2)
    # add the percentage of common sequences in the plot
    for i, txt in enumerate(x_disease):
        plt.annotate(f"{txt:.5f}", (i + 2, x_disease[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    for i, txt in enumerate(x_healthy):
        plt.annotate(f"{txt:.5f}", (i + 2, x_healthy[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.xlabel("Number of Patients")
    plt.ylabel("Percentage of Common Sequences (Only CD8)")
    plt.title("Percentage of Common Sequences in Disease and Healthy Samples" + (" (Log Scale)" if log_space else ""))
    plt.ylim(min(min(x_disease), min(x_healthy)), max(max(x_disease + x_disease_std), max(x_healthy + x_healthy_std)) * 1.)
    plt.legend()
    plt.show()


def display_common_sequences_figure_healthy(df_h, l=8, log_space=True):
    study_groups = df_h.groupby('study_id')['patient_id'].unique().apply(list)
    rand_patients = [np.random.choice(x, size=min(5, len(x)), replace=False) for x in study_groups]
    rand_patients = list(chain(*rand_patients))
    df_h = df_h[df_h['patient_id'].isin(rand_patients)]

    # calculate common sequences in healthy samples
    value_to_take = "percent_of_total"  # "percent_of_total" or "num_common"
    x_healthy_list = [common_aaseq_analysis(df_h, num_of_patients=i, mode=1) for i in range(2, l)]
    x_healthy = np.array([x[0][value_to_take] for x in x_healthy_list])
    x_healthy_std = np.array([x[1] for x in x_healthy_list])

    if log_space:
        x_healthy = np.log(x_healthy)

    # plot the results
    plt.figure(figsize=(10, 6))
    # the next color of the default colors of plot
    plt.plot(range(2, len(x_healthy) + 2), x_healthy, label="Healthy")
    plt.fill_between(range(2, len(x_healthy) + 2), x_healthy - x_healthy_std, x_healthy + x_healthy_std, alpha=0.2)
    # add the percentage of common sequences in the plot
    for i, txt in enumerate(x_healthy):
        plt.annotate(f"{txt:.5f}", (i + 2, x_healthy[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.xlabel("Number of Patients")
    plt.ylabel("Percentage of Common Sequences (Only CD8)")
    plt.title("Percentage of Common Sequences in Healthy Samples" + (" (Log Scale)" if log_space else ""))
    # plt.ylim(min(x_healthy), max(x_healthy + x_healthy_std))
    plt.legend()
    plt.show()


# TODO: Remove the variable healthy_unique from the function signature
# TODO: This function loads the synapse dataframe of Mal-ID of only TCR and healthy samples for sure.
def get_full_healthy_synapse_mal_id_dataframe(healthy_unique, get_all=False):
    # Defining constants
    synapse_db_folder = "db/synapse_Mal_ID"
    synapse_metadata_file = os.path.join(synapse_db_folder, "metadata.tsv")

    # Define the filename based on the get_all flag
    file_suffix = "_all" if get_all else "_healthy_only"
    df_filename = os.path.join(synapse_db_folder, f"synapse_mal_id_dataframe{file_suffix}.pkl")
    # Check if the DataFrame is already saved
    if os.path.exists(df_filename):
        # Load the DataFrame from file
        df = pd.read_pickle(df_filename)
        return df

    # Reading metadata
    synapse_metadata = pd.read_csv(synapse_metadata_file, sep='\t')

    # Reading and interpreting data files
    synapse_datafiles = [x for x in os.listdir(synapse_db_folder) if x.endswith(".bz2")]

    healthy_samples = []
    for datafile in synapse_datafiles:
        datafile_id = datafile.split("_")[-1][:-4]
        datafile_path = os.path.join(synapse_db_folder, datafile)

        # Reading metadata and data
        metadata = synapse_metadata[synapse_metadata['participant_label'] == datafile_id]
        condition = metadata['disease'].values[0]
        if get_all or 'Healthy' in condition:
            data = pd.read_csv(datafile_path, sep='\t', compression='bz2')
            data = data.loc[:, ['cdr3_seq_aa_q', 'participant_label', 'specimen_tissue']]
            data['cdr3_seq_aa_q'] = data['cdr3_seq_aa_q'].str.replace(' ', '')
            healthy_samples.append(data)

    # need to make a df with: AASeq, patient_id, tissue, cell_type
    df = pd.concat(healthy_samples, ignore_index=True)
    df = df.dropna(subset=['cdr3_seq_aa_q'])
    # rename columns
    df = df.rename(columns={'cdr3_seq_aa_q': 'AASeq', 'participant_label': 'patient_id', 'specimen_tissue': 'tissue'})

    # Save the DataFrame for future use
    df.to_pickle(df_filename)

    return df


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


def process_in_batches(df, all_common_seqs, batch_size, lev_dist_accept):
    all_common_seqs = list(all_common_seqs)
    # Split all_common_seqs into batches
    all_common_seqs_batches = [all_common_seqs[i:i + batch_size] for i in range(0, len(all_common_seqs), batch_size)]

    valid_seqs_set = set()  # Use a set to store valid sequences across batches

    for batch in tqdm(all_common_seqs_batches):
        df_new = df.copy()

        # Flag 'valid' for sequences in the current batch
        df_new['patient_id'] = df_new['AASeq'].apply(lambda x: 'valid' if x in batch else 'all')

        # Separate valid and all sequences
        valid_df = df_new[df_new['patient_id'] == 'valid'].drop_duplicates(subset='AASeq')
        all_df = df_new[df_new['patient_id'] == 'all'].drop_duplicates(subset='AASeq')

        # Concatenate and reset index
        df_combined = pd.concat([valid_df, all_df]).reset_index(drop=True)

        # Apply the analysis function to the valid sequences
        valid_seqs_batch = helper_function_common_aaseq_analysis(df_combined, lev_dist_accept, only_valid=True, verbose=False)

        # Accumulate valid sequences from this batch (as a set)
        valid_seqs_set.update(valid_seqs_batch)

    # The result is a set of valid sequences
    return valid_seqs_set


def calculate_valid_near_sequences(df, save_name, lev_dist_accept=1, num_of_patients=3, all_common_seqs=None):
    save_folder = "cache/valid_sequences/multiple_sclerosis"
    save_file = os.path.join(save_folder, f"{save_name}_valid_seqs_dist_{lev_dist_accept}.pkl")
    if not os.path.exists(save_file):
        if all_common_seqs is None:
            all_common_seqs = find_all_common_sequences(df, num_of_patients=num_of_patients)
        valid_seqs = process_in_batches(df, all_common_seqs, 512, lev_dist_accept)
        os.makedirs(save_folder, exist_ok=True)
        with open(save_file, "wb") as f:
            pickle.dump(valid_seqs, f)
    else:
        with open(save_file, "rb") as f:
            valid_seqs = pickle.load(f)
    return valid_seqs


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


def plot_output_distribution(model, set_name, model_type, pos_seqs, neg_seqs=None, device='cuda'):
    model.to(device)
    model.eval()

    # Convert sequences to tensors and move to device
    pos_inputs = pos_seqs
    if neg_seqs is not None:
        neg_inputs = neg_seqs

    # Get model outputs (logits)
    with torch.no_grad():
        pos_logits = model(pos_inputs)  # Shape: (N_pos, 2)
        if neg_seqs is not None:
            neg_logits = model(neg_inputs)  # Shape: (N_neg, 2)

    # Convert logits to probabilities (Softmax\Sigmoid)
    # pos_probs = torch.sigmoid(pos_logits[:, 1]).cpu().numpy()  # Positive class probability
    pos_probs = torch.softmax(pos_logits, dim=1)[:, 1].cpu().numpy()  # Positive class probability
    if neg_seqs is not None:
        # neg_probs = torch.sigmoid(neg_logits[:, 1]).cpu().numpy()  # Positive class probability
        neg_probs = torch.softmax(neg_logits, dim=1)[:, 1].cpu().numpy()  # Positive class probability

    # Plot histogram and KDE
    plt.figure(figsize=(8, 6))
    sns.histplot(pos_probs, bins=30, kde=True, color='blue', label="Positive Samples", stat="density", alpha=0.6)
    if neg_seqs is not None:
        sns.histplot(neg_probs, bins=30, kde=True, color='red', label="Negative Samples", stat="density", alpha=0.6)

    plt.xlabel("Predicted Probability for Positive Class")
    plt.ylabel("Density")
    plt.title(f"Distribution of Model Outputs on {set_name} Set")
    plt.legend()

    # Save the plot
    os.makedirs(f"plots/{model_type}_model/dist_model_output", exist_ok=True)
    plt.savefig(f"plots/{model_type}_model/dist_model_output/{set_name}.png")
    plt.show()


def wand_init(model_type, epochs, batch_size, neg_pos_ratio, pos_weights, learning_rate, freeze_embed_model, device):
    wandb.login(key="c8ebb98c8047d30555fd4d042ea969052ca18607")  # Replace with your API key

    # Start a new wandb run to track this script.
    run = wandb.init(
        entity="amir-weinfeld",  # Set the wandb entity where your project will be logged
        project="TCRep",  # Set the wandb project where this run will be logged
        config={
            "model_type": model_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "neg_pos_ratio": neg_pos_ratio,
            "pos_weights": pos_weights,
            "learning_rate": learning_rate,
            "freeze_embed_model": freeze_embed_model,
            "device": device,
        },
    )
    return run


def plot_output_distributions_claude(trained_model, valid_patient_inds, unique_patient_ids,
                                     valid_masks, positive_seqs, df_bld, patient_id_masks,
                                     train_patient_inds, train_inds, df_hlt, model_type, log_wandb, device='cuda'):
    """
    Create a comprehensive visualization of model output distributions across validation,
    training, and healthy patient sets.
    Parameters:
    - trained_model: The trained neural network model
    - Various data-related parameters to extract sequences and patient sets
    """
    # Set up the figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f"Model Output Distributions for {model_type} Model", fontsize=16)
    # Color palettes for different sets
    validation_colors = plt.cm.Blues(np.linspace(0.35, 0.95, len(valid_patient_inds)))
    train_colors = plt.cm.Greens(np.linspace(0.35, 0.95, 3))
    healthy_colors = plt.cm.Oranges(np.linspace(0.35, 0.95, 3))
    # Subplot 1: Validation Set Distributions
    ax1.set_title("Validation Set")
    for i, patient_ind in enumerate(valid_patient_inds):
        mask = (valid_masks[i] == 1)
        pos_seqs = np.array(positive_seqs)[mask]
        # Negative sequences for this patient
        neg_seqs = df_bld.loc[df_bld["patient_id"] == unique_patient_ids[patient_ind], "AASeq"].values
        neg_seqs = np.random.choice(neg_seqs, size=min(10 * len(pos_seqs), len(neg_seqs)), replace=False)
        # Get model outputs
        trained_model.to(device)
        trained_model.eval()
        with torch.no_grad():
            pos_logits = trained_model(pos_seqs)
            neg_logits = trained_model(neg_seqs)
        # Convert to probabilities
        pos_probs = torch.softmax(pos_logits, dim=1)[:, 1].cpu().numpy()
        neg_probs = torch.softmax(neg_logits, dim=1)[:, 1].cpu().numpy()
        # Plot KDE for positive and negative samples
        sns.kdeplot(pos_probs, ax=ax1, color=validation_colors[i], label=f'Pos Set {i}', fill=False)
        sns.kdeplot(neg_probs, ax=ax1, color=validation_colors[i], linestyle='--', label=f'Neg Set {i}')
    ax1.set_xlabel("Predicted Probability for Positive Class")
    ax1.set_ylabel("Density")
    ax1.legend()
    # Subplot 2: Training Set Distributions
    ax2.set_title("Training Set")
    for i, patient_ind in enumerate(train_patient_inds[:3]):  # Limit to first 3 for visibility
        mask_i = patient_id_masks[patient_ind]
        mask = ((mask_i == 1) & train_inds)
        pos_seqs = np.array(positive_seqs)[mask]
        # Negative sequences for this patient
        neg_seqs = df_bld.loc[df_bld["patient_id"] == unique_patient_ids[patient_ind], "AASeq"].values
        neg_seqs = np.random.choice(neg_seqs, size=min(10 * len(pos_seqs), len(neg_seqs)), replace=False)
        # Get model outputs
        trained_model.to(device)
        trained_model.eval()
        with torch.no_grad():
            pos_logits = trained_model(pos_seqs)
            neg_logits = trained_model(neg_seqs)
        # Convert to probabilities
        pos_probs = torch.softmax(pos_logits, dim=1)[:, 1].cpu().numpy()
        neg_probs = torch.softmax(neg_logits, dim=1)[:, 1].cpu().numpy()
        # Plot KDE for positive and negative samples
        sns.kdeplot(pos_probs, ax=ax2, color=train_colors[i], label=f'Pos Set {i}', fill=False)
        sns.kdeplot(neg_probs, ax=ax2, color=train_colors[i], linestyle='--', label=f'Neg Set {i}')
    ax2.set_xlabel("Predicted Probability for Positive Class")
    ax2.set_ylabel("Density")
    ax2.legend()
    # Subplot 3: Healthy Patients Distributions
    ax3.set_title("Healthy Patients")
    healthy_patients = df_hlt["patient_id"].unique()
    for i in range(3):
        patient = healthy_patients[i]
        seqs = df_hlt.loc[df_hlt["patient_id"] == patient, "AASeq"].values
        seqs = np.random.choice(seqs, size=2000, replace=False)
        # Get model outputs
        trained_model.to(device)
        trained_model.eval()
        with torch.no_grad():
            logits = trained_model(seqs)
        # Convert to probabilities
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        # Plot KDE for healthy patient samples
        sns.kdeplot(probs, ax=ax3, color=healthy_colors[i], label=f'Healthy Set {i}', fill=False)
    ax3.set_xlabel("Predicted Probability for Positive Class")
    ax3.set_ylabel("Density")
    ax3.legend()
    # Save the plot
    os.makedirs(f"plots/{model_type}_model/dist_model_output", exist_ok=True)
    plt.savefig(f"plots/{model_type}_model/dist_model_output/comprehensive_distribution.png")
    plt.tight_layout()
    # plt.show()

    # save the figure in wandb:
    if log_wandb:
        wandb.log({"output_distributions": wandb.Image(plt)})


if __name__ == '__main__':
    # get program arguments
    model_types = ['ff', 'cvc', 'esmc']
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=model_types, default='ff', help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=6, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=330, help='Batch size for training')
    parser.add_argument('--neg_pos_ratio', type=int, default=10, help='Negative to positive sample ratio')
    parser.add_argument('--pos_weights', type=float, default=3, help='Positive class weight for loss function')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for optimizer')
    parser.add_argument('--freeze_embed_model', '-freeze', action='store_true', help='Freeze the embedding model (the classification layers are unfrozen)')
    parser.add_argument('--no_wandb_log', '-nolog', action='store_true', help='Disable Weights & Biases logging')
    args = parser.parse_args()

    model_type = args.model_type
    epochs = args.epochs
    batch_size = args.batch_size
    neg_pos_ratio = args.neg_pos_ratio
    pos_weights = args.pos_weights
    learning_rate = args.learning_rate
    freeze_embed_model = args.freeze_embed_model if model_type == 'cvc' else False  # Only CVC model can freeze the embedding model
    log_wandb = not args.no_wandb_log

    assert model_type in model_types, f"Model type must be one of {model_types}"

    print("RUN CONFIGURATION:")
    print(f"\tModel Type: {args.model_type}")
    print(f"\tEpochs: {args.epochs}")
    print(f"\tBatch Size: {args.batch_size}")
    print(f"\tNegative to Positive Ratio: {args.neg_pos_ratio}")
    print(f"\tPositive Class Weight: {args.pos_weights}")
    print(f"\tLearning Rate: {args.learning_rate}")
    print("\n")

    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # wandb init
    if log_wandb:
        run = wand_init(
            model_type=model_type,
            epochs=epochs,
            batch_size=batch_size,
            neg_pos_ratio=neg_pos_ratio,
            pos_weights=pos_weights,
            learning_rate=learning_rate,
            freeze_embed_model=freeze_embed_model,
            device=device,
        )

    # Load studies
    disease = 'Multiple sclerosis'
    df = get_all_usable_disease_data(disease=disease)
    df_h = get_all_usable_healthy_data()

    # Get all common sequences
    if TO_DISPLAY_NUMBER_OF_COMMON_SEQUENCES:
        for i in range(2, 6):
            print(f"Number of Patients: {i}")
            all_common_seqs = find_all_common_sequences(df, num_of_patients=i)
            print(f"Number of unique common sequences: {len(all_common_seqs)}")
            all_common_seqs_healthy = find_all_common_sequences(df_h, num_of_patients=i)
            print(f"Number of unique common sequences (Healthy): {len(all_common_seqs_healthy)}")
            all_common_seqs = all_common_seqs - all_common_seqs_healthy
            print(f"Number of unique common sequences (Disease / Healthy): {len(all_common_seqs)}")
            print()

    # Display common sequences in disease and healthy samples
    if TO_DISPLAY_COMMON_SEQUENCES:
        l = 8
        display_common_sequences_figure(df, df_h, l=l)
        display_common_sequences_figure_healthy(df_h, l=l)

    # Choosing cell type
    cell_type = ['DC8', 'CD4', 'ALL'][2]
    if cell_type != 'ALL':
        # reading blood samples
        df_bld = df[df['cell_type'] == cell_type]
        # reading healthy study:
        df_hlt = df_h[df_h['cell_type'] == cell_type]
    else:
        df_bld, df_hlt = df, df_h

    # TODO: This code checks the intersection of healthy and disease samples with the article
    # healthy_unique = set(df_hlt['AASeq'].unique())
    # df_article = get_full_healthy_synapse_mal_id_dataframe(healthy_unique, get_all=True)
    # article_unique = set(df_article['AASeq'].unique())
    # print(f"Number of sequences in the article: \t{len(df_article['AASeq'])}")
    # print(f"Number of unique sequences in the article: {len(df_article['AASeq'].unique())}")
    # print(f"Number of sequences in the intersection of healthy and article: {len(healthy_unique & article_unique)}")
    # print(f"Number of sequences in the intersection of disease and article: {len(set(df_bld['AASeq']) & article_unique)}")
    # exit(0)

    # Loading all valid sequences for disease and healthy samples
    def get_positive_negative(num_of_patients=3):
        all_common_seqs = find_all_common_sequences(df_bld, num_of_patients=num_of_patients)
        valid_seqs_healthy = find_all_common_sequences(df_hlt, num_of_patients=num_of_patients)
        all_common_seqs = all_common_seqs - valid_seqs_healthy
        # choosing valid samples according to their re-occurrence in different patients and a given distance
        valid_seqs_disease = calculate_valid_near_sequences(df_bld, save_name=f'disease_{cell_type}_neighbours{num_of_patients}', lev_dist_accept=1,
                                                            num_of_patients=num_of_patients, all_common_seqs=all_common_seqs)

        positive_seqs = set(valid_seqs_disease)
        print(f"Valid Disease Sequence (num of common = {num_of_patients}): {len(positive_seqs)}")

        # Extract valid letters
        valid_letters = set(''.join(valid_seqs_healthy))
        # Group healthy sequences by length
        length_groups = {}
        for seq in valid_seqs_healthy:
            length_groups.setdefault(len(seq), set()).add(seq)
        # Process each length group separately
        for seq_len, seq_group in length_groups.items():
            # Generate neighbors for this group
            neighbors = generate_neighbors(seq_group, valid_letters)
            # Remove neighbors from positive_seqs immediately
            positive_seqs -= neighbors  # This prevents storing all neighbors
        negative_seqs = set(valid_seqs_healthy)  # Negative sequences remain unchanged
        return positive_seqs, negative_seqs

    # positive_seqs2, negative_seqs2 = get_positive_negative(num_of_patients=2)
    positive_seqs, negative_seqs = get_positive_negative(num_of_patients=3)
    all_common_seqs = find_all_common_sequences(df_bld, num_of_patients=3)
    valid_seqs_healthy = find_all_common_sequences(df_hlt, num_of_patients=3)
    all_common_seqs = all_common_seqs - valid_seqs_healthy
    positive_seqs.update(all_common_seqs)

    # make list and sort
    positive_seqs = list(positive_seqs)
    positive_seqs.sort()
    # negative_seqs = list(negative_seqs)
    # negative_seqs.sort()
    # shuffle according to a certain seed
    np.random.seed(42)
    np.random.shuffle(positive_seqs)
    # np.random.shuffle(negative_seqs)

    # getting patient id masks in order to do k-fold by patient (according to synovial samples)
    unique_patient_ids = df_bld["patient_id"].unique()
    masks = []
    for patient in unique_patient_ids:
        # Get sequences that belong to the current patient
        patient_seqs = set(df_bld.loc[df_bld["patient_id"] == patient, "AASeq"])
        # Create a mask for sequences
        mask = np.array([1 if seq in patient_seqs else 0 for seq in positive_seqs])
        if 1 in mask:
            masks.append(mask)
    # Convert to ndarray
    patient_id_masks = np.array(masks)  # Shape: (num_unique_patients, len(positive_seqs))

    # pick index of 10 unique patients from unique_patient_ids as test patients and the rest as train patients
    num_test_patients = 8
    unique_patient_ids = np.random.permutation(unique_patient_ids)
    test_patient_ids = unique_patient_ids[:num_test_patients]
    test_patient_ids, valid_patient_ids = test_patient_ids[:num_test_patients // 2], test_patient_ids[num_test_patients // 2:]
    train_patient_ids = unique_patient_ids[num_test_patients:]
    # now translate back to the inds according to unique_patient_ids
    test_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in test_patient_ids])
    valid_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in valid_patient_ids])
    train_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in train_patient_ids])

    # get the masks for the test and train patients
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
    print(f"Number of Positive Sequences in General: {len(positive_seqs)}")
    print(f"Number of Positive Sequences in Test: {sum(test_inds)}, Percentage: {sum(test_inds) / len(positive_seqs) * 100:.2f}%")
    print(f"Number of Positive Sequences in Valid: {sum(valid_inds)}, Percentage: {sum(valid_inds) / len(positive_seqs) * 100:.2f}%")
    print(f"Number of Positive Sequences in Train: {sum(train_inds)}, Percentage: {sum(train_inds) / len(positive_seqs) * 100:.2f}%\n")

    # Get the positive sequences for the test and train sets
    test_pos_seqs = np.array(positive_seqs)[test_inds]
    valid_pos_seqs = np.array(positive_seqs)[valid_inds]
    train_pos_seqs = np.array(positive_seqs)[train_inds]
    # Get the negative sequences
    neg_seqs = df_bld[df_bld['patient_id'].isin(train_patient_ids)]['AASeq'].unique()
    test_neg_seqs = df_bld[df_bld['patient_id'].isin(test_patient_ids)]['AASeq'].unique()
    valid_neg_seqs = df_bld[df_bld['patient_id'].isin(valid_patient_ids)]['AASeq'].unique()

    if model_type == 'ff':
        max_seq_len = max(len(seq) for seq in positive_seqs)
        model = FeedForwardClassifier(max_seq_len)
    elif model_type == 'cvc':
        model = CVCClassifierModel(batch_size=batch_size, freeze_embed_model=freeze_embed_model, device=device)
        # model = CVCClassifierModel(batch_size=1024 // 2, device=device)
    elif model_type == 'esmc':
        model = ESMCFeedForwardClassifier(device=device)
    else:
        raise ValueError(f"Model type {model_type} is not supported")

    # load the model if possible
    model_cache = f"cache/{model_type}_model.pth"
    if os.path.exists(model_cache) and not TO_RETRAIN_CLASSIFIER_MODEL:
        trained_model = model
        # load on the device
        trained_model.load_state_dict(torch.load(model_cache, map_location=device))
    else:
        # Training the model and saving it
        trained_model, history = train_model(model, train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs,
                                             epochs=epochs,
                                             lr=learning_rate,
                                             pos_batch_size=batch_size // neg_pos_ratio,
                                             neg_pos_ratio=neg_pos_ratio,
                                             log_wandb=log_wandb)
        display_training_results(history, model_type)
        torch.save(trained_model.state_dict(), model_cache)

    # # Use the model to display the histogram of the output distribution on validation set
    # for i, patient_ind in enumerate(valid_patient_inds):
    #     mask = (valid_masks[i] == 1)
    #     pos_seqs = np.array(positive_seqs)[mask]
    #     neg_seqs = df_bld.loc[df_bld["patient_id"] == unique_patient_ids[patient_ind], "AASeq"].values
    #     neg_seqs = np.random.choice(neg_seqs, size=min(10 * len(pos_seqs), len(neg_seqs)), replace=False)
    #
    #     plot_output_distribution(trained_model, f'Validation (set {i})', model_type, pos_seqs, neg_seqs, device=device)
    #
    # # Display of train set
    # for i in range(3):
    #     mask_i = patient_id_masks[train_patient_inds[i]]
    #     mask = ((mask_i == 1) & train_inds)
    #     pos_seqs = np.array(positive_seqs)[mask]
    #     neg_seqs = df_bld.loc[df_bld["patient_id"] == unique_patient_ids[train_patient_inds[i]], "AASeq"].values
    #     neg_seqs = np.random.choice(neg_seqs, size=min(10 * len(pos_seqs), len(neg_seqs)), replace=False)
    #
    #     plot_output_distribution(trained_model, f'Train (set {i})', model_type, pos_seqs, neg_seqs, device=device)
    #
    # # Display of healthy patients
    # healthy_patients = df_hlt["patient_id"].unique()
    # for i in range(3):
    #     patient = healthy_patients[i]
    #     seqs = df_hlt.loc[df_hlt["patient_id"] == patient, "AASeq"].values
    #     seqs = np.random.choice(seqs, size=2000, replace=False)
    #
    #     plot_output_distribution(trained_model, f'Healthy (set {i})', model_type, seqs, neg_seqs=None, device=device)

    plot_output_distributions_claude(trained_model, valid_patient_inds, unique_patient_ids,
                              valid_masks, positive_seqs, df_bld, patient_id_masks,
                              train_patient_inds, train_inds, df_hlt, model_type, log_wandb, device)

    exit(0)

    """
    # Calculating embeddings (or loading if it is available)
    embed_type = ['esmc', 'esmc_finetuning', 'cvc'][2]
    embed_bld = get_cached_embeddings(positive_seqs, disease, name=f'{disease}_{cell_type}_{embed_type}_bld' + name_opt, embed_type=embed_type)
    embed_hlt = get_cached_embeddings(negative_seqs, "healthy", name=f'{cell_type}_{embed_type}_h' + name_opt, embed_type=embed_type)

    # Displaying the results hyperparameters
    n_neighbors = 9
    neg_to_pos_ratio = 3

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

    # Display success figure per patient
    if TO_DISPLAY_ACCURACY_BIN_BY_DIST:
        display_accuracy_bin_by_dist_figure(cd4_syn, cd4_h, cd4_bld, cd4_syn_patient_id_masks, cd4_bld_patient_id_masks,
                                           sr_cd4_syn_vld, sr_cd4_bld_vld, valid_seqs_cd4_h,
                                           k_fold_type, study.name, ratio=neg_to_pos_ratio, n_neighbors=n_neighbors,
                                           cd_type='4', name_opt=name_opt)
        display_accuracy_bin_by_dist_figure(cd8_syn, cd8_h, cd8_bld, cd8_syn_patient_id_masks, cd8_bld_patient_id_masks,
                                           sr_cd8_syn_vld, sr_cd8_bld_vld, valid_seqs_cd8_h,
                                           k_fold_type, study.name, ratio=neg_to_pos_ratio, n_neighbors=n_neighbors,
                                           cd_type='8', name_opt=name_opt)

    # Displaying the results
    if TO_DISPLAY_RESULTS:
        print(f"Samples {cell_type} Disease: {len(embed_bld)}, Healthy: {len(embed_hlt)}")
        mean_acc, std_acc = process_and_evaluate(embed_bld, embed_hlt,
                                                 patient_id_masks, k_fold_type,
                                                 cell_type=cell_type, name_opt=name_opt,
                                                 ratio=neg_to_pos_ratio, n_neighbors=n_neighbors,
                                                 study_name=disease)
        print(f"{cell_type} - KNN {n_neighbors} neighbours ({len(patient_id_masks)}-Fold): Accuracy: {mean_acc:.3f} ± {std_acc:.3f}")
    """