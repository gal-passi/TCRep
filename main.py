import warnings
from gc import freeze
from sched import scheduler

from pyarrow.dataset import dataset
from triton.language.semantic import device_print
warnings.simplefilter("ignore", category=FutureWarning)
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
from cache_handler import load_model_state
from inference.plot_handler import plot_output_distributions_claude, plot_output_distributions_per_patient, plot_output_distributions_per_patient_new, plot_output_distributions_unseen_ms, display_ratio_figures
import yaml
from collections import defaultdict
from dataset_loader import DatasetLoader


# Constants
VALID_SEQ_CACHE = "cache/valid_sequences"
TO_DISPLAY_LENGTHS_HIST = False
TO_DISPLAY_COMMON_SEQUENCES = True
TO_DISPLAY_ACCURACY_BIN_BY_DIST = False
TO_DISPLAY_RESULTS = False
TO_DISPLAY_RESULTS_PLOT_TSNE = False
TO_DISPLAY_NUMBER_OF_COMMON_SEQUENCES = False
TO_LOAD_FULL_SYNAPSE_DATA = False
TO_DISPLAY_RATIO_FIGURES = False

dataset_loader = None

# Best CVC Model params: --model_type "cvc" --epochs 22 --loss_type "ce_entropy" -scrit --reg_coef 0.3 --pos_weights 5 --learning_rate 0.0025 --embedding_lr 0.00005 --scheduler_type "ReduceLROnPlateau" --dropout 0 -nolog -dont_inference -dont_plot
# Best CVC Model params: --model_type "cvc" --epochs 30 --loss_type "ce_entropy" -scrit --reg_coef 0.3 --pos_weights 5.75 --learning_rate 0.0025 --embedding_lr 0.00005 --scheduler_type "ReduceLROnPlateau" --dropout 0 -nolog -dont_inference -dont_plot
# Best Article Model Pa: --model_type "cvc" --epochs 22 --loss_type "ce_entropy" -scrit --reg_coef 0.3 --pos_weights 5 --learning_rate 0.0025 --embedding_lr 0.00005 --scheduler_type "ReduceLROnPlateau" --dropout 0 -nolog -dont_inference -dont_plot --dataset_type "article" --test_mode_epoch 18


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
        raise NotImplementedError("t-SNE visualization is not implemented in this code.")
        # tsne = TSNE(n_components=2, perplexity=5, random_state=42)
        # X_tsne = tsne.fit_transform(np.concatenate([X_bld]))
        # X_tsne_syn = X_tsne[:len(syn)]
        # X_tsne_bld = X_tsne[len(syn):len(syn) + len(bld)]
        #
        # # plotting
        # plt.figure(figsize=(12, 8))
        # plt.scatter(X_tsne_bld[:, 0], X_tsne_bld[:, 1],
        #             c='red', label='Blood', alpha=0.4)
        # plt.scatter(X_tsne_syn[:, 0], X_tsne_syn[:, 1],
        #             c='blue', label='Synovial Fluid', alpha=0.4)
        # plt.legend()
        # plt.title(f't-SNE Visualization of CD{cd_type} Data')
        # plt.xlabel('t-SNE 1')
        # plt.ylabel('t-SNE 2')
        # # save plot
        # plots_folder = f"plots/{study_name}"
        # os.makedirs(plots_folder, exist_ok=True)
        # plt.savefig(os.path.join(plots_folder, f"tsne_cd{cd_type}{name_opt}.png"))
        # plt.show()

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


def combine_to_dataframe(metrics_data, additional_values):
    """
    Combines two variables into a single pandas DataFrame.

    Parameters:
    metrics_data (list): List of dictionaries containing metrics
    additional_values (list): List of additional values to be added as a column

    Returns:
    pandas.DataFrame: Combined DataFrame with all data
    """
    # Convert the first variable (list of dictionaries) to a DataFrame
    df = pd.DataFrame(metrics_data)

    # Add the second variable as a new column
    df['std'] = additional_values

    # Ensure the length of additional_values matches the number of rows in the DataFrame
    if len(additional_values) != len(df):
        raise ValueError(
            f"Length mismatch: metrics_data has {len(df)} entries but additional_values has {len(additional_values)} entries")

    return df


def display_common_sequences_figure(dataset_loader, df, df_h, dataset_type, l=8, log_space=True):
    base_plot_save_path = f"plots/common_seqs/{dataset_type}"
    os.makedirs(base_plot_save_path, exist_ok=True)

    if os.path.exists(os.path.join(base_plot_save_path, 'plot_common_sequences.png')):
        print(f"Common Sequences plot already exists for dataset {dataset_type}, skipping...")
        return

    # find max len of uniques patient_id
    if l == None:
        l = min(1 + len(df_h['patient_id'].unique()), len(df['patient_id'].unique())) + 1

    # check if study_id is in the df
    if dataset_type == 'cmv':
        study_groups = df.groupby('study_id')['patient_id'].unique().apply(list)
        rand_patients = [np.random.choice(x, size=min(15, len(x)), replace=False) for x in study_groups]
        rand_patients = list(chain(*rand_patients))
    elif 'study_id' in df.columns and df.iloc[0]['study_id'] == 'article2':
        rand_patients = np.random.choice(df['patient_id'].unique(), size=15, replace=False)
    elif 'study_id' in df.columns:
        study_groups = df.groupby('study_id')['patient_id'].unique().apply(list)
        rand_patients = [np.random.choice(x, size=5, replace=False) for x in study_groups]
        rand_patients = list(chain(*rand_patients))
    else:
        # random patients from the df
        rand_patients = np.random.choice(df['patient_id'].unique(), size=15, replace=False)
    df = df[df['patient_id'].isin(rand_patients)]

    # calculate common sequences in disease and healthy samples
    value_to_take = "percent_of_total"  # "percent_of_total" or "num_common"
    x_disease_list = [dataset_loader.common_aaseq_analysis(df, num_of_patients=i, mode=1) for i in range(2, l)]
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
        x_healthy = [dataset_loader.common_aaseq_analysis(df_h_comb, num_of_patients=i, mode=2) for i in range(2, l)]
        return x_healthy

    # Average the results of all patients with disease
    x_healthy_list_all = [calculate_common_healthy(patient_id_bld) for patient_id_bld in df['patient_id'].unique()]
    x_healthy_list = [[y[0][value_to_take] for y in x] for x in x_healthy_list_all]
    x_healthy_list_std = [[y[1] for y in x] for x in x_healthy_list_all]
    x_healthy = np.array(x_healthy_list).mean(axis=0)
    x_healthy_std = np.array(x_healthy_list_std).mean(axis=0)

    # Average the results of all patients with disease and healthy then save them to a csv file
    x_avg_hlt = average_dicts(x_healthy_list_all)
    disease_df = combine_to_dataframe([x[0] for x in x_disease_list], x_disease_std)
    healthy_df = combine_to_dataframe(x_avg_hlt, x_healthy_std)
    # Save the dfs
    # disease_df.to_csv("cache/disease_df.csv", index=False)
    # healthy_df.to_csv("cache/healthy_df.csv", index=False)

    if log_space:
        x_disease = np.log(x_disease)
        x_healthy = np.log(x_healthy)

    # Figure without legend
    plt.figure(figsize=(6, 6), dpi=600)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.75)
    plt.plot(range(2, len(x_disease) + 2), x_disease, label="Patients", color="#FFA500")
    plt.plot(range(2, len(x_healthy) + 2), x_healthy, label="Healthy", color="#7BC8F6")
    plt.fill_between(range(2, len(x_disease) + 2), x_disease - x_disease_std, x_disease + x_disease_std,
                     color="#FFA500", alpha=0.2)
    plt.fill_between(range(2, len(x_healthy) + 2), x_healthy - x_healthy_std, x_healthy + x_healthy_std,
                     color="#7BC8F6", alpha=0.2)
    # add the percentage of common sequences in the plot with rounded values
    for i, txt in enumerate(x_disease):
        plt.annotate(f"{txt:.2f}", (i + 2, x_disease[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    for i, txt in enumerate(x_healthy):
        plt.annotate(f"{txt:.2f}", (i + 2, x_healthy[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.xlabel("Number of Patients")
    plt.ylabel("Percentage of Common Sequences")
    plt.title("Percentage of Common Sequences in Patients" + (" (Log Scale)" if log_space else ""))
    plt.ylim(min(min(x_disease), min(x_healthy)),
             max(max(x_disease + x_disease_std), max(x_healthy + x_healthy_std)) * 1.)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(base_plot_save_path, 'plot_common_sequences_nolegend.png'))
    plt.legend(framealpha=1.0)
    plt.savefig(os.path.join(base_plot_save_path, 'plot_common_sequences.png'))
    plt.show()


def display_common_sequences_figure_healthy(dataset_loader, df_h, l=8, log_space=True):
    if 'study_id' in df_h.columns:
        study_groups = df_h.groupby('study_id')['patient_id'].unique().apply(list)
        rand_patients = [np.random.choice(x, size=min(5, len(x)), replace=False) for x in study_groups]
        rand_patients = list(chain(*rand_patients))
    else:
        rand_patients = np.random.choice(df_h['patient_id'].unique(), size=15, replace=False)
    df_h = df_h[df_h['patient_id'].isin(rand_patients)]

    # calculate common sequences in healthy samples
    value_to_take = "percent_of_total"  # "percent_of_total" or "num_common"
    x_healthy_list = [dataset_loader.common_aaseq_analysis(df_h, num_of_patients=i, mode=1) for i in range(2, l)]
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


def wand_init(model_type, loss_type, dataset_type, epochs, batch_size, neg_pos_ratio, pos_weights,
              learning_rate, reg_coef, freeze_embed_model, special_criterion, embedding_lr, ch_dropout,
              scheduler_type, cvc_layers_to_train, k_fold, lora, device):
    wandb.login(key="c8ebb98c8047d30555fd4d042ea969052ca18607")  # Replace with your API key

    # Start a new wandb run to track this script.
    run = wandb.init(
        entity="amir-weinfeld",  # Set the wandb entity where your project will be logged
        project="TCRep",  # Set the wandb project where this run will be logged
        config={
            "model_type": model_type,
            "loss_type": loss_type,
            "dataset_type": dataset_type,
            "epochs": epochs,
            "batch_size": batch_size,
            "neg_pos_ratio": neg_pos_ratio,
            "pos_weights": pos_weights,
            "learning_rate": learning_rate,
            "reg_coef": reg_coef,
            "freeze_embed_model": freeze_embed_model,
            "special_criterion": special_criterion,
            "embedding_lr": embedding_lr,
            "ch_dropout": ch_dropout,
            "scheduler_type": scheduler_type,
            "cvc_layers_to_train": cvc_layers_to_train,
            "k_fold": k_fold,
            "lora": lora,
            "device": device,
        },
        notes="Added dropout on classification head of 0.2",
    )
    return run


def sweep_model():
    wandb.init()

    # Define the sweep configuration
    model_type = wandb.config.model_type
    loss_type = wandb.config.loss_type
    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    neg_pos_ratio = wandb.config.neg_pos_ratio
    pos_weights = wandb.config.pos_weights
    learning_rate = wandb.config.learning_rate
    reg_coef = wandb.config.regularization_coefficient
    freeze_embed_model = wandb.config.freeze_embed_model
    special_criterion = wandb.config.special_criterion
    embedding_lr = wandb.config.embedding_lr
    ch_dropout = wandb.config.classification_dropout
    log_wandb = not wandb.config.no_wandb_log
    scheduler_type = wandb.config.scheduler_type.lower()
    cvc_layers_to_train = wandb.config.cvc_layers_to_train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the dataset
    global dataset_loader
    df_bld, df_hlt = dataset_loader.get_dfs()
    positive_seqs = dataset_loader.positive_seqs
    train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs = dataset_loader.get_seqs()
    # train_patient_ids, valid_patient_ids, test_patient_ids = dataset_loader.get_patient_ids()
    train_patient_inds, valid_patient_inds, test_patient_inds = dataset_loader.get_patient_inds()
    train_masks, valid_masks, test_masks = dataset_loader.get_masks()
    train_inds = dataset_loader.train_inds
    unique_patient_ids = dataset_loader.unique_patient_ids
    patient_id_masks = dataset_loader.patient_id_masks
    aaseq_to_ratio = dataset_loader.get_aaseq_to_ratio_func()

    # Initialize Weights & Biases
    if model_type == 'ff':
        max_seq_len = max(len(seq) for seq in positive_seqs)
        model = FeedForwardClassifier(max_seq_len).to(device)
    elif model_type == 'cvc':
        model = CVCClassifierModel(batch_size=batch_size, ch_dropout=ch_dropout, freeze_embed_model=freeze_embed_model,
                                   cvc_layers_to_train=cvc_layers_to_train, device=device)
    elif model_type == 'esmc':
        model = ESMCFeedForwardClassifier(device=device)
    else:
        raise ValueError(f"Model type {model_type} is not supported")

    # load the model if possible
    trained_model, history = train_model(model, train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs,
                                         epochs=epochs,
                                         lr=learning_rate,
                                         pos_batch_size=batch_size // neg_pos_ratio,
                                         neg_pos_ratio=neg_pos_ratio,
                                         log_wandb=log_wandb,
                                         model_type=model_type,
                                         loss_type=loss_type,
                                         freeze_embed_model=freeze_embed_model,
                                         special_criterion=special_criterion,
                                         embedding_lr=embedding_lr,
                                         reg_coef=reg_coef,
                                         pos_weights=pos_weights,
                                         scheduler_type=scheduler_type,
                                         aaseq_to_ratio=aaseq_to_ratio,
                                         args=args,
                                         )

    # Plotting distributions per patient (new)
    plot_output_distributions_per_patient_new(trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
                                              test_masks, valid_masks, positive_seqs, df_bld,
                                              df_hlt, model_type, log_wandb, args, device)

    # Plotting the output distributions
    plot_output_distributions_claude(trained_model, valid_patient_inds, unique_patient_ids,
                                     valid_masks, positive_seqs, df_bld, patient_id_masks,
                                     train_patient_inds, train_inds, df_hlt, model_type, log_wandb, args, device)

    # Other distribution plot
    plot_output_distributions_per_patient(trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
                                          test_masks, valid_masks, positive_seqs, df_bld,
                                          df_hlt, model_type, log_wandb, args, device)


if __name__ == '__main__':
    # get program arguments
    model_types = ['ff', 'cvc', 'esmc']
    loss_types = ['ce', 'ce_l2', 'ce_entropy']
    scheduler_types = ['None', 'StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'ExponentialLR']
    dataset_types = ['ms', 'article', 'article2', 'cmv', 'article_sle', 'ms_plus_article2_ms']  # ms is TCRdb Multiple Sclerosis, article is Mal-ID Diabetes Type 1, article 2 is TCR MS CSF dataset, CMV is TCRdb CMV.
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=model_types, default='cvc', help='Type of model to train')
    parser.add_argument('--loss_type', type=str, choices=loss_types, default='ce', help='Type of loss function to use')
    parser.add_argument('--scheduler_type', type=str, choices=scheduler_types, default='None', help='Type of schedulers to use')
    parser.add_argument('--dataset_type', type=str, choices=dataset_types, default='ms', help='Type of the dataset to run on')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=330, help='Batch size for training')
    parser.add_argument('--neg_pos_ratio', type=int, default=10, help='Negative to positive sample ratio')
    parser.add_argument('--pos_weights', type=float, default=3, help='Positive class weight for loss function')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='Learning rate for optimizer')
    parser.add_argument('--regularization_coefficient', '--reg_coef', type=float, default=0.25, help='Coefficient for the regularization term')
    parser.add_argument('--freeze_embed_model', '-freeze', action='store_true', help='Freeze the embedding model (the classification layers are unfrozen)')
    parser.add_argument('--special_criterion', '-scrit', action='store_true', help='Using a more complex criterion for the model (different lrs)')
    parser.add_argument('--embedding_lr', '--embed_lr', type=float, default=0.00005, help='Learning rate for optimizer of the embedding model only')
    parser.add_argument('--no_wandb_log', '-nolog', action='store_true', help='Disable Weights & Biases logging')
    parser.add_argument('--test_mode_epoch', type=int, default=-1, help='Only Inferencing mode. Loading the model instead of training in the given epoch')
    parser.add_argument('--classification_dropout', '--dropout', type=float, default=0.2, help='Dropout rate for the classification head')
    parser.add_argument('--to_sweep', '--sweep', '-sweep', action='store_true', help='Sweep the hyperparameters using Weights & Biases')
    parser.add_argument('--force_retrain', '-retrain', action='store_true', help='Forces the model to retrain even if a similar model .pth file already exists')
    parser.add_argument('-v2_inference', action='store_true', help='V2 Inference')  # TODO: REMOVE!
    parser.add_argument('-dont_inference', action='store_true', help='Do not inference')  # TODO: REMOVE!
    parser.add_argument('-dont_plot', action='store_true', help='Do not create plots')  # TODO: REMOVE!
    parser.add_argument('--cvc_layers_to_train', type=int, default=3, help='Number of layers to train in case we use the CVC model')
    parser.add_argument('--k_fold', type=int, default=0, help='K-Fold Index (0 for no k-fold)')
    parser.add_argument('--lora', '-lora', action='store_true', help='Use LoRA')
    parser.add_argument('--sweep_version', type=int, default=0, help='Version of the sweep file to use')

    args = parser.parse_args()

    model_type = args.model_type.lower()
    loss_type = args.loss_type.lower()
    dataset_type = args.dataset_type.lower()
    epochs = args.epochs
    batch_size = args.batch_size
    neg_pos_ratio = args.neg_pos_ratio
    pos_weights = args.pos_weights
    learning_rate = args.learning_rate
    reg_coef = args.regularization_coefficient if loss_type != 'ce' else 0  # Regularization only for 'ce_l2' and 'ce_entropy'
    freeze_embed_model = args.freeze_embed_model if model_type == 'cvc' else False  # Only CVC model can freeze the embedding model
    special_criterion = args.special_criterion
    embedding_lr = args.embedding_lr if special_criterion else 0  # Only used when special_criterion is True
    log_wandb = not args.no_wandb_log
    test_mode_epoch = args.test_mode_epoch
    ch_dropout = args.classification_dropout
    to_sweep = args.to_sweep
    scheduler_type = args.scheduler_type.lower()
    force_retrain = args.force_retrain
    v2_inference = args.v2_inference
    dont_inference = args.dont_inference
    dont_plot = args.dont_plot
    cvc_layers_to_train = args.cvc_layers_to_train if not freeze_embed_model else 0  # No layers to train if embedding model is frozen
    k_fold = args.k_fold if args.k_fold >= 0 else 0  # Set to 0 if negative
    to_k_fold = k_fold > 0
    lora = args.lora if model_type == 'cvc' else False  # LoRA is only applicable for CVC model
    sweep_version = args.sweep_version

    assert model_type in model_types, f"Model type must be one of {model_types}"
    assert loss_type in loss_types, f"Loss type must be one of {loss_types}"
    assert scheduler_type in [x.lower() for x in scheduler_types], f"Scheduler type must be one of {scheduler_types}"
    assert dataset_type in dataset_types, f"Dataset type must be one of {dataset_types}"
    assert not (freeze_embed_model and special_criterion), "Cannot use both freeze_embed_model and special_criterion"
    assert not (freeze_embed_model and model_type != 'cvc'), "Only CVC model can freeze the embedding model"
    assert not (to_sweep and not log_wandb), "Cannot sweep hyperparameters without logging to wandb"
    if test_mode_epoch >= 0:
        assert not log_wandb, "Cannot log to wandb in test mode"
    assert not (to_k_fold and to_sweep), "Cannot do k-fold cross-validation and sweep at the same time"
    assert not (loss_type == 'ce' and dataset_type in ['article', 'article_sle']), "Cannot use ce loss with article or article_sle datasets. Due to Ratio loss"

    print("RUN CONFIGURATION:")
    print(f"\tModel Type: {args.model_type}")
    print(f"\tLoss Type: {args.loss_type}")
    print(f"\tDataset Type: {args.dataset_type}")
    print(f"\tEpochs: {args.epochs}")
    print(f"\tBatch Size: {args.batch_size}")
    print(f"\tNegative to Positive Ratio: {args.neg_pos_ratio}")
    print(f"\tPositive Class Weight: {args.pos_weights}")
    print(f"\tLearning Rate: {args.learning_rate}")
    print(f"\tRegularization Coefficient: {args.regularization_coefficient}")
    print(f"\tFreeze Embedding Model: {args.freeze_embed_model}")
    print(f"\tSpecial Criterion: {args.special_criterion}")
    print(f"\tEmbedding Learning Rate: {args.embedding_lr}")
    print(f"\tClassification Head Dropout: {args.classification_dropout}")
    print(f"\tScheduler Type: {args.scheduler_type}")
    print(f"\tForce Retrain: {args.force_retrain}")
    print(f"\tCVC model layers to train: {args.cvc_layers_to_train}")
    print(f"\tDo K-Fold Cross-Validation: {args.k_fold}")
    print(f"\tUse LoRA: {args.lora}")
    print("\tDevice:", "cuda" if torch.cuda.is_available() else "cpu")
    print("\n")

    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    unique_patient_ids = None
    if to_k_fold:
        dataset_loader = DatasetLoader(dataset_type=dataset_type, get_only_unique_patient_ids=True)
        df_bld, df_hlt = dataset_loader.get_dfs()
        unique_patient_ids = df_bld["patient_id"].unique()
        unique_patient_ids = np.random.permutation(unique_patient_ids)
        def generate_shifted_lists(patient_ids):
            """
            Generate altered lists by shifting the original list of patient IDs.
            Each altered list is shifted by increments of 8 positions to the right.
            Stops generating lists when any element from the first 8 positions would reappear.

            Args:
                patient_ids: List of unique patient ID strings

            Returns:
                A list of altered lists
            """
            n = len(patient_ids)

            # If the list has 8 or fewer elements, we can only create one list
            if n <= 8:
                return [patient_ids.copy()]

            # Calculate how many shifts we can make without bringing back elements from first 8 positions
            first_eight = set(patient_ids[:8])
            max_shifts = (n // 8) - 1

            # Create the altered lists
            altered_lists = []

            for shift_count in range(max_shifts + 1):
                # Calculate the shift amount
                shift = (shift_count * 8) % n

                # Create a new shifted list
                shifted_list = patient_ids[shift:] + patient_ids[:shift]

                # Check if any of the first 8 elements are in the shifted list
                if len(first_eight.intersection(set(shifted_list[:8]))) > 0 and shift_count > 0:
                    print(shifted_list[:8], patient_ids[:8])

                # Add to our collection of altered lists
                altered_lists.append(shifted_list)

            return np.array(altered_lists)

        altered_lists = generate_shifted_lists(list(unique_patient_ids))
        unique_patient_ids = altered_lists[k_fold]

    dataset_loader = DatasetLoader(dataset_type=dataset_type, unique_patient_ids=unique_patient_ids, k_fold=k_fold)
    df_bld, df_hlt = dataset_loader.get_dfs()
    positive_seqs = dataset_loader.positive_seqs
    train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs = dataset_loader.get_seqs()
    train_patient_ids, valid_patient_ids, test_patient_ids = dataset_loader.get_patient_ids()
    train_patient_inds, valid_patient_inds, test_patient_inds = dataset_loader.get_patient_inds()
    train_masks, valid_masks, test_masks = dataset_loader.get_masks()
    train_inds = dataset_loader.train_inds
    unique_patient_ids = dataset_loader.unique_patient_ids
    patient_id_masks = dataset_loader.patient_id_masks
    aaseq_to_ratio = dataset_loader.get_aaseq_to_ratio_func()

    # Check that each sequences in the dataset starts with 'C' and ends with 'F'! Otherwise, raise an error
    for seq in df_bld['AASeq'].unique().tolist() + df_hlt['AASeq'].unique().tolist():
        if not (seq.startswith('C') and seq.endswith('F')):
            raise ValueError(f"Sequence {seq} does not start with 'C' and end with 'F'! (Working with dataset {dataset_type})")

    # # TODO: ADDED CODE FOR COMPARING BETWEEN OTHER ARTICLE SEQUENCES! REMOVE LATER
    # article2_data_folder = 'db/test_db/data_tcrb'
    # article2_data_files = os.listdir(article2_data_folder)
    # article2_data_files = [x for x in article2_data_files if 'CDR3_list' in x and x.endswith('2.csv')]
    # # Open all files and read the contents
    # all_article2_data = set()
    # all_article2_dfs = []
    # for file_name in article2_data_files:
    #     # read the file as .csv (include header as well)
    #     file_path = os.path.join(article2_data_folder, file_name)
    #     df = pd.read_csv(file_path, names=['AASeq', 'col 1', 'col 2', 'ratio'])
    #
    #     # normalize the ratio column
    #     df['ratio'] -= df['ratio'].min()
    #     df['ratio'] /= df['ratio'].max()
    #
    #     # add patient_id as 5th and 6th columns
    #     patient_id = file_name.split('_')[0]
    #     df['patient_id'] = patient_id
    #     df['study_id'] = 'article2'
    #
    #     # modify AASeq to start with 'C' and end with 'F'
    #     df['AASeq'] = 'C' + df['AASeq'] + 'F'
    #
    #     # add the sequences to the set
    #     all_article2_data.update(df['AASeq'].tolist())
    #     all_article2_dfs.append(df)
    #
    # # Compare between all sequences in the dataset and the article2 data
    # all_dataset_data = set(df_bld['AASeq'].tolist())
    #
    # # Find the common sequences and print statistics
    # common_sequences = all_dataset_data.intersection(all_article2_data)
    # print(f"Number of common sequences (between article 2 and MS TCRdb dataset): {len(common_sequences)}")
    #
    # # Filter the article2 data to only include patients with enough samples
    # article2_df = pd.concat(all_article2_dfs, ignore_index=True)
    # patients_with_samples = [x[0] for x in article2_df.groupby('patient_id')['AASeq'] if len(x[1]) >= 2000]
    # article2_df = article2_df[article2_df['patient_id'].isin(patients_with_samples)]
    # display_common_sequences_figure(dataset_loader, article2_df, df_hlt, dataset_type, l=8)


    # Save "train_pos_seqs, train_neg_seqs, valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs" to cache/temp_split_MS_data
    if dataset_type == 'ms' and not os.path.exists('cache/temp_split_MS_data.npz'):
        np.savez('cache/temp_split_MS_data.npz', train_pos_seqs=train_pos_seqs, neg_seqs=neg_seqs,
                 valid_pos_seqs=valid_pos_seqs, valid_neg_seqs=valid_neg_seqs, test_pos_seqs=test_pos_seqs,
                 test_neg_seqs=test_neg_seqs)

    # Get all common sequences
    if TO_DISPLAY_NUMBER_OF_COMMON_SEQUENCES:
        for i in range(2, 6):
            print(f"Number of Patients: {i}")
            all_common_seqs = dataset_loader.find_all_common_sequences(df_bld, num_of_patients=i)
            print(f"Number of unique common sequences: {len(all_common_seqs)}")
            all_common_seqs_healthy = dataset_loader.find_all_common_sequences(df_hlt, num_of_patients=i)
            print(f"Number of unique common sequences (Healthy): {len(all_common_seqs_healthy)}")
            all_common_seqs = all_common_seqs - all_common_seqs_healthy
            print(f"Number of unique common sequences (Disease / Healthy): {len(all_common_seqs)}")
            print()

    # Display common sequences in disease and healthy samples
    if TO_DISPLAY_COMMON_SEQUENCES:
        l = 8
        try:
            display_common_sequences_figure(dataset_loader, df_bld, df_hlt, dataset_type, l=l)
        except Exception as e:
            print(f"Error displaying common sequences figure: {e}")
            print("Skipping the display of common sequences figure.")
        # display_common_sequences_figure_healthy(dataset_loader, df_hlt, l=l)

    if TO_DISPLAY_RATIO_FIGURES:
        display_ratio_figures(df_bld, positive_seqs, aaseq_to_ratio, dataset_type)

    # TODO: There is a small problem with reloading checkpoints:
    #  The checkpoint continues from the next sweep id instead of re-running the last crashed sweep id.
    if to_sweep:
        file_version = '' if sweep_version == 0 else f'_v{sweep_version}'
        sweep_id_path = f'sweep_yaml/sweep_id{file_version}.txt'

        with open(f'sweep_yaml/sweep{file_version}.yaml', 'r') as f:
            sweep_config = yaml.safe_load(f)

        # Check if a sweep ID already exists
        if os.path.exists(sweep_id_path):
            with open(sweep_id_path, 'r') as f:
                sweep_id = f.read().strip()
            print(f"Resuming existing sweep: {sweep_id}")
        else:
            sweep_id = wandb.sweep(sweep_config, project='TCRep')
            with open(sweep_id_path, 'w') as f:
                f.write(sweep_id)
            print(f"Created new sweep: {sweep_id}")

        wandb.agent(sweep_id, function=sweep_model, count=50, project='TCRep', entity='amir-weinfeld')  # Run sweeps one after the other for count runs
        exit(0)
    else:
        # wandb init
        if log_wandb:
            run = wand_init(
                model_type=model_type,
                loss_type=loss_type,
                dataset_type=dataset_type,
                epochs=epochs,
                batch_size=batch_size,
                neg_pos_ratio=neg_pos_ratio,
                pos_weights=pos_weights,
                learning_rate=learning_rate,
                reg_coef=reg_coef,
                freeze_embed_model=freeze_embed_model,
                special_criterion=special_criterion,
                embedding_lr=embedding_lr,
                ch_dropout=ch_dropout,
                scheduler_type=scheduler_type,
                cvc_layers_to_train=cvc_layers_to_train,
                k_fold=k_fold,
                lora=lora,
                device=device,
            )

        if model_type == 'ff':
            max_seq_len = max(len(seq) for seq in positive_seqs)
            model = FeedForwardClassifier(max_seq_len)
        elif model_type == 'cvc':
            model = CVCClassifierModel(batch_size=batch_size, ch_dropout=ch_dropout, cvc_layers_to_train=cvc_layers_to_train,
                                       freeze_embed_model=freeze_embed_model, lora=lora, device=device)
        elif model_type == 'esmc':
            model = ESMCFeedForwardClassifier(device=device)
        else:
            raise ValueError(f"Model type {model_type} is not supported")

        # load the model if possible
        trained_model = None
        if not force_retrain:
            if test_mode_epoch >= 0:
                trained_model = load_model_state(model, args, test_mode_epoch, device)
                if trained_model is None:
                    print(f"Model for epoch {test_mode_epoch} is not available!")
                    exit(1)
            else:
                trained_model = load_model_state(model, args, args.epochs - 1, device)
        if trained_model is None:
            # Training the model and saving it
            trained_model, history = train_model(model, train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs,
                                                 epochs=epochs,
                                                 lr=learning_rate,
                                                 pos_batch_size=batch_size // neg_pos_ratio,
                                                 neg_pos_ratio=neg_pos_ratio,
                                                 log_wandb=log_wandb,
                                                 model_type=model_type,
                                                 loss_type=loss_type,
                                                 freeze_embed_model=freeze_embed_model,
                                                 special_criterion=special_criterion,
                                                 embedding_lr=embedding_lr,
                                                 reg_coef=reg_coef,
                                                 pos_weights=pos_weights,
                                                 scheduler_type=scheduler_type,
                                                 aaseq_to_ratio=aaseq_to_ratio,
                                                 args=args,
                                                 )
            # display_training_results(history, model_type)

    if not dont_plot:
        # New distribution plot
        print("Plotting the output distributions per patient (New)")
        plot_output_distributions_per_patient_new(trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
                                              test_masks, valid_masks, positive_seqs, df_bld,
                                              df_hlt, model_type, log_wandb, args, device)

        # Plotting the output distributions
        print("Plotting the output distributions")
        plot_output_distributions_claude(trained_model, valid_patient_inds, unique_patient_ids,
                                         valid_masks, positive_seqs, df_bld, patient_id_masks,
                                         train_patient_inds, train_inds, df_hlt, model_type, log_wandb, args, device)

        # Other distribution plot
        print("Plotting the output distributions per patient")
        plot_output_distributions_per_patient(trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
                                              test_masks, valid_masks, positive_seqs, df_bld,
                                              df_hlt, model_type, log_wandb, args, device)

        # Distribution of unseen MS related dataset plot
        # if dataset_type == 'ms':
        #     print("Plotting the output distributions on unseen MS related dataset")
        #     plot_output_distributions_unseen_ms(trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
        #                                           test_masks, valid_masks, positive_seqs, df_bld,
        #                                           df_hlt, model_type, log_wandb, args, device)

    # Inference:
    if not dont_inference and not force_retrain and not log_wandb:
        print("Inference:")
        if not v2_inference:
            from inference.inference_testing import my_dist_inference # background_dist_inference  # dina_inference_suggestion
            # dina_inference_suggestion(df_bld, df_hlt, trained_model, valid_patient_ids)
            # background_dist_inference(df_bld, df_hlt, trained_model, valid_patient_ids, test_patient_ids)
            my_dist_inference(df_bld, df_hlt, trained_model, valid_patient_ids, test_patient_ids, args)
        else:
            from inference.inference_testing_v2 import background_dist_inference
            background_dist_inference(df_bld, df_hlt, trained_model, valid_patient_ids, test_patient_ids)

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
