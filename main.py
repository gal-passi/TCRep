import warnings
from gc import freeze
from sched import scheduler

from pyarrow.dataset import dataset
from torch.ao.nn.quantized.functional import threshold
from triton.language.semantic import device_print
from wandb.sdk.internal.system.assets import asset_registry
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
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, f1_score
from itertools import combinations, chain, product
import multiprocessing as mp
from collections import Counter
from embedding.embedding import get_cached_embeddings
from embedding.esmc_finetuning import load_fine_tuned_esmc, fine_tune_esmc
from models.cvc_model import CVCClassifierModel
from models.ff_model import FeedForwardClassifier
from models.esmc_ff_model import ESMCFeedForwardClassifier
from models.cvc_ensemble_model import CVCEnsembleModel
from models.cvc_cacheing_model import CVCCachingModel
from models.cvc_full_model import CVCClassifierModelFullEmbed
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
from cache_handler import get_model_config_str


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
INFERENCE_TO_RANDOM_FOREST = False
INFERENCE_TO_RF_PLOT_DIST_PER_PATIENT = INFERENCE_TO_RANDOM_FOREST and False
INFERENCE_TO_DISPLAY_OTHER_DATASET_DISTS = False
INFERENCE_CLASSIFICATION_MODEL = False
INFERENCE_TO_PLOT_EMBEDDING_MAPPINGS = False
INFERENCE_TO_CLASSIFICATION_MODEL = True
INFERENCE_RESHEF = True

dataset_loader = None

# Best CVC Model params: --model_type "cvc" --epochs 22 --loss_type "ce_entropy" -scrit --reg_coef 0.3 --pos_weights 5 --learning_rate 0.0025 --embedding_lr 0.00005 --scheduler_type "ReduceLROnPlateau" --dropout 0 -nolog -dont_inference -dont_plot
# Best CVC Model params: --model_type "cvc" --epochs 30 --loss_type "ce_entropy" -scrit --reg_coef 0.3 --pos_weights 5.75 --learning_rate 0.0025 --embedding_lr 0.00005 --scheduler_type "ReduceLROnPlateau" --dropout 0 -nolog -dont_inference -dont_plot
# Best Article Model Pa: --model_type "cvc" --epochs 22 --loss_type "ce_entropy" -scrit --reg_coef 0.3 --pos_weights 5 --learning_rate 0.0025 --embedding_lr 0.00005 --scheduler_type "ReduceLROnPlateau" --dropout 0 -nolog -dont_inference -dont_plot --dataset_type "article" --test_mode_epoch 18
# SLE MODEL: --model_type "cvc" --epochs 22 --loss_type "ce_entropy" --dataset_type "article_sle" -scrit --cvc_layers_to_train 4 --reg_coef 0.3 --pos_weights 5.25 --learning_rate 0.0025 --embedding_lr 0.00005 --scheduler_type "ReduceLROnPlateau" --dropout 0 -dont_inference -lora -nolog --test_mode_epoch 20

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


def display_common_sequences_figure(dataset_loader, df, df_h, dataset_type, l=8, log_space=True, to_recalculate=False):
    base_plot_save_path = f"plots/common_seqs/{dataset_type}"
    os.makedirs(base_plot_save_path, exist_ok=True)

    if not to_recalculate and os.path.exists(os.path.join(base_plot_save_path, 'plot_common_sequences.png')):
        print(f"Common Sequences plot already exists for dataset {dataset_type}, skipping...")
        return
    else:
        print(f"Calculating common sequences for dataset {dataset_type}...")

    # find max len of uniques patient_id
    if l == None:
        l = min(1 + len(df_h['patient_id'].unique()), len(df['patient_id'].unique())) + 1

    # check if study_id is in the df
    if dataset_type == 'cmv':
        study_groups = df.groupby('study_id')['patient_id'].unique().apply(list)
        rand_patients = [np.random.choice(x, size=min(15, len(x)), replace=False) for x in study_groups]
        rand_patients = list(chain(*rand_patients))
    elif 'article_sle' in dataset_type or 't1d' in dataset_type:
        study_groups = df.groupby('study_id')['patient_id'].unique().apply(list)
        rand_patients = [np.random.choice(x, size=min(25, len(x)), replace=False) for x in study_groups]
        rand_patients = list(chain(*rand_patients))
    elif 'study_id' in df.columns and df.iloc[0]['study_id'] == 'article2':
        rand_patients = np.random.choice(df['patient_id'].unique(), size=15, replace=False)
    elif 'study_id' in df.columns:
        study_groups = df.groupby('study_id')['patient_id'].unique().apply(list)
        rand_patients = [np.random.choice(x, size=5, replace=False) for x in study_groups]
        rand_patients = list(chain(*rand_patients))
    else:
        # random patients from the df
        rand_patients = np.random.choice(df['patient_id'].unique(), size=min(len(df['patient_id'].unique()), 25), replace=False)
    if len(rand_patients) < 15:
        rand_patients = np.random.choice(df['patient_id'].unique(), size=min(len(df['patient_id'].unique()), 25), replace=False)
    df = df[df['patient_id'].isin(rand_patients)]

    # per patient id, pick 20,000 AASeqs:
    # df = df.groupby('patient_id').apply(lambda x: x.sample(n=min(20000, len(x)), replace=False)).reset_index(drop=True)

    # calculate common sequences in disease and healthy samples
    value_to_take = "percent_of_total"  # "percent_of_total" or "num_common"
    x_disease_list = [dataset_loader.common_aaseq_analysis(df, num_of_patients=i, mode=1) for i in range(2, l)]
    x_disease = np.array([x[0][value_to_take] for x in x_disease_list])
    x_disease_std = np.array([x[1] for x in x_disease_list])

    def calculate_common_healthy(patient_id_bld, option=1):
        df1 = df[df["patient_id"] == patient_id_bld]
        if option == 1:
            # First Option: Adding all healthy samples to the df as is (samples stays the same for each patient)
            # healthy_patient_ids = df_h['patient_id'].unique()
            if 'article_sle' in dataset_type or 't1d' in dataset_type:
                rand_patients = np.random.choice(df_h['patient_id'].unique(), size=min(20, len(df_h['patient_id'].unique())), replace=False)
                rand_patients = [rand_patients]
            else:
                study_groups = df_h.groupby('study_id')['patient_id'].unique().apply(list)
                rand_patients = [np.random.choice(x, size=min(3, len(x)), replace=False) for x in study_groups]
            healthy_patient_ids = np.array(list(chain(*rand_patients)))
            random_patients = healthy_patient_ids
            # random_patients = np.random.choice(healthy_patient_ids, size=min(15, len(healthy_patient_ids)), replace=False)
            df_h_temp = df_h[df_h['patient_id'].isin(random_patients)]
            df_h_comb = df_h_temp
            # df_h_comb = pd.concat([df1, df_h_temp], ignore_index=True)  # TODO: Change back
            # per patient id, pick 20,000 AASeqs:
            # df_h_comb = df_h_comb.groupby('patient_id').apply(lambda x: x.sample(n=min(20000, len(x)), replace=False)).reset_index(drop=True)
        else:
            # Second Option: Adding random samples from healthy to the df (of the same length as the patient with disease samples)
            patient_seqs_len = len(df1)
            all_seqs_h = df_h["AASeq"]
            df_h_comb = generate_patient_samples(df1, all_seqs_h, patient_seqs_len)
        x_healthy = [dataset_loader.common_aaseq_analysis(df_h_comb, num_of_patients=i, mode=1) for i in range(1, l)]
        # x_healthy = [dataset_loader.common_aaseq_analysis(df_h_comb, num_of_patients=i, mode=2) for i in range(2, l)]  # TODO: Change back
        return x_healthy

    # Average the results of all patients with disease
    # x_healthy_list_all = [calculate_common_healthy(patient_id_bld, option=1) for patient_id_bld in df['patient_id'].unique()]
    # x_healthy_list = [[y[0][value_to_take] for y in x] for x in x_healthy_list_all]
    # x_healthy_list_std = [[y[1] for y in x] for x in x_healthy_list_all]
    # x_healthy = np.array(x_healthy_list).mean(axis=0)
    # x_healthy_std = np.array(x_healthy_list_std).mean(axis=0)

    value_to_take = "percent_of_total"  # "percent_of_total" or "num_common"
    if 'article_sle' in dataset_type or 't1d' in dataset_type:
        rand_patients = np.random.choice(df_h['patient_id'].unique(), size=min(25, len(df_h['patient_id'].unique())), replace=False)
        rand_patients = [rand_patients]
    else:
        study_groups = df_h.groupby('study_id')['patient_id'].unique().apply(list)
        rand_patients = [np.random.choice(x, size=min(3, len(x)), replace=False) for x in study_groups]
        if len(rand_patients) < 15:
            rand_patients = np.random.choice(df_h['patient_id'].unique(), size=min(25, len(df_h['patient_id'].unique())), replace=False)
            rand_patients = [rand_patients]
    healthy_patient_ids = np.array(list(chain(*rand_patients)))
    random_patients = healthy_patient_ids
    # random_patients = np.random.choice(healthy_patient_ids, size=min(15, len(healthy_patient_ids)), replace=False)
    df_h_temp = df_h[df_h['patient_id'].isin(random_patients)]
    x_healthy_list = [dataset_loader.common_aaseq_analysis(df_h_temp, num_of_patients=i, mode=1) for i in range(2, l)]
    x_healthy = np.array([x[0][value_to_take] for x in x_healthy_list])
    x_healthy_std = np.array([x[1] for x in x_healthy_list])

    # Average the results of all patients with disease and healthy then save them to a csv file
    # x_avg_hlt = average_dicts(x_healthy_list_all)
    # disease_df = combine_to_dataframe([x[0] for x in x_disease_list], x_disease_std)
    # healthy_df = combine_to_dataframe(x_avg_hlt, x_healthy_std)
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
    pass


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
              scheduler_type, cvc_layers_to_train, k_fold, lora, masking, ratio, dist_loss_type, use_nneighbors_loss, ch_type, neg_partition,
              use_similar_negatives, filter_num_of_patients, filter_to_inflate, change_negatives, sample_plots, optimizer_type, device):
    wandb.login(key="c8ebb98c8047d30555fd4d042ea969052ca18607")  # Replace with your API key

    # Start a new wandb run to track this script.
    run = wandb.init(
        entity="amir-weinfeld",  # Set the wandb entity where your project will be logged
        project="TCRep",  # Set the wandb project where this run will be logged
        config={
            "model_type": model_type,
            "loss_type": loss_type,
            "dataset_type": dataset_type,
            "optimizer_type": optimizer_type,
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
            "masking": masking,
            "ratio": ratio,
            "dist_loss_type": dist_loss_type,
            "use_nneighbors_loss": use_nneighbors_loss,
            "ch_type": ch_type,
            "neg_partition": neg_partition,
            "use_similar_negatives": use_similar_negatives,
            "filter_num_of_patients": filter_num_of_patients,
            "filter_to_inflate": filter_to_inflate,
            "change_negatives": change_negatives,
            "sample_plots": sample_plots,
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
    lora = wandb.config.lora
    masking = wandb.config.masking
    ratio = wandb.config.ratio
    dist_loss_type = wandb.config.dist_loss_type
    neg_partition = wandb.config.negative_partition
    use_similar_negatives = wandb.config.use_similar_negatives
    filter_num_of_patients = wandb.config.dataset_filter_num_of_patients
    filter_num_of_healthy = wandb.config.dataset_filter_num_of_healthy
    filter_to_inflate = not wandb.config.dataset_filter_dont_inflate
    remove_seqs_by_len = wandb.config.remove_seqs_by_len
    top_percent = wandb.config.top_percent
    top_n_seqs = wandb.config.top_n_seqs
    extra_filter = wandb.config.extra_filter
    use_nneighbors_loss = wandb.config.use_nneighbors_loss
    loss_version = wandb.config.loss_version
    dataset_type = wandb.config.dataset_type
    extra_ms_from_pregnant = wandb.config.extra_ms_from_pregnant
    plus_healthy_mal_id = wandb.config.plus_healthy_mal_id
    use_healthy_as_ms = wandb.config.use_healthy_as_ms
    # ch_type = wandb.config.ch_type
    # changing_negatives = wandb.config.changing_negatives
    changing_negatives = False
    sample_plots = wandb.config.sample_plots
    k_fold = wandb.config.k_fold
    optimizer_type = 'adam'  # TODO: We can add this to sweep config file!
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if top_percent == 'None':
        top_percent = None
    if top_n_seqs == 'None':
        top_n_seqs = None

    if 'ms_tcrdb2' in dataset_type:
        dataset_type += '_plus_hlt_article' if plus_healthy_mal_id else ''
        dataset_type += '_extra_ms' if extra_ms_from_pregnant else ''
        dataset_type += '_hlt_as_ms' if use_healthy_as_ms else ''
        dataset_type += f'_top_{top_percent}' if top_percent is not None else ''
        dataset_type += f'_top_{top_n_seqs}k' if top_n_seqs is not None else ''

    # Load the dataset
    dataset_loader = get_dataset_loader(dataset_type, k_fold=k_fold, to_k_fold=False,
                                        dist_loss_type=dist_loss_type, neg_partition=neg_partition,
                                        use_similar_negatives=use_similar_negatives, neg_pos_ratio=neg_pos_ratio,
                                        filter_num_of_patients=filter_num_of_patients,
                                        filter_num_of_healthy=filter_num_of_healthy,
                                        ratio=ratio,
                                        filter_to_inflate=filter_to_inflate, remove_seqs_by_len=remove_seqs_by_len,
                                        top_percent=top_percent, top_n_seqs=top_n_seqs, extra_filter=extra_filter,
                                        use_nneighbors_loss=use_nneighbors_loss, loss_version=loss_version,
                                        verbose=True)
    df_bld, df_hlt = dataset_loader.get_dfs()
    positive_seqs = dataset_loader.positive_seqs
    train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs = dataset_loader.get_seqs()
    train_patient_inds, valid_patient_inds, test_patient_inds = dataset_loader.get_patient_inds()
    train_masks, valid_masks, test_masks = dataset_loader.get_masks()
    train_inds = dataset_loader.train_inds
    unique_patient_ids = dataset_loader.unique_patient_ids
    patient_id_masks = dataset_loader.patient_id_masks
    aaseq_to_ratio = dataset_loader.get_aaseq_to_ratio_func()
    aaseq_to_dist = dataset_loader.get_aaseq_to_distance_func()
    aaseq_to_nneighbors = dataset_loader.get_aaseq_to_nneighbors_func()

    # Initialize Weights & Biases
    if model_type == 'ff':
        max_seq_len = max(len(seq) for seq in positive_seqs)
        model = FeedForwardClassifier(max_seq_len).to(device)
    elif model_type == 'cvc':
        model = CVCClassifierModel(batch_size=batch_size, ch_dropout=ch_dropout, freeze_embed_model=freeze_embed_model,
                                   cvc_layers_to_train=cvc_layers_to_train, lora=lora, device=device)
    elif model_type == 'esmc':
        model = ESMCFeedForwardClassifier(device=device)
    elif model_type == 'cvc_full' or model_type == 'cvc_weighted':
        method = 'weighted' if model_type == 'cvc_weighted' else 'full'
        model = CVCClassifierModelFullEmbed(batch_size=batch_size, method=method, ch_dropout=ch_dropout, cvc_layers_to_train=cvc_layers_to_train,
                                            freeze_embed_model=freeze_embed_model, lora=lora, ch_type=ch_type, device=device)
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
                                         aaseq_to_dist=aaseq_to_dist,
                                         aaseq_to_nneighbors=aaseq_to_nneighbors,
                                         masking=masking,
                                         ratio=ratio,
                                         change_negatives=changing_negatives,
                                         optimizer_type=optimizer_type,
                                         test_pos_seqs=test_pos_seqs,
                                         test_neg_seqs=test_neg_seqs,
                                         args=args,
                                         )

    print("Plotting the output distributions per patient")
    plot_output_distributions_per_patient(trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
                                          test_masks, valid_masks, positive_seqs, df_bld,
                                          df_hlt, model_type, log_wandb, sample_plots, args, device)

    # # Plotting distributions per patient (new)
    # plot_output_distributions_per_patient_new(trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
    #                                           test_masks, valid_masks, positive_seqs, df_bld,
    #                                           df_hlt, model_type, log_wandb, args, device)
    #
    # # Plotting the output distributions
    # plot_output_distributions_claude(trained_model, valid_patient_inds, unique_patient_ids,
    #                                  valid_masks, positive_seqs, df_bld, patient_id_masks,
    #                                  train_patient_inds, train_inds, df_hlt, model_type, log_wandb, args, device)
    #
    # # Other distribution plot
    # plot_output_distributions_per_patient(trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
    #                                       test_masks, valid_masks, positive_seqs, df_bld,
    #                                       df_hlt, model_type, log_wandb, args, device)


def get_dataset_loader(dataset_type, k_fold=0, to_k_fold=True, dist_loss_type='none', neg_partition=0,
                        use_similar_negatives=False, neg_pos_ratio=10, filter_num_of_patients=None, filter_num_of_healthy=None, ratio=None,
                        filter_to_inflate=False, remove_seqs_by_len=None, top_percent=None, top_n_seqs=None, extra_filter=False,
                       use_nneighbors_loss=False, loss_version=0, run_on_full_data=False, verbose=True):
    np.random.seed(42)
    # Load data
    unique_patient_ids = None
    if dataset_type == 'cmv':
        num_test_patients = 4
    elif 'article_sle' in dataset_type or 't1d' in dataset_type:
        num_test_patients = 10
    else:
        num_test_patients = 8
    if to_k_fold:
        dataset_loader = DatasetLoader(dataset_type=dataset_type, get_only_unique_patient_ids=True, top_percent=top_percent,
                                       extra_filter=extra_filter, top_n_seqs=top_n_seqs, use_nneighbors_loss=use_nneighbors_loss)
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
            if n <= num_test_patients:
                return [patient_ids.copy()]

            # Calculate how many shifts we can make without bringing back elements from first num_test_patients positions
            first_eight = set(patient_ids[:num_test_patients])
            max_shifts = (n // num_test_patients) - 1

            # Create the altered lists
            altered_lists = []

            for shift_count in range(max_shifts + 1):
                # Calculate the shift amount
                shift = (shift_count * num_test_patients) % n

                # Create a new shifted list
                shifted_list = patient_ids[shift:] + patient_ids[:shift]

                # Check if any of the first num_test_patients elements are in the shifted list
                if len(first_eight.intersection(set(shifted_list[:num_test_patients]))) > 0 and shift_count > 0:
                    print(shifted_list[:num_test_patients], patient_ids[:num_test_patients])

                # Add to our collection of altered lists
                altered_lists.append(shifted_list)

            return np.array(altered_lists)

        altered_lists = generate_shifted_lists(list(unique_patient_ids))
        unique_patient_ids = altered_lists[k_fold]
        print(f"Using unique patient IDs for k-fold {k_fold}: {unique_patient_ids}")

    dataset_loader = DatasetLoader(dataset_type=dataset_type, unique_patient_ids=unique_patient_ids,
                                   k_fold=k_fold, dist_loss_type=dist_loss_type, neg_partition=neg_partition,
                                   use_similar_negatives=use_similar_negatives, neg_pos_ratio=neg_pos_ratio,
                                   filter_num_of_patients=filter_num_of_patients, filter_num_of_healthy=filter_num_of_healthy,
                                   ratio=ratio, filter_to_inflate=filter_to_inflate,
                                   remove_seqs_by_len=remove_seqs_by_len, top_percent=top_percent, top_n_seqs=top_n_seqs, extra_filter=extra_filter,
                                   use_nneighbors_loss=use_nneighbors_loss, loss_version=loss_version, run_on_full_data=run_on_full_data, num_test_patients=num_test_patients, verbose=verbose)
    return dataset_loader


def do_sweep(sweep_version, project_name='TCRep_Sweeps'):
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
        sweep_id = wandb.sweep(sweep_config, project=project_name)
        with open(sweep_id_path, 'w') as f:
            f.write(sweep_id)
        print(f"Created new sweep: {sweep_id}")

    wandb.agent(sweep_id, function=sweep_model, count=50, project=project_name,
                entity='amir-weinfeld')  # Run sweeps one after the other for count runs


if __name__ == '__main__':
    # get program arguments
    model_types = ['ff', 'cvc', 'esmc', 'cvc_full', 'cvc_weighted']
    loss_types = ['ce', 'ce_l2', 'ce_entropy']
    scheduler_types = ['None', 'StepLR', 'ReduceLROnPlateau', 'CosineAnnealingLR', 'ExponentialLR']
    # TODO: Article 2 loading is incorrect at the moment. Gal is looking into it.
    dataset_types = ['ms', 'ms_hlt_article', 'ms_plus_hlt_article',
                     'ms_extra', 'ms_extra_hlt_article', 'ms_extra_plus_hlt_article',
                     'ms_no_healthy_ms',
                     'article', 'article2', 'cmv',
                     'article_sle', 'article_sle_hlt_ms_no_healthy_ms', 'article_sle_plus_hlt_ms_no_healthy_ms',
                     't1d', 't1d_hlt_ms_no_healthy_ms', 't1d_plus_hlt_ms_no_healthy_ms',
                     'ms_plus_article2_ms',
                     'ms_tcrdb2', 'ms_tcrdb2_no_healthy_ms', 'ms_tcrdb2_hlt_article',
                     'ms_tcrdb2_no_healthy_ms_plus_hlt_article',
                     'article_hiv', 'article_covid19', 'article_influenza',
                     'jia_tcrdb2']  # ms is TCRdb Multiple Sclerosis, article is Mal-ID Diabetes Type 1, article 2 is TCR MS CSF dataset, CMV is TCRdb CMV.
    dist_loss_types = ['none', 'v1', 'v2', 'v3', 'v4']
    ch_types = ['none', 'v1', 'v2']
    optimizer_types = ['Adam', 'Adafactor']

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
    parser.add_argument('-dont_inference', action='store_true', help='Do not inference')
    parser.add_argument('-dont_plot', action='store_true', help='Do not create plots')
    parser.add_argument('--cvc_layers_to_train', type=int, default=3, help='Number of layers to train in case we use the CVC model')
    parser.add_argument('--k_fold', type=int, default=0, help='K-Fold Index (0 for no k-fold)')
    parser.add_argument('--lora', '-lora', action='store_true', help='Use LoRA')
    parser.add_argument('--sweep_version', type=int, default=0, help='Version of the sweep file to use')
    parser.add_argument('--masking', '-mask', action='store_true', help='Use masking for the model. Only for CVC model')
    parser.add_argument('--ratio', '-ratio', action='store_true', help='Incorporate Ratio into the loss of the model during training')
    parser.add_argument('--dist_loss_type', type=str, choices=dist_loss_types, default='none', help='Type of distribution loss to use')
    parser.add_argument('--ch_type', type=str, choices=ch_types, default='none', help='Type of classification head to use')
    parser.add_argument('--negative_partition', '--neg_partition', type=int, default=0, help='Negative Partition Index (0 for no partitioning of the negative samples)')
    parser.add_argument('--to_ensemble', '--ensemble', '-ensemble', action='store_true', help='Ensemble the models (Only applicable after first training with all 1..5 negative_partitioning)')
    parser.add_argument('--use_similar_negatives', action='store_true', help='Use similar negatives to training positives for training (similar according to Levenstein distance)')
    parser.add_argument('--combine_classification', '-comb_class', action='store_true', help='Combine classification model results (Only applicable after first training with all 1..5 k-folds)')
    parser.add_argument('--dataset_filter_num_of_patients', type=int, default=3, help='Number of patients to filter the dataset by (take positive from this num of patients)')
    parser.add_argument('--dataset_filter_num_of_healthy', type=int, default=-1, help='Number of healthy subjects to filter the dataset with (remove from positives from this num of patients)')
    parser.add_argument('--dataset_filter_dont_inflate', '-no_inflate', action='store_true', help='Do not inflate the dataset when filtering positives and negatives')
    parser.add_argument('--changing_negatives', '-change_neg', action='store_true', help='Whether to run sample the negatives each epoch or not')
    parser.add_argument('--remove_seqs_by_len', type=int, default=False, help='Whether to remove sequences from valid/test sets by length or not')
    parser.add_argument('--train_vae', action='store_true', default=False, help='Train ControlVAE on positive sequences instead of classification model')
    parser.add_argument('--top_percent', type=int, default=None, help='The top percent of sequences take when loading data from TCRdb2')
    parser.add_argument('--top_n_seqs', type=int, default=None, help='The top n*1000 sequences take when loading data from TCRdb2')
    parser.add_argument('--plus_healthy_mal_id', action='store_true', default=False, help='Whether to add healthy Mal-ID sequences to the dataset or not')
    parser.add_argument('--extra_ms_from_pregnant', action='store_true', default=False, help='Whether to add extra MS data of pregnant MS study (only for MS dataset)')
    parser.add_argument('--use_healthy_as_ms', action='store_true', default=False, help='Whether to use some of the healthy patients as MS patients (only for MS dataset)')
    parser.add_argument('--extra_filter', action='store_true', default=False, help='Whether to filter the data more than the default filtering (Remove seqs of certain lengths and remove subjects with not a lot of sequences)')
    parser.add_argument('--use_nneighbors_loss', action='store_true', default=False, help='Add neighbors - common sequences - into loss calculation')
    parser.add_argument('--loss_version', type=int, default=0, help='The version of the loss to use')
    parser.add_argument('--sample_plots', type=int, default=0, help='Sampling when plotting instead of running on all sequences')
    parser.add_argument('--classification_v2', action='store_true', default=False, help='Use the new classification model with a different architecture (Version 2)')
    parser.add_argument('--optimizer_type', type=str, choices=optimizer_types, default='Adam', help='Type of optimizer to use')
    parser.add_argument('-reshef_inference', action='store_true', default=False, help='Whether to save information for Reshef inference or not')
    parser.add_argument('-reshef_filter_train', action='store_true', default=False, help='Whether to save information for Reshef inference or not')
    parser.add_argument('-reshef_negative_part', type=int, default=0, help='Part of the negative partition to use for Reshef inference (0 for no partitioning)')
    parser.add_argument('-run_on_full_data', action='store_true', default=False, help='Whether to save information for Reshef inference or not')
    parser.add_argument('-dont_cache_inference', action='store_true', default=False, help='Whether to cache the inference results or not. If True, it will not cache the results and will run inference every time.')
    args = parser.parse_args()

    to_sweep = args.to_sweep
    sweep_version = args.sweep_version
    if to_sweep:
        do_sweep(sweep_version)
        exit(0)

    model_type = args.model_type.lower()
    loss_type = args.loss_type.lower()
    dataset_type = args.dataset_type.lower()
    epochs = args.epochs
    batch_size = args.batch_size
    neg_pos_ratio = args.neg_pos_ratio
    pos_weights = args.pos_weights
    learning_rate = args.learning_rate
    reg_coef = args.regularization_coefficient if loss_type != 'ce' else 0  # Regularization only for 'ce_l2' and 'ce_entropy'
    freeze_embed_model = args.freeze_embed_model if 'cvc' in model_type else False  # Only CVC model can freeze the embedding model
    special_criterion = args.special_criterion
    embedding_lr = args.embedding_lr if special_criterion else 0  # Only used when special_criterion is True
    log_wandb = not args.no_wandb_log
    test_mode_epoch = args.test_mode_epoch
    ch_dropout = args.classification_dropout
    scheduler_type = args.scheduler_type.lower()
    force_retrain = args.force_retrain
    dont_inference = args.dont_inference
    dont_plot = args.dont_plot
    cvc_layers_to_train = args.cvc_layers_to_train if not freeze_embed_model else 0  # No layers to train if embedding model is frozen
    k_fold = args.k_fold if args.k_fold >= 0 else 0  # Set to 0 if negative
    to_k_fold = k_fold > 0
    lora = args.lora if 'cvc' in model_type else False  # LoRA is only applicable for CVC model
    masking = args.masking if 'cvc' in model_type else False  # Masking is only applicable for CVC model
    ratio = args.ratio
    dist_loss_type = args.dist_loss_type
    ch_type = args.ch_type.lower() if 'cvc' in model_type else 'none'  # Only CVC model can use dist loss
    neg_partition = args.negative_partition
    to_ensemble = args.to_ensemble
    use_similar_negatives = args.use_similar_negatives
    combine_classification = args.combine_classification
    filter_num_of_patients = args.dataset_filter_num_of_patients
    filter_num_of_healthy = args.dataset_filter_num_of_healthy if args.dataset_filter_num_of_healthy >= 0 else filter_num_of_patients
    filter_to_inflate = not args.dataset_filter_dont_inflate
    changing_negatives = args.changing_negatives
    remove_seqs_by_len = args.remove_seqs_by_len
    top_percent = args.top_percent
    top_n_seqs = args.top_n_seqs
    plus_healthy_mal_id = args.plus_healthy_mal_id
    extra_ms_from_pregnant = args.extra_ms_from_pregnant
    use_healthy_as_ms = args.use_healthy_as_ms
    extra_filter = args.extra_filter
    use_nneighbors_loss = args.use_nneighbors_loss
    loss_version = args.loss_version
    sample_plots = args.sample_plots
    classification_v2 = args.classification_v2
    optimizer_type = args.optimizer_type.lower()
    reshef_inference = args.reshef_inference
    reshef_filter_train = args.reshef_filter_train
    reshef_negative_part = args.reshef_negative_part if reshef_inference else 0
    run_on_full_data = args.run_on_full_data
    dont_cache_inference = args.dont_cache_inference

    if combine_classification and not dont_plot:
        dont_plot = True  # If combining classification, we don't plot the individual results

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
    assert not (dist_loss_type != 'none' and ratio), "Cannot use dist_loss_type and ratio at the same time"
    assert not ((neg_partition > 0) and to_sweep), "Cannot use negative partitioning and sweep at the same time"
    assert not ((neg_partition > 0) and to_ensemble), "Cannot use negative partitioning and ensemble at the same time"
    assert not (top_percent is not None and 'tcrdb2' not in dataset_type), "Cannot use top_percent when dataset_type does not contain 'tcrdb2'"
    # assert not (top_n_seqs is not None and 'tcrdb2' not in dataset_type), "Cannot use top_n_seqs when dataset_type does not contain 'tcrdb2'"
    assert not (top_percent is not None and top_n_seqs is not None), "Cannot use both top_percent and top_n_seqs at the same time"
    assert not (extra_ms_from_pregnant and 'ms' not in dataset_type), "extra_ms_from_pregnant can only be used with MS dataset of tcrdb2.0"
    assert not (use_healthy_as_ms and 'ms' not in dataset_type), "use_healthy_as_ms can only be used with MS dataset of tcrdb2.0"
    # assert not (extra_filter and 'ms_tcrdb2' not in dataset_type), "extra_filter can only be used with MS TCRdb2 dataset"
    assert not (reshef_inference and changing_negatives), "Reshef inference is not applicable if changing negatives"
    assert not (reshef_filter_train and not reshef_inference), "Reshef filter train must be on when reshef inference is on"

    if 'ms_tcrdb2' in dataset_type:
        dataset_type += '_plus_hlt_article' if plus_healthy_mal_id else ''
        dataset_type += '_extra_ms' if extra_ms_from_pregnant else ''
        dataset_type += '_hlt_as_ms' if use_healthy_as_ms else ''
        dataset_type += f'_top_{top_percent}' if top_percent is not None else ''
        dataset_type += f'_top_{top_n_seqs}k' if top_n_seqs is not None else ''
        dataset_type += f'_run_on_full_data' if run_on_full_data else ''

    print("RUN CONFIGURATION:")
    print(f"\tModel Type: {args.model_type}")
    print(f"\tLoss Type: {args.loss_type}")
    print(f"\tDataset Type: {args.dataset_type}")
    print(f"\tOptimizer Type: {args.optimizer_type}")
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
    print(f"\tTest Mode Epoch: {args.negative_partition}")
    print(f"\tUse LoRA: {args.lora}")
    print(f"\tMasking: {args.masking}")
    print(f"\tUsing Ratio: {args.ratio}")
    print(f"\tDist Loss Type: {args.dist_loss_type}")
    print(f"\tClassification Head Type: {args.ch_type}")
    print(f"\tDataset Filter Number of Patients: {args.dataset_filter_num_of_patients}")
    print(f"\tDataset Filter Number of Healthy: {args.dataset_filter_num_of_healthy}")
    print(f"\tDataset Filter Inflate: {not args.dataset_filter_dont_inflate}")
    print(f"\tNegative Partition Index: {args.negative_partition}")
    print(f"\tChanging Negatives: {args.changing_negatives}")
    print(f"\tLoss Version: {args.loss_version}")
    print(f"\tUse Neighbors Loss: {args.use_nneighbors_loss}")
    print(f"\tSample Plots: {args.sample_plots}")
    print(f"\tRun on Full Data: {args.run_on_full_data}")
    print("\tDevice:", "cuda" if torch.cuda.is_available() else "cpu")
    print("\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_loader = get_dataset_loader(dataset_type, k_fold=k_fold, to_k_fold=to_k_fold,
                                        dist_loss_type=dist_loss_type, neg_partition=neg_partition,
                                        use_similar_negatives=use_similar_negatives, neg_pos_ratio=neg_pos_ratio,
                                        filter_num_of_patients=filter_num_of_patients, filter_num_of_healthy=filter_num_of_healthy,
                                        ratio=ratio,
                                        filter_to_inflate=filter_to_inflate, remove_seqs_by_len=remove_seqs_by_len,
                                        top_percent=top_percent, top_n_seqs=top_n_seqs, extra_filter=extra_filter,
                                        use_nneighbors_loss=use_nneighbors_loss, loss_version=loss_version, run_on_full_data=run_on_full_data, verbose=True)
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
    aaseq_to_dist = dataset_loader.get_aaseq_to_distance_func()
    aaseq_to_nneighbors = dataset_loader.get_aaseq_to_nneighbors_func()

    dont_enter = True
    if not dont_enter:
        df_blds, df_hlts = [], []
        patient_ids = []
        for kf in range(1, 6):
            dataset_loader = get_dataset_loader(dataset_type, k_fold=kf, to_k_fold=True,
                                                dist_loss_type=dist_loss_type, neg_partition=neg_partition,
                                                use_similar_negatives=use_similar_negatives, neg_pos_ratio=neg_pos_ratio,
                                                filter_num_of_patients=filter_num_of_patients, filter_num_of_healthy=filter_num_of_healthy,
                                                ratio=ratio,
                                                filter_to_inflate=filter_to_inflate, remove_seqs_by_len=remove_seqs_by_len,
                                                top_percent=top_percent, top_n_seqs=top_n_seqs, extra_filter=extra_filter,
                                                use_nneighbors_loss=use_nneighbors_loss, loss_version=loss_version, verbose=True)
            df_bld, df_hlt = dataset_loader.get_dfs()
            train_patient_ids, valid_patient_ids, test_patient_ids = dataset_loader.get_patient_ids()
            df_blds.append(df_bld)
            df_hlts.append(df_hlt)
            patient_ids.append([train_patient_ids, valid_patient_ids, test_patient_ids])
        print('Done loading all k-folds datasets')

    # TODO: Adding VAE training here! dont just leave it here!
    # Training VAE
    train_vae = args.train_vae
    if train_vae:
        from vae.vae_training import run_vae_training_and_inference
        run_vae_training_and_inference(args, dataset_loader, device)

        # Note: we won't get to this part because there is an exit command in the previous vae line
        from vae.vae_training_dynamic import train_and_inference_vae
        train_and_inference_vae(args, dataset_loader, device)

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
        display_common_sequences_figure(dataset_loader, df_bld, df_hlt, dataset_type, l=min(l, len(train_patient_ids)))
        try:
            pass
        except Exception as e:
            print(f"Error displaying common sequences figure: {e}")
            print("Skipping the display of common sequences figure.")
        # display_common_sequences_figure_healthy(dataset_loader, df_hlt, l=l)

    if TO_DISPLAY_RATIO_FIGURES or ratio == True:  # TODO: REMOVE THIS ADDED LOSS TYPE SCENARIO!!!!
        display_ratio_figures(df_bld, positive_seqs, neg_seqs, aaseq_to_ratio, dataset_type)

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
            masking=masking,
            ratio=ratio,
            dist_loss_type=dist_loss_type,
            use_nneighbors_loss=use_nneighbors_loss,
            ch_type=ch_type,
            neg_partition=neg_partition,
            use_similar_negatives=use_similar_negatives,
            filter_num_of_patients=filter_num_of_patients,
            filter_to_inflate=filter_to_inflate,
            change_negatives=changing_negatives,
            sample_plots=sample_plots,
            optimizer_type=optimizer_type,
            device=device,
        )

    if model_type == 'ff':
        max_seq_len = max(len(seq) for seq in positive_seqs)
        model = FeedForwardClassifier(max_seq_len)
    elif model_type == 'cvc':
        model = CVCClassifierModel(batch_size=batch_size, ch_dropout=ch_dropout, cvc_layers_to_train=cvc_layers_to_train,
                                   freeze_embed_model=freeze_embed_model, lora=lora, ch_type=ch_type, device=device)
    elif model_type == 'esmc':
        model = ESMCFeedForwardClassifier(device=device)
    elif model_type == 'cvc_full' or model_type == 'cvc_weighted':
        method = 'weighted' if model_type == 'cvc_weighted' else 'full'
        model = CVCClassifierModelFullEmbed(batch_size=batch_size, method=method, ch_dropout=ch_dropout,
                                            cvc_layers_to_train=cvc_layers_to_train,
                                            freeze_embed_model=freeze_embed_model, lora=lora, ch_type=ch_type,
                                            device=device)
    else:
        raise ValueError(f"Model type {model_type} is not supported")

    # load the model if possible
    trained_model = None
    if not force_retrain:
        if combine_classification:
            trained_model = model
        elif to_ensemble:
            from cache_handler import get_model_dir
            cache_dir = get_model_dir(args)
            trained_model = CVCEnsembleModel(args, device, cache_dir=cache_dir, default_to_return='min')  # or 'weighted_sum'
        elif test_mode_epoch >= 0:
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
                                             aaseq_to_dist=aaseq_to_dist,
                                             aaseq_to_nneighbors=aaseq_to_nneighbors,
                                             masking=masking,
                                             ratio=ratio,
                                             change_negatives=changing_negatives,
                                             optimizer_type=optimizer_type,
                                             reshef_inference=reshef_inference,
                                             reshef_filter_train=reshef_filter_train,
                                             reshef_negative_part=reshef_negative_part,
                                             test_pos_seqs=test_pos_seqs,
                                             test_neg_seqs=test_neg_seqs,
                                             args=args,
                                             )

    if not dont_plot:
        plot_extra_sequence_logo = False
        if plot_extra_sequence_logo:
            import logomaker
            from collections import Counter

            other_seqs = np.concatenate([neg_seqs, valid_neg_seqs, test_neg_seqs])

            sequence_logo_version = 2
            if sequence_logo_version == 1:
                # === STEP 1: Find top 3 lengths in positive_seqs ===
                positive_lengths = [len(seq) for seq in positive_seqs]
                top3_lengths = [length for length, _ in Counter(positive_lengths).most_common(3)]

                # === STEP 2: Function to create Position Frequency Matrix (PFM) ===
                def get_pfm(seqs, seq_len):
                    filtered_seqs = [seq for seq in seqs if len(seq) == seq_len]
                    if not filtered_seqs:
                        return None
                    pfm = pd.DataFrame([Counter(col) for col in zip(*filtered_seqs)]).fillna(0)
                    return pfm

                # === STEP 3: Generate logos for each of top 3 lengths ===
                fig, axs = plt.subplots(len(top3_lengths), 2, figsize=(14, 4 * len(top3_lengths)))
                for i, seq_len in enumerate(top3_lengths):
                    pos_pfm = get_pfm(positive_seqs, seq_len)
                    oth_pfm = get_pfm(other_seqs, seq_len)
                    if pos_pfm is not None:
                        logomaker.Logo(pos_pfm, ax=axs[i, 0])
                        axs[i, 0].set_title(f"Positive Sequences (Length {seq_len})")
                        axs[i, 0].set_ylabel("Freq")
                    else:
                        axs[i, 0].text(0.5, 0.5, "No sequences", ha='center', va='center')
                        axs[i, 0].axis('off')
                    if oth_pfm is not None:
                        logomaker.Logo(oth_pfm, ax=axs[i, 1])
                        axs[i, 1].set_title(f"Other Sequences (Length {seq_len})")
                        axs[i, 1].set_ylabel("Freq")
                    else:
                        axs[i, 1].text(0.5, 0.5, "No sequences", ha='center', va='center')
                        axs[i, 1].axis('off')

                    for ax in axs[i]:
                        ax.set_xlabel("Position")

                plt.tight_layout()
                plt.show()
            else:
                # taking without C and F (can change to: [4:-4])
                tmp_pos_seqs = np.array([seq[1:-1] for seq in positive_seqs])  # Remove start and stop codons
                tmp_other_seqs = np.array([seq[1:-1] for seq in other_seqs])  # Remove start and stop codons

                # === STEP 1: Find top 3 most common sequence lengths ===
                positive_lengths = [len(seq) for seq in tmp_pos_seqs]
                top3_lengths = [length for length, _ in Counter(positive_lengths).most_common(3)][:1]  # Taking only 1st

                # === STEP 2: Function to create information content matrix ===
                def get_info_matrix(seqs, seq_len):
                    filtered_seqs = [seq for seq in seqs if len(seq) == seq_len]
                    if not filtered_seqs:
                        return None

                    # Create frequency matrix
                    columns = list(zip(*filtered_seqs))
                    pfm_dicts = []
                    for pos in columns:
                        count = Counter(pos)
                        pfm_dicts.append(count)

                    counts_df = pd.DataFrame(pfm_dicts).fillna(0)

                    # Ensure columns are amino acids and rows are positions (1-based)
                    counts_df.index = range(1, seq_len + 1)

                    # Convert counts to information content (bits)
                    info_df = logomaker.transform_matrix(counts_df, from_type='counts', to_type='information')
                    return info_df

                # === STEP 3: Plotting ===
                fig, axs = plt.subplots(len(top3_lengths), 2, figsize=(14, 4 * len(top3_lengths)))
                axs = axs.reshape(len(top3_lengths), -1)
                for ax in axs.flatten():
                    for spine in ax.spines.values():
                        spine.set_edgecolor('black')
                        spine.set_linewidth(1.0)

                for i, seq_len in enumerate(top3_lengths):
                    pos_matrix = get_info_matrix(tmp_pos_seqs, seq_len)
                    oth_matrix = get_info_matrix(tmp_other_seqs, seq_len)

                    # Positive
                    if pos_matrix is not None:
                        logomaker.Logo(pos_matrix, ax=axs[i, 0])
                        axs[i, 0].set_title(f"Positive Sequences (Length {seq_len})")
                        axs[i, 0].set_ylabel("Bits")
                        axs[i, 0].set_xticks(range(1, seq_len + 1))
                    else:
                        axs[i, 0].text(0.5, 0.5, "No sequences", ha='center', va='center')
                        axs[i, 0].axis('off')
                    # Other
                    if oth_matrix is not None:
                        logomaker.Logo(oth_matrix, ax=axs[i, 1])
                        axs[i, 1].set_title(f"Other Sequences (Length {seq_len})")
                        axs[i, 1].set_ylabel("Bits")
                        axs[i, 1].set_xticks(range(1, seq_len + 1))
                    else:
                        axs[i, 1].text(0.5, 0.5, "No sequences", ha='center', va='center')
                        axs[i, 1].axis('off')

                    for ax in axs[i]:
                        ax.set_xlabel("Position")
                        ax.set_ylim(0, 3.55)  # for [1:-1]
                        # ax.set_ylim(0, 1.48)  # for [4:-4]
                        # ax.set_ylim(0, 4.32)  # Maximum entropy for 20 amino acids ≈ log2(20)

                plt.tight_layout()
                plt.show()
                pass

        # if not dont_inference and not force_retrain and not log_wandb:
        #     class TrainedModelWrapper:
        #         def __init__(self, trained_model, aaseq_to_ratio, ratio_threshold=0.1e-5):
        #             self.trained_model = trained_model
        #             self.aaseq_to_ratio = aaseq_to_ratio
        #             self.ratio_threshold = ratio_threshold
        #
        #         def to(self, device):
        #             self.trained_model.to(device)
        #
        #         def eval(self):
        #             self.trained_model.eval()
        #
        #         def __call__(self, sequences):
        #             sequences = sequences[
        #                 self.aaseq_to_ratio(sequences, dont_use_function=True).values >= self.ratio_threshold]
        #             return self.trained_model(sequences)
        #     trained_model = TrainedModelWrapper(trained_model, aaseq_to_ratio)

        # New distribution plot
        np.random.seed(42)
        # from models.cvc_basic_cacheing_model import CVCBasicCachingModel
        # print("Plotting the output distributions per patient (New)")
        # caching_model = trained_model
        # # caching_model = CVCBasicCachingModel(trained_model, args, device, verbose=True)
        # # class TmpModel(nn.Module):
        # #     def __init__(self, model):
        # #         super(TmpModel, self).__init__()
        # #         self.model = model
        # #
        # #     def forward(self, x):
        # #         """
        # #         Args:
        # #             x: numpy array of shape (batch_size,) containing sequences as strings
        # #         Returns:
        # #             torch.Tensor of shape (batch_size, 2), raw logits
        # #         """
        # #         if isinstance(x, np.ndarray):
        # #             x = x.tolist()
        # #
        # #         batch_size = len(x)
        # #
        # #         # Decide for each sample whether to make it a "low score" or "high score"
        # #         rand_vals = torch.rand(batch_size)
        # #         is_low = rand_vals < 0.9  # 90% get values close to softmax [0, 1]
        # #
        # #         logits = torch.empty((batch_size, 2))
        # #
        # #         # Low confidence for dim 0 → large negative value
        # #         logits[is_low] = torch.tensor([10.0, -10.0])
        # #
        # #         # High confidence for dim 0 → large positive value
        # #         logits[~is_low] = torch.tensor([-10.0, 10.0])
        # #
        # #         rand_vals = torch.rand(batch_size)
        # #         to_change = 0.7 < rand_vals
        # #         logits[to_change] = torch.tensor([1.0, -1.0])
        # #         to_change_2 = 0.85 < rand_vals
        # #         logits[to_change_2] = torch.tensor([-1.0, 1.0])
        # #
        # #         return logits
        # # trained_model = TmpModel(trained_model)
        # # args.model_type = 'tmp_model'
        # plot_output_distributions_per_patient_new(caching_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
        #                                           test_masks, valid_masks, positive_seqs, df_bld,
        #                                           df_hlt, model_type, log_wandb, args, device)
        # Plotting the output distributions
        # print("Plotting the output distributions")
        # plot_output_distributions_claude(trained_model, valid_patient_inds, unique_patient_ids,
        #                                  valid_masks, positive_seqs, df_bld, patient_id_masks,
        #                                  train_patient_inds, train_inds, df_hlt, model_type, log_wandb, args, device)
        #
        # Other distribution plot
        print("Plotting the output distributions per patient")
        plot_output_distributions_per_patient(trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
                                              test_masks, valid_masks, positive_seqs, df_bld,
                                              df_hlt, model_type, log_wandb, sample_plots, args, device)
        # Other distribution plot
        # print("Plotting the output distributions per patient - With threshold=0.5")
        # plot_output_distributions_per_patient(trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
        #                                       test_masks, valid_masks, positive_seqs, df_bld,
        #                                       df_hlt, model_type, log_wandb, sample_plots, args, device, threshold=0.5)

        # Distribution of unseen MS related dataset plot
        # if dataset_type == 'ms':
        #     print("Plotting the output distributions on unseen MS related dataset")
        #     plot_output_distributions_unseen_ms(trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
        #                                           test_masks, valid_masks, positive_seqs, df_bld,
        #                                           df_hlt, model_type, log_wandb, args, device)

    # Inference:
    # if not dont_inference and not force_retrain and not log_wandb:
    if not dont_inference:
        print("\nInference:")

        # Inference ensemble model
        if to_ensemble and not combine_classification:
            np.random.seed(42)
            # TODO: Added this inference: remove later!!!
            # def plot_histogram_of_lengths(all_seqs, title_extra=''):
            #     lengths = [len(seq) for seq in all_seqs]
            #     possible_lengths = sorted(set(lengths))
            #     # Extend bins so each length is fully included (add +1 to the max for right edge)
            #     bins = list(range(min(possible_lengths), max(possible_lengths) + 2))
            #     plt.figure(figsize=(10, 6))
            #     counts, bins, patches = plt.hist(lengths, bins=bins, color='blue', alpha=0.7, align='left')
            #     # Annotate each bar with the count
            #     for count, patch in zip(counts, patches):
            #         if count > 0:
            #             plt.text(patch.get_x() + patch.get_width() / 2, count + 0.5, str(int(count)),
            #                      ha='center', va='bottom', fontsize=9)
            #     # Set x-ticks to all possible lengths
            #     plt.xticks(possible_lengths)
            #     plt.xlabel('Sequence Length')
            #     plt.ylabel('Count')
            #     plt.title(f'Histogram of Sequence Lengths {title_extra} - Total Sequences: {len(all_seqs)}')
            #     plt.tight_layout()
            #     plt.show()
            # all_seqs = df_bld['AASeq'].unique()
            # plot_histogram_of_lengths(all_seqs, '(All Sequences)')
            # plot_histogram_of_lengths(positive_seqs, '(Positive Sequences)')
            # plot_histogram_of_lengths(neg_seqs, '(Negative Sequences)')

            trained_model.to(device)
            from model_trainer import evaluate_model, CustomLossCriterion
            class_weights = torch.tensor([1.0, pos_weights], dtype=torch.float, device=device)
            criterion = CustomLossCriterion(loss_type=loss_type, class_weights=class_weights, R=reg_coef, ratio=ratio, aaseq_to_ratio=aaseq_to_ratio, aaseq_to_dist=aaseq_to_dist, device=device)

            # Added this to plot confusion matrices
            from inference.inference_ensemble import plot_confusion_matrices
            model_string = get_model_config_str(args)
            plot_confusion_matrices(trained_model, valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs, model_string, device)

            default_to_return = trained_model.default_to_return
            if default_to_return == 'weighted_sum':
                models_tprs = []
                for i in range(5):
                    trained_model.default_to_return = i
                    test_metrics = evaluate_model(trained_model, test_pos_seqs, test_neg_seqs, criterion, device)
                    models_tprs.append(test_metrics[12])
                model_weights = np.array(models_tprs) / np.sum(models_tprs)
                trained_model.weights = model_weights
                trained_model.default_to_return = default_to_return

            val_metrics = evaluate_model(trained_model, valid_pos_seqs, valid_neg_seqs, criterion, device)
            val_loss, val_acc, val_auc, val_prauc, val_tp, val_fp, val_tn, val_fn, val_pos_acc, val_neg_acc, val_precision, val_recall, val_tpr, val_tnr, val_fpr, val_fnr, val_f1 = val_metrics
            print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation AUC: {val_auc:.4f}, Validation PR AUC: {val_prauc:.4f}")
            print(f"Validation TPR: {val_tpr}, Validation FPR: {val_fpr}, Validation TNR: {val_tnr}, Validation FNR: {val_fnr}")
            print(f"Validation Positive Accuracy: {val_pos_acc:.4f}, Validation Negative Accuracy: {val_neg_acc:.4f}")
            print(f"Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}")
            print(f"Validation F1: {val_f1:.4f}")

        if neg_partition > 0:
            np.random.seed(42)
            from model_trainer import evaluate_model, CustomLossCriterion
            class_weights = torch.tensor([1.0, pos_weights], dtype=torch.float, device=device)
            criterion = CustomLossCriterion(loss_type=loss_type, class_weights=class_weights, R=reg_coef, ratio=ratio,
                                            aaseq_to_ratio=aaseq_to_ratio, aaseq_to_dist=aaseq_to_dist, device=device)
            val_metrics = evaluate_model(trained_model, valid_pos_seqs, valid_neg_seqs, criterion, device)
            val_loss, val_acc, val_auc, val_prauc, val_tp, val_fp, val_tn, val_fn, val_pos_acc, val_neg_acc, val_precision, val_recall, val_tpr, val_tnr, val_fpr, val_fnr, val_f1 = val_metrics
            print(
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}, Validation AUC: {val_auc:.4f}, Validation PR AUC: {val_prauc:.4f}")
            print(
                f"Validation TPR: {val_tpr}, Validation FPR: {val_fpr}, Validation TNR: {val_tnr}, Validation FNR: {val_fnr}")
            print(f"Validation Positive Accuracy: {val_pos_acc:.4f}, Validation Negative Accuracy: {val_neg_acc:.4f}")
            print(f"Validation Precision: {val_precision:.4f}, Validation Recall: {val_recall:.4f}")
            print(f"Validation F1: {val_f1:.4f}")

        # Create a dataframe with the AASeq, embedding, label and set_origin as columns
        if INFERENCE_TO_RANDOM_FOREST:
            np.random.seed(42)
            from inference.inference_testing import get_df_embeddings_onehot, get_df_embeddings
            print("Getting the embeddings the dataset and saving as df to cache")
            df_embed_onehot = get_df_embeddings_onehot(train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs)
            untrained_model = CVCClassifierModel(batch_size=batch_size, ch_dropout=ch_dropout, cvc_layers_to_train=cvc_layers_to_train,
                                                 freeze_embed_model=freeze_embed_model, lora=lora, ch_type=ch_type, device=device)
            model_for_embedding = untrained_model
            df_embed = get_df_embeddings(args, model_for_embedding, train_pos_seqs, neg_seqs,
                                         valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs, metadata_name='untrained')

            neg_to_pos_inference_ratio = 10
            print(f"Testing out random forest classifier - CVC Embedding (Taking neg to pos ratio: {neg_to_pos_inference_ratio})")
            from inference.inference_testing import analyze_embeddings
            rf_classifier = analyze_embeddings(df_embed, train_pos_seqs, neg_seqs,
                                               valid_pos_seqs, valid_neg_seqs,
                                               test_pos_seqs, test_neg_seqs,
                                               x=neg_to_pos_inference_ratio,
                                               nw=1, pw=20,
                                               threshold=0.145,
                                               n_estimators=100,
                                               to_balance=True,
                                               to_plot=True)[1]

            print(f"Testing out random forest classifier - Onehot Embedding (Taking neg to pos ratio: {neg_to_pos_inference_ratio})")
            from inference.inference_testing import analyze_embeddings
            analyze_embeddings(df_embed_onehot, train_pos_seqs, neg_seqs,
                               valid_pos_seqs, valid_neg_seqs,
                               test_pos_seqs, test_neg_seqs,
                               x=neg_to_pos_inference_ratio,
                               nw=1, pw=20,
                               threshold=0.145,
                               n_estimators=100,
                               to_balance=True,
                               to_plot=True)

            if INFERENCE_TO_RF_PLOT_DIST_PER_PATIENT:
                from inference.inference_testing import plot_output_distributions_per_patient_random_forest
                plot_output_distributions_per_patient_random_forest(untrained_model, rf_classifier, test_patient_inds, valid_patient_inds,
                                                                    unique_patient_ids,
                                                                    test_masks, valid_masks, positive_seqs, df_bld,
                                                                    df_hlt, model_type, args, device=device)

        # Display distribution on other dataset (article is T1D)
        if INFERENCE_TO_DISPLAY_OTHER_DATASET_DISTS:
            np.random.seed(42)
            other_dataset_type = 'article' if dataset_type == 'ms' else 'ms'
            print(f"Plotting the output distributions per patient on {dataset_type} and {other_dataset_type}")
            from inference.plot_handler import kde_normalizer
            from inference.inference_testing import display_distributions_on_different_sets
            # TODO: There is some problem her with loading the positive sequences from the other dataset, figure this out.
            other_dataset_loader = DatasetLoader(dataset_type=other_dataset_type, unique_patient_ids=unique_patient_ids,
                                                 k_fold=k_fold, dist_loss_type=dist_loss_type)
            df_bld_other, df_hlt_other = other_dataset_loader.get_dfs()
            display_distributions_on_different_sets(trained_model, dataset_type, other_dataset_type,
                                                    unique_patient_ids, test_patient_inds, valid_patient_inds,
                                                    df_hlt, df_bld, df_bld_other, df_hlt_other, kde_normalizer, device)

        if INFERENCE_TO_PLOT_EMBEDDING_MAPPINGS:
            np.random.seed(42)
            from inference.plot_handler import plot_embedding_mappings
            df_bld_val_test = df_bld[df_bld['patient_id'].isin(np.concatenate([valid_patient_ids, test_patient_ids]))]
            model_config = f"{dataset_type}_" + get_model_config_str(args)
            plot_embedding_mappings(trained_model, df_bld_val_test, valid_pos_seqs, test_pos_seqs, valid_patient_ids, test_patient_ids, model_config, device=device)

        # Inference classification model
        if INFERENCE_TO_CLASSIFICATION_MODEL:
            np.random.seed(42)
            if k_fold == 0 and combine_classification:
                from inference.inference_classification import inference_classification_model_combined
                def get_data_loader_wrapper(k_fold_index):
                    return get_dataset_loader(dataset_type, k_fold=k_fold_index, to_k_fold=True,
                                              dist_loss_type=dist_loss_type, neg_partition=neg_partition,
                                              use_similar_negatives=use_similar_negatives, neg_pos_ratio=neg_pos_ratio,
                                              filter_num_of_patients=filter_num_of_patients, filter_num_of_healthy=filter_num_of_healthy,
                                              ratio=ratio,
                                              filter_to_inflate=filter_to_inflate, remove_seqs_by_len=remove_seqs_by_len,
                                              top_percent=top_percent, top_n_seqs=top_n_seqs, extra_filter=extra_filter,
                                              use_nneighbors_loss=use_nneighbors_loss, loss_version=loss_version, verbose=True)
                inference_classification_model_combined(trained_model, args, to_ensemble, get_data_loader_wrapper, device)
            else:
                if classification_v2:
                    to_v2_tmp = False
                    if to_v2_tmp:
                        from inference.inference_classification import inference_classification_model_version2_tmp
                        model_non_trained = CVCClassifierModel(batch_size=batch_size, ch_dropout=ch_dropout, cvc_layers_to_train=cvc_layers_to_train, freeze_embed_model=freeze_embed_model, lora=lora, ch_type=ch_type, device=device)
                        inner_fold, components = 1, 2
                        inference_classification_model_version2_tmp(trained_model, args, df_bld, df_hlt,
                                                                test_patient_ids, valid_patient_ids,
                                                                valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs,
                                                                aaseq_to_ratio, to_ensemble, model_non_trained, device,
                                                                k_fold_disease=inner_fold, chosen_components=components)
                    from inference.inference_classification import inference_classification_model_version2
                    model_non_trained = CVCClassifierModel(batch_size=batch_size, ch_dropout=ch_dropout, cvc_layers_to_train=cvc_layers_to_train, freeze_embed_model=freeze_embed_model, lora=lora, ch_type=ch_type, device=device)
                    # for inner_fold, components in product([1, 2, 3], [2, 3]):
                    for inner_fold, components in product([1], [2]):
                        inference_classification_model_version2(trained_model, args, df_bld, df_hlt,
                                                                test_patient_ids, valid_patient_ids,
                                                                valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs,
                                                                aaseq_to_ratio, to_ensemble, model_non_trained, device,
                                                                k_fold_disease=inner_fold, chosen_components=components, dont_cache_inference=dont_cache_inference)
                else:
                    from inference.inference_classification import inference_classification_model
                    inference_classification_model(trained_model, args, df_bld, df_hlt,
                                                   test_patient_inds, valid_patient_inds, unique_patient_ids,
                                                   valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs,
                                                   aaseq_to_ratio, to_ensemble, device)

        if INFERENCE_RESHEF and reshef_inference:
            np.random.seed(42)
            from inference.reshef_inference import reshef_inference
            to_save_train_data = False if reshef_filter_train else True
            reshef_inference(train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs, reshef_negative_part, args, to_save_train_data=to_save_train_data)
