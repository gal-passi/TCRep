import warnings
warnings.simplefilter("ignore", category=FutureWarning)
from Curation import Study
import pandas as pd
import numpy as np
import torch
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
from sklearn.metrics import accuracy_score
from itertools import combinations, chain
import multiprocessing as mp
from collections import Counter


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
TO_DISPLAY_RESULTS = True
TO_DISPLAY_RESULTS_PLOT_TSNE = False
TO_DISPLAY_NUMBER_OF_COMMON_SEQUENCES = False
TO_LOAD_FULL_SYNAPSE_DATA = False


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


def get_valid_seqs_original(df, df_ind, name_opt, study_name):
    if df_ind <= 1:
        valid_seqs = df.groupby('AASeq')['patient_id'].nunique()
        valid_seqs = valid_seqs[valid_seqs >= 2]
        valid_seqs = set(valid_seqs.index)
    else:
        valid_seqs = set(df['AASeq'].unique())

    base_path = os.path.join(VALID_SEQ_CACHE, study_name)
    os.makedirs(base_path, exist_ok=True)
    cache_pkl = os.path.join(base_path, f"valid_sequences_id_{df_ind}{name_opt}.pkl")
    with open(cache_pkl, 'wb') as f:
        pickle.dump(valid_seqs, f)
    return valid_seqs


def embed_esmc(seqs, batch_size=512, to_mean=True, **kwargs):
    # set model_type from kwargs
    model_type = kwargs.get('model_type', 'esmc')
    # load model according to model type
    if model_type == 'esmc':
        from esm.models.esmc import ESMC
        model = ESMC.from_pretrained("esmc_300m")
    else:
        model = load_fine_tuned_esmc("cache/esm_c_checkpoints")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    # Convert seqs to numpy array
    seqs_np = np.array(seqs)
    # List to accumulate final embeddings
    all_embeds = []
    # Process in batches
    # for start_idx in range(0, len(seqs), batch_size):  # add tqdm and specify the total
    for start_idx in tqdm(range(0, len(seqs), batch_size), total=math.ceil(len(seqs) / batch_size)):
        end_idx = min(start_idx + batch_size, len(seqs))
        batch_seqs = seqs_np[start_idx:end_idx]
        with torch.no_grad():
            if model_type == 'esmc':
                # Tokenize the batch sequences
                tokenized = model._tokenize(batch_seqs)
                # Get embeddings for the batch
                embeds = model(tokenized).embeddings.to(torch.float32).cpu()
            else:
                # Tokenize the batch sequences
                tokenized = model.esmc_model._tokenize(batch_seqs)
                # Get embeddings for the batch
                embeds = model(sequences=tokenized).to(torch.float32).cpu()
        # Clean the start and end of each sequence in the embedding
        batch_embeds = [embed[1:len(seq) - 1] for embed, seq in zip(embeds, batch_seqs)]
        # Apply mean if needed
        if to_mean:
            batch_embeds = [embeds.mean(dim=0) for embeds in batch_embeds]
        else:
            batch_embeds = [embeds for embeds in batch_embeds]
        # Accumulate the batch embeddings
        all_embeds.extend(batch_embeds)
    return all_embeds


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


def get_cached_embeddings(sequences, study_name, name='', cache_dir="cache/esm_c/", embed_fn=None, **kwargs):
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
    embeddings = embed_fn(sequences, **kwargs)

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
    cache_pkl = os.path.join(base_path, f"valid_sequences_id_{df_ind}{name_opt}.pkl")
    if os.path.exists(cache_pkl):
        with open(cache_pkl, 'rb') as f:
            valid_sequences = pickle.load(f)
            return valid_sequences
        # return df[df['AASeq'].isin(valid_sequences)]['AASeq'].unique()
    else:
        return None


def calculate_valid_seqs_with_patient_id(df, df_ind, name_opt, study_name):
    # Getting from cache if it was already calculated
    base_path = os.path.join(VALID_SEQ_CACHE, study_name)
    os.makedirs(base_path, exist_ok=True)
    cache_pkl = os.path.join(base_path, f"valid_sequences_id_{df_ind}{name_opt}.pkl")
    if os.path.exists(cache_pkl):
        with open(cache_pkl, 'rb') as f:
            return pickle.load(f)

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
            # Select seqs2 that are in the range [length-1, length+1]
            min_len, max_len = length - 1, length + 1
            mask = (seqs2_lens >= min_len) & (seqs2_lens <= max_len)
            group2 = seqs2_by_len[mask]

            if len(group2) > 0:
                # Compute pairwise identity matrix
                pwc_mat = pairwise_scores(group1, group2, score=levenshtein_dist)  # OR score=seq_identity
                sim_inxs = np.where(pwc_mat >= 0.9)

                # Add matching sequences with patient_id to valid set
                for x, y in zip(sim_inxs[0], sim_inxs[1]):
                    valid_sequences.add((group1[x], pid1, pid2))  # Tuple (seq, originating pid x2)
                    valid_sequences.add((group2[y], pid1, pid2))  # Tuple (seq, originating pid x2)

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


def get_patient_ids_masks(df, sequences, meta_data, seq_type='syn'):
    unique_patient_ids = sorted(df["patient_id"].unique())
    masks = []
    if seq_type == 'syn':
        get_seq_data = lambda x: [n for n in meta_data if n[0] == x]
        for seq in sequences:
            # Getting sequence data
            seq_data = get_seq_data(seq)
            # Getting all patients that have this
            seq_patients = set(chain.from_iterable([data[1:] for data in seq_data]))

            current = np.zeros(len(unique_patient_ids))
            for i, patient_id in enumerate(unique_patient_ids):
                if patient_id in seq_patients:
                    current[i] = 1

            masks.append(current)
        masks = np.stack(masks, axis=1)
    else:
        for patient in unique_patient_ids:
            # Get sequences that belong to the current patient
            patient_seqs = set(df.loc[df["patient_id"] == patient, "AASeq"])

            # Create a mask for sequences
            mask = np.array([1 if seq in patient_seqs else 0 for seq in sequences])
            masks.append(mask)

        # Convert to ndarray
        masks = np.array(masks)  # Shape: (num_unique_patients, len(sequences))
    return masks


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


def temp_helper_function_common_aaseq_analysis(df, lev_dist_accept, only_valid=False):
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
        for length, group1 in tqdm(seqs1_by_len.items(), total=len(seqs1_by_len)):
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
            masks = temp_helper_function_common_aaseq_analysis(temp_df, lev_dist_accept)
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
            'num_common': num_common,
            'percent_of_total': percent_of_total,
            'percent_of_min': percent_of_min,
            'percent_of_max': percent_of_max
        })

    # Calculate means
    mean_results = {
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


# This function loads the synapse dataframe of Mal-ID of only TCR and healthy samples for sure.
def get_full_healthy_synapse_mal_id_dataframe():
    # Defining constants
    synapse_db_folder = "db/synapse_Mal_ID"
    synapse_metadata_file = os.path.join(synapse_db_folder, "metadata.tsv")

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
        if 'Healthy' in condition:
            data = pd.read_csv(datafile_path, sep='\t', compression='bz2')
            data = data.loc[:, ['cdr3_seq_aa_q', 'participant_label', 'specimen_tissue']]
            data['cdr3_seq_aa_q'] = data['cdr3_seq_aa_q'].str.replace(' ', '')
            healthy_samples.append(data)

    # need to make a df with: AASeq, patient_id, tissue, cell_type
    df = pd.concat(healthy_samples, ignore_index=True)
    df = df.dropna(subset=['cdr3_seq_aa_q'])
    # rename columns
    df = df.rename(columns={'cdr3_seq_aa_q': 'AASeq', 'participant_label': 'patient_id', 'specimen_tissue': 'tissue'})

    # plot_length_histogram(df["AASeq"], labels=["Healthy"], title_text="Healthy Sequences")
    return df

# TODO: VAE TRAINING STARTS HERE!.......................................................

class Single_Seq_processing:
    def __init__(self, MSA_location, theta=0.2, use_weights=False, weights_location=None, alphabet="ACDEFGHIKLMNPQRSTVWY"):
        self.MSA_location = MSA_location
        self.theta = theta
        self.use_weights = use_weights
        self.weights_location = weights_location
        self.alphabet = alphabet
        self.alphabet_size = len(self.alphabet)
        self.Neff = 1

        from other_models.eve.utils import data_utils
        data = data_utils.MSA_processing(
            MSA_location=MSA_location,
            theta=theta,
            use_weights=True,
            weights_location=weights_location
        )
        self.one_hot_encoding_msa = data.one_hot_encoding
        self.weights_msa = data.weights
        self.seq_len_msa = data.seq_len

        # load my sequences this time
        sequences = self.load_valid_sequences()[::50]
        self.one_hot_encoding = self.one_hot_encode(sequences)
        self.weights = np.ones(len(sequences), dtype=np.float64)
        self.seq_len = self.one_hot_encoding.shape[1]
        self.num_sequences = self.one_hot_encoding.shape[0]

        assert self.one_hot_encoding.shape[0] == self.weights.shape[0], "One-hot encoding shape mismatch WITH WEIGHTS!"
        assert self.one_hot_encoding.shape[0] == self.num_sequences, "One-hot encoding shape mismatch WITH NUM SEQUENCES!"
        assert self.one_hot_encoding.shape[1] == self.seq_len, "One-hot encoding shape mismatch WITH SEQ LEN!"
        assert self.one_hot_encoding.shape[2] == self.alphabet_size, "One-hot encoding shape mismatch WITH ALPHABET SIZE!"

    @staticmethod
    def one_hot_encode(sequences, alphabet="ACDEFGHIKLMNPQRSTVWY", max_length=None):
        """
        One-hot encode an array of protein sequences.

        Args:
            sequences (numpy.ndarray): Array of string sequences
            alphabet (str): Alphabet to use for one-hot encoding, default is 20 standard amino acids
            max_length (int, optional): Maximum sequence length. If None, uses the length of the longest sequence

        Returns:
            torch.Tensor: One-hot encoded sequences with shape (n_sequences, max_length, len(alphabet))
        """
        # Create mapping from amino acid to position
        aa_to_idx = {aa: idx for idx, aa in enumerate(alphabet)}

        # Determine max_length if not provided
        if max_length is None:
            max_length = max(len(seq) for seq in sequences)

        n_sequences = len(sequences)
        alphabet_size = len(alphabet)

        # Pre-allocate tensor
        one_hot = torch.zeros((n_sequences, max_length, alphabet_size), dtype=torch.float32)

        # Process each sequence
        for i, seq in tqdm(enumerate(sequences), total=n_sequences, desc="One-hot encoding"):
            seq_len = len(seq)

            # Calculate padding for centering
            pad_left = (max_length - seq_len) // 2

            # For each position in the sequence
            for j, aa in enumerate(seq):
                if aa in aa_to_idx:  # Handle case where amino acid is not in alphabet
                    # Calculate the position in the padded sequence
                    pos = pad_left + j

                    # Only set value if position is within bounds
                    if 0 <= pos < max_length:
                        one_hot[i, pos, aa_to_idx[aa]] = 1.0

        return one_hot

    @staticmethod
    def load_valid_sequences():
        df = get_all_usable_disease_data(disease='Multiple sclerosis')
        df_h = get_all_usable_healthy_data()
        valid_seqs_disease = calculate_valid_near_sequences(df, save_name='disease', lev_dist_accept=1, num_of_patients=3)
        # valid_seqs_healthy = calculate_valid_near_sequences(df_h, save_name='healthy', lev_dist_accept=1, num_of_patients=3)
        valid_seqs_healthy = find_all_common_sequences(df_h, num_of_patients=2)  # TODO: Added for now to speed up the process!

        positive_seqs = set([x[0] for x in valid_seqs_disease])
        negative_seqs = set([x[0] for x in valid_seqs_healthy])
        # add negative and positives to a single list
        all_seqs = list(positive_seqs) + list(negative_seqs)
        # sequences_d = df['AASeq'].unique()  # [::35]  # TODO: Removing some sequences for faster embedding
        # sequences_h = df_h['AASeq'].unique()
        # sequences = np.unique(np.concatenate([sequences_d, sequences_h]))
        return np.array(all_seqs)


def train_vae_eve_model():
    import sys
    import json
    from other_models.eve.utils import data_utils
    from other_models.eve.EVE.VAE_model import VAE_model

    # Define the base path
    base_path = r'/cs/labs/dina/amir_2000/TCRep/other_models/eve'

    # Default values
    MSA_data_folder = base_path + '/data/MSA'
    MSA_list = base_path + '/data/mappings/example_mapping.csv'
    protein_index = 0
    MSA_weights_location = base_path + '/data/weights'
    theta_reweighting = None  # Default: None
    VAE_checkpoint_location = base_path + '/results/VAE_parameters'
    model_name_suffix = 'Jan1_PTEN_example'
    model_parameters_location = base_path + '/EVE/default_model_params.json'
    training_logs_location = base_path + '/logs'
    seed = 42

    # Load mapping file and extract protein data
    mapping_file = pd.read_csv(MSA_list)
    protein_name = mapping_file['protein_name'][protein_index]
    msa_location = MSA_data_folder + os.sep + mapping_file['msa_location'][protein_index]
    print("Protein name: " + str(protein_name))
    print("MSA file: " + str(msa_location))

    # Determine theta value (if not provided, default to 0.2)
    if theta_reweighting is not None:
        theta = theta_reweighting
    else:
        try:
            theta = float(mapping_file['theta'][protein_index])
        except:
            theta = 0.2
    print("Theta MSA re-weighting: " + str(theta))

    # Process MSA data
    # data = data_utils.MSA_processing(
    #     MSA_location=msa_location,
    #     theta=theta,
    #     use_weights=True,
    #     weights_location=MSA_weights_location + os.sep + protein_name + '_theta_' + str(theta) + '.npy'
    # )
    data = Single_Seq_processing(
        MSA_location=msa_location,
        theta=theta,
        use_weights=True,
        weights_location=MSA_weights_location + os.sep + protein_name + '_theta_' + str(theta) + '.npy'
    )

    # Construct model name
    model_name = protein_name + "_" + model_name_suffix
    print("Model name: " + str(model_name))

    # Load model parameters
    model_params = json.load(open(model_parameters_location))

    # Initialize model
    model = VAE_model(
        model_name=model_name,
        data=data,
        encoder_parameters=model_params["encoder_parameters"],
        decoder_parameters=model_params["decoder_parameters"],
        random_seed=seed
    )
    model = model.to(model.device)

    # Update training parameters with checkpoint and log locations
    model_params["training_parameters"]['training_logs_location'] = training_logs_location
    model_params["training_parameters"]['model_checkpoint_location'] = VAE_checkpoint_location

    # TODO: Added the next params myself
    model_params["training_parameters"]['log_training_info'] = True
    model_params["training_parameters"]['num_training_steps'] = 40000

    # Train the model
    print("Starting to train model: " + model_name)
    model.train_model(data=data, training_parameters=model_params["training_parameters"])

    # Save the model
    print("Saving model: " + model_name)
    model.save(
        model_checkpoint=model_params["training_parameters"][
                             'model_checkpoint_location'] + os.sep + model_name + "_final",
        encoder_parameters=model_params["encoder_parameters"],
        decoder_parameters=model_params["decoder_parameters"],
        training_parameters=model_params["training_parameters"]
    )

# TODO: VAE TRAINING ENDS HERE!.......................................................

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


def calculate_valid_near_sequences(df, save_name, lev_dist_accept=1, num_of_patients=3, all_common_seqs=None):
    save_folder = "cache/valid_sequences/multiple_sclerosis"
    save_file = os.path.join(save_folder, f"{save_name}_valid_seqs_dist_{lev_dist_accept}.pkl")
    if not os.path.exists(save_file):
        if all_common_seqs is None:
            all_common_seqs = find_all_common_sequences(df, num_of_patients=num_of_patients)
        df_new = df.copy()
        df_new['patient_id'] = df_new['AASeq'].apply(lambda x: 'valid' if x in all_common_seqs else 'all')
        valid_df = df_new[df_new['patient_id'] == 'valid'].drop_duplicates(subset='AASeq')
        all_df = df_new[df_new['patient_id'] == 'all'].drop_duplicates(subset='AASeq')
        df_combined = pd.concat([valid_df, all_df])
        df_combined = df_combined.reset_index(drop=True)
        valid_seqs = temp_helper_function_common_aaseq_analysis(df_combined, lev_dist_accept, only_valid=True)
        print(f"Number of valid sequences: {len(valid_seqs)}")
        os.makedirs(save_folder, exist_ok=True)
        with open(save_file, "wb") as f:
            pickle.dump(valid_seqs, f)
    else:
        with open(save_file, "rb") as f:
            valid_seqs = pickle.load(f)
    return valid_seqs


# TODO: ESM-C FINE-TUNING STARTS HERE!.......................................................
from esm.sdk.api import ESMProtein, SamplingConfig, LogitsConfig

def print_trainable_parameters(model):
    """
  printing the number of trainable paramters in the model
  """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")


class MockConfig:
    def __init__(self, model_type="esm", tie_word_embeddings=False):
        self.model_type = model_type

    def get(self, key, default=None):
        return getattr(self, key, default)


class CustomModelWrapperesm(torch.nn.Module):
    def __init__(self, esmc_model):
        super(CustomModelWrapperesm, self).__init__()
        self.esmc_model = esmc_model
        self.config = MockConfig(model_type="esmc")

    def forward(self, sequences, **kwargs):
        return self.esmc_model(sequences).embeddings

        # sequences = [ESMProtein(sequence=seq).sequence for seq in input]
        # max_len = max(len(seq) for seq in sequences)
        # masks = torch.zeros((len(sequences), max_len))
        # for i, seq in enumerate(sequences):
        #     masks[i, :len(seq)] = 1
        input_ids = self.esmc_model._tokenize(sequences)#, attention_mask=masks)
        output = self.esmc_model(input_ids).embeddings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            self.esmc_model._detokenize(input_ids)
        # self.esmc_model._detokenize(sequences)
        return output

    def prepare_inputs_for_generation(self, inputs, **kwargs):
        # Dummy implementation, does nothing
        return inputs

    def __getattr__(self, name):
        # Ensure attributes in this wrapper class are handled first
        try:
            return super().__getattr__(name)
        except AttributeError:
            # Delegate to the wrapped model
            if hasattr(self.esmc_model, name):
                return getattr(self.esmc_model, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


def fine_tune_esmc_tanya():
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
    from peft import LoraConfig, get_peft_model, TaskType
    import inspect

    device = "cuda" if torch.cuda.is_available() else "cpu"

    peft_config_esmc = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias='none',
        layers_to_transform=[29, 28, 27, 26, 25, 24, 23, 22, 21],
        task_type=TaskType.CAUSAL_LM,
        target_modules=['attn.out_proj']
    )

    clientc = ESMC.from_pretrained("esmc_300m")  # .to("cuda") # or "cpu"
    # for name, module in clientc.named_modules():
    #     print(name)
    #     for sub in module.named_modules():
    #         print(sub)
    # print(inspect.signature(clientc._tokenize))
    clientc = CustomModelWrapperesm(clientc)
    clientc.config = MockConfig(model_type="esmc")

    peft_model = get_peft_model(clientc, peft_config_esmc)
    # print_trainable_parameters(clientc)

    # print(clientc.config.model_type)
    # if torch.cuda.is_available():
    #     clientc = clientc.to("cuda")
    # out = clientc(["AAAAA", "GG"])  # , "." * 10])
    peft_model = peft_model.to(device)
    # out = peft_model(["AAAAA", "GG"])
    out = peft_model(sequences=["AAAAA", "GG"])

    # save the embeddings
    # torch.save(out, "ESMC_retrained.pt")
    print(f"Output shape: {out.shape}")


def fine_tune_esmc(sequences, batch_size=8, num_epochs=5, learning_rate=5e-5, checkpoint_dir="checkpoints",
                   save_every=1, resume_from=None):
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
    from peft import LoraConfig, get_peft_model, TaskType
    import inspect
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    import glob
    import re

    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Automatically find the most recent checkpoint if resume_from is None
    if resume_from is None:
        # Look for checkpoint files in the checkpoint directory
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))

        if checkpoint_files:
            # Extract epoch numbers from checkpoint filenames
            epoch_numbers = []
            for file_path in checkpoint_files:
                match = re.search(r'checkpoint_epoch_(\d+)\.pt', file_path)
                if match:
                    epoch_numbers.append((int(match.group(1)), file_path))

            # Find the most recent checkpoint
            if epoch_numbers:
                _, resume_from = max(epoch_numbers, key=lambda x: x[0])
                print(f"Auto-resuming from most recent checkpoint: {resume_from}")

        # Also check for final_model.pt
        final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
        if os.path.exists(final_model_path) and (not checkpoint_files or
                                                 os.path.getmtime(final_model_path) > os.path.getmtime(resume_from)):
            resume_from = final_model_path
            print(f"Auto-resuming from final model checkpoint: {resume_from}")

    # Define PEFT configuration
    peft_config_esmc = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias='none',
        layers_to_transform=[29, 28, 27, 26, 25, 24, 23, 22, 21],
        task_type=TaskType.CAUSAL_LM,
        target_modules=['attn.out_proj']
    )

    # Load pre-trained model
    clientc = ESMC.from_pretrained("esmc_300m")
    clientc = CustomModelWrapperesm(clientc)
    clientc.config = MockConfig(model_type="esmc")

    # Apply PEFT
    peft_model = get_peft_model(clientc, peft_config_esmc)
    peft_model = peft_model.to(device)

    # Create dataset class
    class ProteinDataset(Dataset):
        def __init__(self, sequences):
            self.sequences = sequences

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx]

    # Masking function similar to ESM approach
    def mask_tokens(sequence, mask_prob=0.15):
        # Tokenize sequence using the model's tokenizer
        tokens = peft_model._tokenize(sequence)
        tokens = tokens.to(device)

        # Create a clone to use as labels
        labels = tokens.clone()

        # Determine which tokens to mask
        probability_matrix = torch.full(tokens.shape, mask_prob)

        # Don't mask special tokens like BOS, EOS, etc.
        special_tokens_mask = [
            peft_model.tokenizer.convert_tokens_to_ids("<cls>"),
            peft_model.tokenizer.convert_tokens_to_ids("<eos>"),
            peft_model.tokenizer.convert_tokens_to_ids("<pad>")
        ]
        for special_token in special_tokens_mask:
            probability_matrix[tokens == special_token] = 0.0

        # Select tokens to mask
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Create input by replacing masked tokens
        inputs = tokens.clone()

        # In ESM-style masking, we:
        # - 80% of the time, replace with <mask>
        # - 10% of the time, replace with random amino acid
        # - 10% of the time, keep the original token

        # Get the mask token ID
        mask_token_id = peft_model.tokenizer.convert_tokens_to_ids("<mask>")

        # Indices to replace with mask token (80% of masked tokens)
        indices_mask = torch.bernoulli(torch.full(masked_indices.shape, 0.8)).bool() & masked_indices
        inputs[indices_mask] = mask_token_id

        # Indices to replace with random token (10% of masked tokens)
        indices_random = torch.bernoulli(torch.full(masked_indices.shape, 0.5)).bool() & masked_indices & ~indices_mask
        random_amino_acids = torch.randint(5, 25, indices_random.sum().shape,
                                           device=device)  # Amino acid tokens typically in this range
        inputs[indices_random] = random_amino_acids

        # The remaining 10% masked tokens are kept unchanged

        # For computing loss, we only consider masked tokens
        # Set non-masked tokens to -100 (ignored by CrossEntropyLoss)
        labels[~masked_indices] = -100

        return inputs, labels

    # Create dataset and dataloader
    dataset = ProteinDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(peft_model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # Variables to track best model and training state
    best_loss = float('inf')
    start_epoch = 0
    training_history = []

    # Load from checkpoint if specified or auto-detected
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        peft_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']
        training_history = checkpoint.get('training_history', [])

        print(f"Resuming from epoch {start_epoch} with best loss: {best_loss:.4f}")
    else:
        if resume_from is not None:
            print(f"Checkpoint file {resume_from} not found. Starting from scratch.")
        else:
            print("No checkpoints found. Starting from scratch.")

    # Check if we've already completed all epochs
    if start_epoch >= num_epochs:
        print(f"Training already completed (start_epoch={start_epoch}, num_epochs={num_epochs})")
        # Load best model if available
        best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=device)
            peft_model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model with loss: {checkpoint['loss']:.4f}")

        # Get embeddings and return
        peft_model.eval()
        with torch.no_grad():
            tokenized_seqs = peft_model.esmc_model._tokenize(sequences[:10])
            embeddings = peft_model(sequences=tokenized_seqs)

        return {
            "model": peft_model,
            "embeddings": embeddings,
            "training_history": training_history,
            "best_loss": best_loss,
            "checkpoint_dir": checkpoint_dir,
            "status": "already_completed"
        }

    # Training loop
    peft_model.train()
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            # Process batch of sequences
            batch_inputs = []
            batch_labels = []

            for seq in batch:
                inputs, labels = mask_tokens(seq)
                batch_inputs.append(inputs)
                batch_labels.append(labels)

            # Pad sequences to same length in batch
            max_len = max(len(inp) for inp in batch_inputs)
            pad_token_id = peft_model.tokenizer.convert_tokens_to_ids("<pad>")

            # Pad inputs and labels
            padded_inputs = torch.ones((len(batch), max_len), dtype=torch.long, device=device) * pad_token_id
            padded_labels = torch.ones((len(batch), max_len), dtype=torch.long, device=device) * -100

            for i, (inp, lab) in enumerate(zip(batch_inputs, batch_labels)):
                padded_inputs[i, :len(inp)] = inp[:, 1]  # TODO: Added the slicing part on my own, idk if its correct
                padded_labels[i, :len(lab)] = lab[:, 1]

            # Forward pass
            optimizer.zero_grad()
            outputs = peft_model.forward(sequences=padded_inputs)

            # Reshape for loss calculation
            logits = outputs.view(-1, outputs.size(-1))  # (batch_size * seq_len, vocab_size)
            labels_view = padded_labels.view(-1)  # (batch_size * seq_len)

            # Calculate loss
            loss = criterion(logits, labels_view)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update progress bar
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / (progress_bar.n + 1)})

        # Calculate average loss for this epoch
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_epoch_loss:.4f}")

        # Update learning rate based on validation loss
        scheduler.step(avg_epoch_loss)

        # Track training history
        training_history.append({
            'epoch': epoch + 1,
            'loss': avg_epoch_loss,
            'lr': optimizer.param_groups[0]['lr']
        })

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': peft_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
                'best_loss': best_loss,
                'training_history': training_history
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': peft_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_epoch_loss,
                'best_loss': best_loss,
                'training_history': training_history
            }, best_model_path)
            print(f"New best model saved with loss: {best_loss:.4f}")

    # Save the final model
    final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
    torch.save({
        'epoch': num_epochs - 1,  # -1 since epoch is 0-indexed
        'model_state_dict': peft_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': avg_epoch_loss,
        'best_loss': best_loss,
        'training_history': training_history
    }, final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Save the model in PEFT format for easier reloading
    # peft_model.save_pretrained(os.path.join(checkpoint_dir, "peft_model"))

    # Get embeddings from the fine-tuned model
    peft_model.eval()
    with torch.no_grad():
        tokenized_seqs = peft_model.esmc_model._tokenize(sequences[:10])
        embeddings = peft_model(input=tokenized_seqs)  # Get embeddings for first 10 sequences as an example

    print(f"Output embeddings shape: {embeddings.shape}")

    # Return useful objects
    return {
        "model": peft_model,
        "embeddings": embeddings,
        "training_history": training_history,
        "best_loss": best_loss,
        "final_loss": avg_epoch_loss,
        "checkpoint_dir": checkpoint_dir,
        "status": "completed"
    }


def load_fine_tuned_esmc(checkpoint_dir, resume_from=None):
    from esm.models.esmc import ESMC
    from peft import LoraConfig, get_peft_model, TaskType
    import glob
    import re

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Automatically find the most recent checkpoint if resume_from is None
    if resume_from is None:
        # Look for checkpoint files in the checkpoint directory
        checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pt"))

        if checkpoint_files:
            # Extract epoch numbers from checkpoint filenames
            epoch_numbers = []
            for file_path in checkpoint_files:
                match = re.search(r'checkpoint_epoch_(\d+)\.pt', file_path)
                if match:
                    epoch_numbers.append((int(match.group(1)), file_path))

            # Find the most recent checkpoint
            if epoch_numbers:
                _, resume_from = max(epoch_numbers, key=lambda x: x[0])
                print(f"Auto-resuming from most recent checkpoint: {resume_from}")

        # Also check for final_model.pt
        final_model_path = os.path.join(checkpoint_dir, "final_model.pt")
        if os.path.exists(final_model_path) and (not checkpoint_files or
                                                 os.path.getmtime(final_model_path) > os.path.getmtime(resume_from)):
            resume_from = final_model_path
            print(f"Auto-resuming from final model checkpoint: {resume_from}")

    # Define PEFT configuration
    peft_config_esmc = LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias='none',
        layers_to_transform=[29, 28, 27, 26, 25, 24, 23, 22, 21],
        task_type=TaskType.CAUSAL_LM,
        target_modules=['attn.out_proj']
    )

    # Load pre-trained model
    clientc = ESMC.from_pretrained("esmc_300m")
    clientc = CustomModelWrapperesm(clientc)
    clientc.config = MockConfig(model_type="esmc")

    # Apply PEFT
    peft_model = get_peft_model(clientc, peft_config_esmc)
    peft_model = peft_model.to(device)

    # Load from checkpoint if specified or auto-detected
    if resume_from is not None and os.path.exists(resume_from):
        print(f"Loading checkpoint from {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        peft_model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']

        print(f"Resuming from epoch {start_epoch} with best loss: {best_loss:.4f}")
    else:
        if resume_from is not None:
            print(f"Checkpoint file {resume_from} not found. Starting from scratch.")
        else:
            print("No checkpoints found. Starting from scratch.")

    # Set model to eval
    peft_model.eval()

    # Return the model
    return peft_model


# TODO: ESM-C FINE-TUNING ENDS HERE!.......................................................

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

if __name__ == '__main__':
    # fine-tune esm-c
    # Load Sequences
    # df = get_all_usable_disease_data(disease='Multiple sclerosis')
    # df_h = get_all_usable_healthy_data()
    # seqs_d = df["AASeq"].unique()
    # seqs_h = df_h["AASeq"].unique()
    # sequences = np.unique(np.concatenate([seqs_d, seqs_h]))
    # sequences = np.sort(sequences)
    # np.random.seed(42)
    # sequences = np.random.permutation(sequences)

    # Fine-tune Process
    # fine_tune_esmc(sequences, batch_size=512, num_epochs=20, learning_rate=5e-5,
    #                checkpoint_dir="cache/esm_c_checkpoints", save_every=1, resume_from=None)
    # model = load_fine_tuned_esmc("cache/esm_c_checkpoints")
    # out1 = model(sequences=model.esmc_model._tokenize(np.array(["AAAAAA", "AAA"])))
    # out2 = embed(np.array(["AAAAAA", "AAA"]), to_mean=False)
    # exit(0)

    # train_vae_eve_model()
    # exit(0)

    # get df_ind from program arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--df_ind', type=int, default=-1)
    parser.add_argument('--dist_option', type=int, default=1)
    parser.add_argument('--study_ind', type=int, default=0)
    parser.add_argument('--k_fold_type', type=int, default=1)
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
    np.random.seed(42)

    name_opt = '_opt1' if dist_option == 1 else ''

    # Load study
    # study = Study(STUDY_ID)
    disease = 'Multiple sclerosis'
    df = get_all_usable_disease_data(disease=disease)
    df_h = get_all_usable_healthy_data()

    # reading samples from the study
    # df_study = study._samples['usable']
    # df_study = study.read_sample(df_study)
    # df_study = df_study[df_study['condition'] == 'Ankylosing Spondylitis']  # TODO: This is hardcoded for now! It depends on the study!

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


    # TODO: This part loads the other healthy synapse data and compares it with our healthy data
    if TO_LOAD_FULL_SYNAPSE_DATA:
        # Loading the full synapse dataframe of Mal-ID (only TCR and healthy samples)
        df_synapse = get_full_healthy_synapse_mal_id_dataframe()
        data = df_synapse['AASeq'].unique()
        print("Synapse Data Statistics:")
        print(f"Number of Unique Sequences in Synapse: {len(data)}")
        print(f"Number of Unique Patients in Synapse: {len(df_synapse['patient_id'].unique())}")
        # check number of common sequence between synapse and our data
        common_seqs_synapse = set(data).intersection(set(df_h['AASeq']))
        print(f"Number of Common Sequences in Synapse and Healthy Data: {len(common_seqs_synapse)}")

    cell_type = ['DC8', 'CD4', 'ALL'][2]

    if cell_type != 'ALL':
        # reading blood samples
        df_bld = df[df['cell_type'] == cell_type]
        # reading healthy study:
        df_hlt = df_h[df_h['cell_type'] == cell_type]
    else:
        df_bld, df_hlt = df, df_h

    all_common_seqs = find_all_common_sequences(df, num_of_patients=3)
    valid_seqs_healthy = find_all_common_sequences(df_hlt, num_of_patients=3)
    all_common_seqs = all_common_seqs - valid_seqs_healthy
    # choosing valid samples according to their re-occurrence in different patients and a given distance
    valid_seqs_disease = calculate_valid_near_sequences(df_bld, save_name=f'disease_{cell_type}', lev_dist_accept=1, num_of_patients=3, all_common_seqs=all_common_seqs)
    # valid_seqs_healthy = calculate_valid_near_sequences(df_hlt, save_name='healthy', lev_dist_accept=1, num_of_patients=3)

    # get all neighbours of healthy sequences:
    valid_letters = set(''.join(valid_seqs_healthy))
    neighbors = generate_neighbors(valid_seqs_healthy, valid_letters)  # len is ~12 million!

    # filtering the sequences to positive and negative (disease and healthy respectively)
    positive_seqs = set([x[0] for x in valid_seqs_disease])
    negative_seqs = set(valid_seqs_healthy)
    positive_seqs = positive_seqs - neighbors

    # make list and sort
    positive_seqs = list(positive_seqs)
    positive_seqs.sort()
    negative_seqs = list(negative_seqs)
    negative_seqs.sort()
    # shuffle according to a certain seed
    np.random.seed(42)
    np.random.shuffle(positive_seqs)
    np.random.shuffle(negative_seqs)

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

    # Calculating embeddings (or loading if it is available)
    embed_bld = get_cached_embeddings(positive_seqs, disease, name=f'{disease}_{cell_type}_bld' + name_opt, embed_fn=embed_esmc, model_type="esmc")
    embed_hlt = get_cached_embeddings(negative_seqs, "healthy", name=f'{cell_type}_h' + name_opt, embed_fn=embed_esmc, model_type="esmc")
    # embed_bld = embed_esmc(positive_seqs, model_type="esmc")
    # embed_hlt = embed_esmc(negative_seqs, model_type="esmc")

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
