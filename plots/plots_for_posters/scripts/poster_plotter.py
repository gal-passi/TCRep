import os
import numpy as np
import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm
import re
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix


# DISCLAIMER:
# All data loaded in this script is from the trained model on the MS dataset.

# Note to self:
# Configurations used for this dataset is (for plots before plot_ablation_on_folds!):
# --model_type cvc --dataset_type ms_tcrdb2_no_healthy_ms --top_n_seqs 20 --epochs 20 --loss_type ce_entropy -scrit --cvc_layers_to_train 3 --reg_coef 0.3 --pos_weights 5.45 --learning_rate 0.0023 --embedding_lr 5.1e-05 --scheduler_type "ExponentialLR" --dropout 0.135 --ch_type v1 --neg_pos_ratio 10 --extra_ms_from_pregnant --extra_filter --sample_plots 1 --classification_v2 --optimizer_type "Adam" --k_fold 1 -nolog


BASE_PROJECT_PATH = "/cs/labs/dina/amir_2000/TCRep"
BASE_PATH = os.path.join(BASE_PROJECT_PATH, "plots", "plots_for_posters")
BASE_DATA_PATH = os.path.join(BASE_PATH, "data")
BASE_PLOTS_PATH = os.path.join(BASE_PATH, "plots for posters")


def plot_embeddings(legend_fontsize=14, legend_markersize=10):
    # Load the data
    data = np.load(os.path.join(BASE_DATA_PATH, "embedding_plots_data.npz"))
    pos_embeds = torch.tensor(data['pos_embeds'])
    neg_embeds = torch.tensor(data['neg_embeds'])
    pos_outputs = torch.tensor(data['pos_outputs'])
    neg_outputs = torch.tensor(data['neg_outputs'])

    confident_pos_indices = pos_outputs[:, 1] > 0.75
    confident_neg_indices = neg_outputs[:, 1] > 0.75
    pos_alphas = np.where(confident_pos_indices, 0.5, 0.035)
    neg_alphas = np.where(confident_neg_indices, 0.5, 0.035)

    custom_legend = [
        Line2D([0], [0], marker='o', color='w', label='MS Sequences',
               markerfacecolor='darkorange', markersize=legend_markersize),
        Line2D([0], [0], marker='o', color='w', label='Healthy Sequences',
               markerfacecolor='blue', markersize=legend_markersize),
    ]

    all_embeds = torch.cat([pos_embeds, neg_embeds], dim=0).cpu().numpy()
    labels = np.array([1] * len(pos_embeds) + [0] * len(neg_embeds))
    # Create the figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(6 * 3, 6), dpi=600)
    spine_thickness = 0.75
    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(all_embeds)
    axes[0].scatter(pca_result[labels == 0, 0], pca_result[labels == 0, 1], c='blue', label='Healthy Sequences',
                    alpha=neg_alphas)
    axes[0].scatter(pca_result[labels == 1, 0], pca_result[labels == 1, 1], c='darkorange', label='MS Sequences',
                    alpha=pos_alphas)
    axes[0].set_title("PCA of Embeddings")
    # axes[0].set_xlabel("Dim 1")
    # axes[0].set_ylabel("Dim 2")
    axes[0].legend(handles=custom_legend, loc='upper left', fontsize=legend_fontsize)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for spine in axes[0].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(spine_thickness)
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_result = tsne.fit_transform(all_embeds)
    axes[1].scatter(tsne_result[labels == 0, 0], tsne_result[labels == 0, 1], c='blue', label='Healthy Sequences',
                    alpha=neg_alphas)
    axes[1].scatter(tsne_result[labels == 1, 0], tsne_result[labels == 1, 1], c='darkorange', label='MS Sequences',
                    alpha=pos_alphas)
    axes[1].set_title("t-SNE of Embeddings")
    # axes[1].set_xlabel("Dim 1")
    # axes[1].set_ylabel("Dim 2")
    axes[1].legend(handles=custom_legend, loc='upper left', fontsize=legend_fontsize)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    for spine in axes[1].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(spine_thickness)
    # UMAP
    umap_model = umap.UMAP(n_components=2, random_state=42)
    umap_result = umap_model.fit_transform(all_embeds)
    axes[2].scatter(umap_result[labels == 0, 0], umap_result[labels == 0, 1], c='blue', label='Healthy Sequences',
                    alpha=neg_alphas)
    axes[2].scatter(umap_result[labels == 1, 0], umap_result[labels == 1, 1], c='darkorange', label='MS Sequences',
                    alpha=pos_alphas)
    axes[2].set_title("UMAP of Embeddings")
    # axes[2].set_xlabel("Dim 1")
    # axes[2].set_ylabel("Dim 2")
    axes[2].legend(handles=custom_legend, loc='upper left', fontsize=legend_fontsize)
    plt.suptitle(f"Trained Model - Embeddings Visualization", fontsize=16)
    axes[2].set_xticks([])
    axes[2].set_yticks([])
    for spine in axes[2].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(spine_thickness)
    plt.tight_layout()
    plt.show()
    return


def fill_area_after_threshold(ax, line, threshold, color, alpha=0.2):
    """
    Fill the area under a curve after a given threshold.

    Parameters:
    - ax: Matplotlib axis
    - line: The line object from the KDE plot
    - threshold: The x-value threshold after which to fill
    - color: Color for the fill
    - alpha: Transparency of the fill
    """
    x_data = line.get_xdata()
    y_data = line.get_ydata()

    # Find indices where x >= threshold
    mask = x_data >= threshold

    if np.any(mask):
        # Fill area under curve after threshold
        ax.fill_between(x_data[mask], -1, y_data[mask],
                        color=color, alpha=alpha)


def plot_average_dist_with_std(ax, distributions, color, label, bins=50, min_val=0, max_val=1):
    """
    Plot average distribution with standard deviation shading around KDE curve,
    with normalization applied at the end to make max value equal to 1.

    Parameters:
    - ax: Matplotlib axis
    - distributions: List of probability arrays
    - color: Color for the plot
    - label: Label for the legend
    - bins: Number of bins for histogram
    - min_val, max_val: Range for the distribution
    """
    # Create KDE plots for each distribution individually (with alpha=0 to make them invisible)
    # This allows us to extract the curve data for each one
    kde_lines = []
    x_values_list = []
    y_values_list = []

    for dist in distributions:
        if len(dist) > 0:  # Ensure we have data to plot
            kde = sns.kdeplot(dist, ax=ax, common_norm=True, alpha=0)
            line = kde.get_lines()[-1]  # Get the last line added to the plot
            x_values = line.get_xdata()
            y_values = line.get_ydata()

            x_values_list.append(x_values)
            y_values_list.append(y_values)

            # Remove the invisible line to keep the plot clean
            line.remove()

    # Create the visible KDE plot for the combined data
    kde = sns.kdeplot(np.concatenate(distributions), ax=ax, color=color, common_norm=True)
    line = kde.get_lines()[-1]  # Get the last line

    # Get the data from the main KDE curve
    main_x = line.get_xdata()
    main_y = line.get_ydata()

    # Calculate standard deviation
    std_y = np.zeros_like(main_y)

    # Ensure we have distributions to calculate std from
    if len(x_values_list) > 1:
        # Interpolate all individual KDE y-values to match the main x-values grid
        interpolated_y_values = []
        for x_vals, y_vals in zip(x_values_list, y_values_list):
            # Limit interpolation to the range of x_vals to avoid extrapolation
            mask = (main_x >= min(x_vals)) & (main_x <= max(x_vals))
            if np.any(mask):
                interp_y = np.interp(main_x[mask], x_vals, y_vals)
                temp_y = np.zeros_like(main_x)
                temp_y[mask] = interp_y
                interpolated_y_values.append(temp_y)

        # Only proceed if we have valid interpolated values
        if interpolated_y_values:
            # Calculate std at each x-point
            interpolated_y_array = np.array(interpolated_y_values)
            std_y = np.std(interpolated_y_array, axis=0)

    # NORMALIZE EVERYTHING AT THE END
    # Find the maximum value of the main curve for normalization
    max_value = np.max(main_y)
    if max_value > 0:
        # Normalize the main curve
        normalized_main_y = main_y / max_value
        line.set_ydata(normalized_main_y)

        # Normalize the standard deviation
        normalized_std_y = std_y / max_value

        # Add the normalized std shading around the main KDE line
        ax.fill_between(
            main_x,
            normalized_main_y - normalized_std_y,
            normalized_main_y + normalized_std_y,
            color=color,
            alpha=0.2
        )

    # Set the label after normalization
    line.set_label(label)

    return line


def plot_average_embeddings(gmm_linewidth=1.5, display_threshold_line=False):
    # Load the data
    with open(os.path.join(BASE_DATA_PATH, "average_distribution_plot_data.pkl"), 'rb') as f:
        data = pickle.load(f)
    healthy_dists = data['healthy_dists']
    disease_dists = data['disease_dists']

    # Add means with weights and stds as text labels here:
    healthy_means = [0.01233453, 0.74463677]  # Your two healthy means
    healthy_weights = [0.9109375, 0.0890625]  # Corresponding weights
    healthy_stds = [0.00474603, 0.00474603]  # Corresponding standard deviations

    disease_means = [0.01504089, 0.80275572]  # Your two disease means
    disease_weights = [0.86672419, 0.13327581]  # Corresponding weights
    disease_stds = [0.00638802, 0.00638802]  # Corresponding standard deviations

    # FIGUREs variables:
    xlim = (-0.15, 1.05)
    ylim = (-0.02, 1.2)
    dpi = 600
    threshold = 0.85  # Define the threshold

    # FIGURE 1: Average distributions with std
    fig1, ax1 = plt.subplots(figsize=(6, 6), dpi=dpi)
    for spine in ax1.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.75)
    # fig1.suptitle(f"Average Output Distributions for {model_type} Model", fontsize=16)

    # Plot average disease distribution with std
    disease_line = plot_average_dist_with_std(ax1, disease_dists, color="#FFA500", label="MS Distribution")
    # draw line at disease_means
    ax1.axvline(x=disease_means[0], color="#FFA500", linestyle='--', linewidth=gmm_linewidth, alpha=0.5, label="MS GMM μ=0.74")
    ax1.axvline(x=disease_means[1], color="#FFA500", linestyle='--', linewidth=gmm_linewidth, alpha=0.5)

    # Plot average healthy distribution with std
    healthy_line = plot_average_dist_with_std(ax1, healthy_dists, color="#7BC8F6", label="Healthy Distribution")
    # draw line at healthy_means
    ax1.axvline(x=healthy_means[0], color="#7BC8F6", linestyle='--', linewidth=gmm_linewidth, alpha=0.5, label="Healthy GMM μ=0.74")
    ax1.axvline(x=healthy_means[1], color="#7BC8F6", linestyle='--', linewidth=gmm_linewidth, alpha=0.5)

    # Fill area under curves after threshold
    fill_area_after_threshold(ax1, disease_line, threshold+0.00205, color="grey", alpha=0.2)
    fill_area_after_threshold(ax1, healthy_line, threshold, color="grey", alpha=0.2)

    if display_threshold_line:
        # Add threshold line
        ax1.plot([threshold, threshold], [-1, 0.2], color='gray', linestyle='--', linewidth=1.5, alpha=0.3,
                 label=f'Threshold = {threshold}')
        ax1.text(threshold + 0.01, 0.20, 'Confident\nPathogenic', rotation=0, va='bottom', ha='center', fontsize=7,
                 alpha=0.9, fontweight='normal', bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.2'))

    ax1.set_xlabel("Predicted Probability for Positive Class")
    ax1.set_ylabel("Density")
    ax1.set_xlim(xlim[0], xlim[1])  # Focus on the set range
    ax1.set_ylim(ylim[0], ylim[1])  # Focus on the set range
    ax1.set_xticks(np.arange(0.0, 1.1, 0.2))  # Ticks from 0.0 to 1.0 with step 0.1
    ax1.set_yticks(np.arange(0.0, 1.1, 0.2))  # Ticks from 0.0 to 1.0 with step 0.2
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # add text at the top of the plot for healthy means
    # txt_height = 1.23 + 0.04
    # ax1.text(healthy_means[0] - 0.03, txt_height, f'{healthy_weights[0]:.2f}',
    #          color="black", fontsize=8, ha='center', va='top', rotation=45)
    # ax1.text(healthy_means[1] - 0.03, txt_height, f'{healthy_weights[1]:.2f}',
    #          color="black", fontsize=8, ha='center', va='top', rotation=45)
    # # add text at the top of the plot for disease means
    # ax1.text(disease_means[0] + 0.03, txt_height, f'{disease_weights[0]:.2f}',
    #          color="black", fontsize=8, ha='center', va='top', rotation=45)
    # ax1.text(disease_means[1] + 0.03, txt_height, f'{disease_weights[1]:.2f}',
    #          color="black", fontsize=8, ha='center', va='top', rotation=45)    # if we move the tight_layout to this line, we need to change 1.23 -> 1.27
    # plt.legend(fontsize=14, framealpha=1.0)
    plt.tight_layout()
    plt.show()
    return


def calculations_for_poster():
    def compute_avg_sensitivity_specificity(conf_matrices, normalization_option=0):
        sensitivities = []
        specificities = []
        tps = []
        fps = []
        tns = []
        fns = []

        for cm in conf_matrices:
            tn, fp = cm[0]
            fn, tp = cm[1]
            if normalization_option == 1:
                tn+=52-tn-fp
                fn+=8-tp-fn

            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

            if normalization_option == 1:
                tps.append(tp / 8)
                fps.append(fp / 52)
                tns.append(tn / 52)
                fns.append(fn / 8)
            else:
                tps.append(tp / sum(cm[1]))
                fps.append(fp / sum(cm[0]))
                tns.append(tn / sum(cm[0]))
                fns.append(fn / sum(cm[1]))
            sensitivities.append(sensitivity)
            specificities.append(specificity)

        avg_sensitivity = np.nanmean(sensitivities)
        avg_specificity = np.nanmean(specificities)

        return avg_sensitivity, avg_specificity, np.mean(tps), np.mean(tns), np.mean(fps), np.mean(fns)

    gmm_75_percentile_cms = [
        [[34, 5],
         [2, 5]],
        [[36, 5],
         [1, 3]],
        [[31, 5],
         [0, 3]],
        [[35, 5],
         [0, 5]],
        [[35, 5],
         [2, 3]]
    ]
    gmm_90_percentile_cms = [
        [[38, 9],
         [1, 6]],
        [[38, 7],
         [1, 7]],
        [[38, 10],
         [0, 6]],
        [[38, 11],
         [0, 5]],
        [[38, 8],
         [2, 6]]
    ]
    svm_75_percentile_cms = [
        [[40, 0],
         [1, 4]],
        [[38, 2],
         [1, 4]],
        [[38, 2],
         [1, 4]],
        [[38, 2],
         [0, 5]],
        [[39, 1],
         [2, 3]]
    ]

    print("Calculating Average Sensitivity and Specificity for GMM and SVM models:")
    gmm_sens, gmm_spec, tp, tn, fp, fn = compute_avg_sensitivity_specificity(gmm_75_percentile_cms)
    print(f"GMM 75th Percentile - \nTP: {tp:.3f}, TN: {tn:.3f},\nFP: {fp:.3f}, FN: {fn:.3f}")
    gmm_90_sens, gmm_90_spec, tp, tn, fp, fn = compute_avg_sensitivity_specificity(gmm_90_percentile_cms)
    print(f"GMM 90th Percentile - \nTP: {tp:.3f}, TN: {tn:.3f},\nFP: {fp:.3f}, FN: {fn:.3f}")
    svm_sens, svm_spec, tp, tn, fp, fn = compute_avg_sensitivity_specificity(svm_75_percentile_cms)
    print(f"SVM 75th Percentile - \nTP: {tp:.3f}, TN: {tn:.3f},\nFP: {fp:.3f}, FN: {fn:.3f}")

    print("\nAverage Sensitivity and Specificity Across Folds:")
    print(f"GMM 75th Percentile:\n"
          f"\tsensitivity: {gmm_sens:.3f}, specificity: {gmm_spec:.3f}")
    print(f"GMM 90th Percentile:\n"
          f"\tsensitivity: {gmm_90_sens:.3f}, specificity: {gmm_90_spec:.3f}")
    print(f"SVM 75th Percentile:\n"
          f"\tsensitivity: {svm_sens:.3f}, specificity: {svm_spec:.3f}")


def plot_ablation_on_folds(n_folds=8):
    val_f1 = [0.16036, 0.24463, 0.25254, 0.24101, 0.102, 0.16456, 0.16056, 0.15261, 0.11184, 0.17993, 0.19084, 0.2027, 0.12983, 0.20327, 0.19976, 0.19702, 0.11933, 0.18205, 0.17235, 0.17512, 0.11732, 0.16997, 0.17016, 0.17387, 0.13861, 0.23182, 0.20886, 0.22035, 0.13027, 0.18538, 0.18526, 0.18966]
    val_tpr = [0.73674, 0.67287, 0.67409, 0.69312, 0.68308, 0.54638, 0.54279, 0.5787, 0.66152, 0.60961, 0.57844, 0.57307, 0.70011, 0.53348, 0.541, 0.51593, 0.64883, 0.55895, 0.58, 0.58611, 0.67459, 0.51458, 0.49389, 0.49012, 0.62191, 0.59371, 0.61731, 0.61175, 0.67703, 0.62616, 0.6087, 0.60675]
    val_fnr = [1 - x for x in val_tpr]  # False Negative Rate
    val_tnr = [0.85941, 0.87868, 0.88388, 0.87138, 0.86852, 0.91396, 0.91186, 0.89857, 0.8742, 0.89346, 0.90755, 0.91584, 0.85513, 0.91243, 0.90866, 0.91229, 0.88305, 0.9084, 0.89704, 0.89789, 0.86821, 0.91411, 0.91845, 0.92154, 0.89829, 0.91784, 0.90002, 0.90824, 0.86027, 0.87712, 0.88111, 0.88521]
    val_fpr = [1 - x for x in val_tnr]  # False Positive Rate


    # Create ablation figure
    # Dictionary to select which metric to plot
    metrics = {
        "F1": val_f1,
        "TPR": val_tpr,
        "FNR": val_fnr,
        "TNR": val_tnr,
        "FPR": val_fpr,
    }

    # Model labels
    model_labels = [
        "Normal",
        "Inflation (by edit distance)",
        "Special Entropy Term",
        "Confidence Neighbors Term"
    ]

    # Reshape helper
    def reshape_values(values):
        return np.array(values).reshape(n_folds, 4)  # n_folds × 4 models

    # Plot all metrics in subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))  # 2 rows, 3 cols (last will be empty)
    axes = axes.flatten()

    for idx, (metric_name, metric_values) in enumerate(metrics.items()):
        ax = axes[idx]
        values = reshape_values(metric_values)

        # Boxplots
        ax.boxplot(values, positions=np.arange(1, 5), widths=0.5)

        # Scatter per fold
        for fold in range(n_folds):
            ax.scatter(np.arange(1, 5), values[fold, :], alpha=0.7, label=f"Fold {fold + 1}" if idx == 0 else None)

        ax.set_xticks([1, 2, 3, 4])
        ax.set_xticklabels(model_labels, rotation=20, ha="right")
        ax.set_title(metric_name)
        ax.set_ylabel(metric_name)

    # Add legend only once
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    # Remove the last empty subplot
    fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()


def load_pickle(path):
    """Load variables from a pickle file into a dictionary."""
    with open(path, "rb") as f:
        return pickle.load(f)


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


def generate_full_neighbors(seqs, valid_letters=None, include_seqs=True):
    if valid_letters is None:
        valid_letters = set(''.join(seqs))
    all_neighbors = set()
    length_groups = {}
    for seq in seqs:
        length_groups.setdefault(len(seq), set()).add(seq)
    # Process each length group separately
    for seq_len, seq_group in length_groups.items():
        # Generate neighbors for this group
        neighbors = generate_neighbors(seq_group, valid_letters)
        all_neighbors.update(neighbors)
    if include_seqs:
        all_neighbors.update(seqs) # Include original sequences in the neighbors
    return np.array(list(all_neighbors))


def make_patient_df(patient_train_seqs, patient_train_data, bad_seqs, close_bad_seqs):
    """
    Create DataFrame with patient sequences, their data, and flags for bad/close_bad sets.
    """
    df = pd.DataFrame({
        "sequence": patient_train_seqs,
        "data": patient_train_data
    })
    df["is_bad"] = df["sequence"].isin(bad_seqs)
    df["is_close_bad"] = df["sequence"].isin(close_bad_seqs)
    return df


def filter_sequences(df, low, high):
    """
    Filter sequences based on the following rules:
    - If data < low or data > high -> remove if in bad_seqs
    - If low <= data <= high -> remove if in close_bad_seqs
    """
    mask_remove = (
        ((df["data"] < low) | (df["data"] > high)) & df["is_bad"]
    ) | (
        (df["data"] >= low) & (df["data"] <= high) & df["is_close_bad"]
    )
    return df[~mask_remove]  # keep only non-removed


def get_filtered_data(train_seqs, train_data, bad_seqs, close_bad_seqs, low, high):
    """
    Create a DataFrame, filter it according to the given bounds,
    and return only the 'data' column (floats) of the kept sequences.
    """
    df = make_patient_df(train_seqs, train_data, bad_seqs, close_bad_seqs)
    filtered_df = filter_sequences(df, low, high)
    return filtered_df["data"].values


def get_filtered_data_probs(test_seqs, test_probs, bad_seqs, close_bad_seqs, low_bound, high_bound):
    test_probs_filtered = []
    for i, test_seq in enumerate(test_seqs):
        test_seqs = test_seq[1]
        test_prob = test_probs[i]
        # Filter patient test probabilities based on bad and close bad sequences
        test_prob_filtered = get_filtered_data(test_seqs, test_prob, bad_seqs, close_bad_seqs, low_bound, high_bound)
        test_probs_filtered.append(test_prob_filtered)
    return test_probs_filtered


def plot_classification_ablation():
    chosen_components = 2
    covariance_type = 'full'
    reshef_inference_ablation_data = os.path.join(BASE_PROJECT_PATH, "cache", "reshef_inference", "ablation_data")

    # Step 1: list directory and extract model_config_strs
    print("Loading inference ablation data...")
    files = os.listdir(reshef_inference_ablation_data)
    model_files = [f for f in files if f.startswith("reshef_inference_ablation_data_") and f.endswith(".pkl") and "ce_entropy" in f]
    model_files = [f for f in model_files if 'fold-6_e' not in f and 'fold-7_e' not in f and 'fold-8_e' not in f]

    # Extract model_config_strs
    model_config_strs = [f[len("reshef_inference_ablation_data_"):-len(".pkl")] for f in model_files]

    all_data = {}
    for model_config_str in model_config_strs:
        model_dict = {}

        # Load ablation data (pickle)
        ablation_path = os.path.join(reshef_inference_ablation_data,
                                     f"reshef_inference_ablation_data_{model_config_str}.pkl")
        model_dict["ablation_data"] = load_pickle(ablation_path)

        # Find all sequence files for this model_config_str
        seq_files = [f for f in files if f.startswith("sequences") and model_config_str in f]

        sequences_data = {}
        for seq_file in seq_files:
            key_name = seq_file.replace(f"_{model_config_str}.pkl", "").replace("sequences_", "")
            path = os.path.join(reshef_inference_ablation_data, seq_file)
            sequences_data[key_name] = load_pickle(path)

        model_dict["sequences"] = sequences_data
        all_data[model_config_str] = model_dict
    print('Done.')

    # Get the set of hard to learn sequences
    print("Loading hard to learn sequences...")
    reshef_cache_folder = os.path.join(BASE_PROJECT_PATH, "cache", "reshef_inference")
    reshef_inference_data_path = os.path.join(reshef_cache_folder, "reshef_inference_low_confidence_neg_seqs_all_partitions.npy")
    if not os.path.exists(reshef_inference_data_path):
        print(f"Reshef inference data not found at {reshef_inference_data_path}.")
    bad_seqs_path = os.path.join(reshef_cache_folder, "filtered_bad_seqs_and_close_bad_seqs.pkl")
    if os.path.exists(bad_seqs_path):
        print(f"Loading bad sequences from {bad_seqs_path}...")
        with open(bad_seqs_path, 'rb') as f:
            bad_seqs_data = pickle.load(f)
        bad_seqs, close_bad_seqs = bad_seqs_data['bad_seqs'], bad_seqs_data['close_bad_seqs']
    else:
        print(f"Bad sequences file not found at {bad_seqs_path}. Generating from inference data...")
        # Generate bad sequences and their close neighbors
        bad_seqs = np.load(reshef_inference_data_path)
        bad_seqs, close_bad_seqs = set(bad_seqs), set(generate_full_neighbors(bad_seqs))

        # Concatenate the set of all sequences possible in the train, validation and test sets then keep only the intersection between them
        all_sequences = set()
        for model_config, data in all_data.items():
            for key, seqs in data['sequences'].items():
                for seq in seqs:
                    all_sequences.update(set(seq[1]))

        bad_seqs = set(bad_seqs) & all_sequences
        close_bad_seqs = set(close_bad_seqs) & all_sequences

        # save to pickle
        with open(bad_seqs_path, 'wb') as f:
            pickle.dump({'bad_seqs': bad_seqs, 'close_bad_seqs': close_bad_seqs}, f)
    print(f"Done.")

    # Step 2: Iterate over all model_config_strs and perform GMM classification
    print("Performing GMM classification ablation...")
    all_results = []
    low_bounds = [-1, 0, 0.05, 0.1, 0.2]
    high_bounds = [0.75, 0.85, 0.9, 1.0, -1]
    results_path = os.path.join(reshef_inference_ablation_data, "classification_ablation_results.pkl")
    # Load from pickle if it exists
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            all_results = pickle.load(f)
        # check validity of all_results
        if len(all_results) != len(low_bounds):
            all_results = []
        elif len(all_results[0]['accuracies']) != len(model_files):
            all_results = []
        if all_results:
            print("Loaded existing classification results from pickle.")
    if not all_results:  # recalculate if we did not find a proper save of the current results
        first_do_not_filter = True
        for low_bound, high_bound in zip(low_bounds, high_bounds[::-1]):
            all_classification_names = []
            all_cm_gmms = []
            all_accuracies = []
            for model_config, data in all_data.items():
                classification_names = []
                cm_gmms = []
                accuracies = []

                ablation_data = data['ablation_data']
                patient_train_data = ablation_data['patient_train_data']
                healthy_train_data = ablation_data['healthy_train_data']

                sequences_data = data['sequences']
                patient_train_seqs = np.concatenate([np.array(x[1]) for x in sequences_data['patient_train']])
                healthy_train_seqs = np.concatenate([np.array(x[1]) for x in sequences_data['healthy_all']][:52])

                # Filter patient and healthy train data based on bad and close bad sequences
                if not first_do_not_filter:
                    patient_train_data = get_filtered_data(patient_train_seqs, patient_train_data.reshape((-1,)), bad_seqs, close_bad_seqs, low_bound, high_bound)
                    healthy_train_data = get_filtered_data(healthy_train_seqs, healthy_train_data.reshape((-1,)), bad_seqs, close_bad_seqs, low_bound, high_bound)

                final_patient_gmm = GaussianMixture(n_components=chosen_components, random_state=42, covariance_type=covariance_type)
                final_patient_gmm.fit(patient_train_data.reshape((-1, 1)))
                final_healthy_gmm = GaussianMixture(n_components=chosen_components, random_state=42, covariance_type=covariance_type)
                final_healthy_gmm.fit(healthy_train_data.reshape((-1, 1)))

                # Extract patient and healthy test probabilities
                patient_test_probs = ablation_data['patient_test_probs']
                healthy_test_probs = ablation_data['healthy_test_probs']

                patient_test_seqs = sequences_data['patient_test']
                healthy_test_seqs = sequences_data['healthy_all'][52:]

                # Filter patient and healthy test probabilities based on bad and close bad sequences
                if not first_do_not_filter:
                    patient_test_probs = get_filtered_data_probs(patient_test_seqs, patient_test_probs, bad_seqs, close_bad_seqs, low_bound, high_bound)
                    healthy_test_probs = get_filtered_data_probs(healthy_test_seqs, healthy_test_probs, bad_seqs, close_bad_seqs, low_bound, high_bound)

                def calculate_all_differences():
                    """Calculate all log-likelihood differences for percentile analysis"""
                    all_differences = []

                    # Process patient test subjects
                    for patient_probs in patient_test_probs:
                        patient_probs_reshaped = patient_probs.flatten().reshape(-1, 1)
                        patient_ll = np.mean(final_patient_gmm.score_samples(patient_probs_reshaped))
                        healthy_ll = np.mean(final_healthy_gmm.score_samples(patient_probs_reshaped))
                        diff = patient_ll - healthy_ll
                        all_differences.append(diff)

                    # Process healthy test subjects
                    for healthy_test_prob in healthy_test_probs:
                        healthy_test_prob_reshaped = healthy_test_prob.flatten().reshape(-1, 1)
                        patient_ll = np.mean(final_patient_gmm.score_samples(healthy_test_prob_reshaped))
                        healthy_ll = np.mean(final_healthy_gmm.score_samples(healthy_test_prob_reshaped))
                        diff = patient_ll - healthy_ll
                        all_differences.append(diff)

                    return np.array(all_differences)

                def get_gmm_predictions(threshold):
                    """Helper function to get predictions for a given threshold"""
                    test_predictions = []
                    test_true_labels = []

                    # Process patient test subjects
                    for i, patient_probs in enumerate(patient_test_probs):
                        patient_probs_reshaped = patient_probs.flatten().reshape(-1, 1)

                        # Calculate average log-likelihood for this patient across all their probability values
                        patient_ll = np.mean(final_patient_gmm.score_samples(patient_probs_reshaped))
                        healthy_ll = np.mean(final_healthy_gmm.score_samples(patient_probs_reshaped))

                        # If threshold is set, classify based on the difference only if the abs difference is above the threshold
                        if threshold > 0:
                            diff = patient_ll - healthy_ll
                            if abs(diff) < threshold:
                                continue

                        # Classify as patient if closer to patient GMM
                        prediction = 1 if patient_ll > healthy_ll else 0

                        test_predictions.append(prediction)
                        test_true_labels.append(1)  # True label is patient

                    # Process healthy test subjects
                    for i, healthy_test_prob in enumerate(healthy_test_probs):
                        healthy_test_prob_reshaped = healthy_test_prob.flatten().reshape(-1, 1)

                        # Calculate average log-likelihood for this healthy subject across all their probability values
                        patient_ll = np.mean(final_patient_gmm.score_samples(healthy_test_prob_reshaped))
                        healthy_ll = np.mean(final_healthy_gmm.score_samples(healthy_test_prob_reshaped))

                        # If threshold is set, classify based on the difference only if the abs difference is above the threshold
                        if threshold > 0:
                            diff = patient_ll - healthy_ll
                            if abs(diff) < threshold:
                                continue

                        # Classify as patient if closer to patient GMM
                        prediction = 1 if patient_ll > healthy_ll else 0

                        test_predictions.append(prediction)
                        test_true_labels.append(0)  # True label is healthy

                    return np.array(test_predictions), np.array(test_true_labels)

                all_diffs = calculate_all_differences()
                abs_diffs = np.abs(all_diffs)

                percentiles = [90, 75, 50, 25]
                percentiles = [100 - p for p in percentiles]  # Convert to 100 - percentile
                percentile_values = np.percentile(abs_diffs, percentiles)

                gmm_predictions, y_test = get_gmm_predictions(0)
                cm_gmm = confusion_matrix(y_test, gmm_predictions)
                accuracy = np.mean(gmm_predictions == y_test)
                cm_gmms.append(cm_gmm)
                accuracies.append(accuracy)
                name_metadata = f" (l={low_bound},h={high_bound})" if not first_do_not_filter else ""
                classification_names.append(f'normal{name_metadata}')

                for percentile, threshold in zip(percentiles, percentile_values):
                    gmm_predictions, y_test = get_gmm_predictions(threshold)
                    cm_gmm = confusion_matrix(y_test, gmm_predictions)
                    accuracy = np.mean(gmm_predictions == y_test)
                    cm_gmms.append(cm_gmm)
                    accuracies.append(accuracy)
                    classification_names.append(f'{100 - percentile}th Percentile{name_metadata}')

                # Append results for this model
                all_classification_names.append(classification_names)
                all_cm_gmms.append(cm_gmms)
                all_accuracies.append(accuracies)
            first_do_not_filter = False  # Only do not filter for the first model!

            # create a dict of all_classification_names,...
            results = {
                'classification_names': all_classification_names,
                'cm_gmms': all_cm_gmms,
                'accuracies': all_accuracies,
                'low_bound': low_bound,
                'high_bound': high_bound
            }
            all_results.append(results)
    print("Done.")

    # save results to pickle
    with open(results_path, 'wb') as f:
        pickle.dump(all_results, f)

    # Step 3: Plot the results
    ablation_results = []
    ablation_names = []

    for result in all_results:
        classification_names = result['classification_names'][0]
        accuracies = np.array(result['accuracies'])

        for i, class_name in enumerate(classification_names):
            ablation_results.append(accuracies[:, i])
            ablation_names.append(class_name)

    def plot_grouped_ablation_subplots(ablation_results, ablation_names):
        """
        Group ablation results by the (l=...,h=...) part of their names (if any).
        Create one subplot per group with a boxplot + scatter overlay.
        """
        # --- Group results by (l=...,h=...) pattern ---
        grouped = defaultdict(list)
        for results, name in zip(ablation_results, ablation_names):
            match = re.search(r"\(l=.*?,h=.*?\)", name)
            group = match.group(0) if match else "default"
            grouped[group].append((name, results))

        group_labels = list(grouped.keys())

        # --- Setup subplots ---
        n_groups = len(group_labels)
        figsize = (2, 4 * n_groups)  # Adjust height based on number of groups
        fig, axes = plt.subplots(n_groups, 1, figsize=(figsize[0] * n_groups, figsize[1]), squeeze=False)
        axes = axes.flatten()

        # --- Plot each group ---
        for idx, group in enumerate(group_labels):
            ax = axes[idx]
            group_items = grouped[group]

            # Extract data + labels
            names = [n for n, _ in group_items]
            data = [r for _, r in group_items]

            # Boxplot
            bp = ax.boxplot(data, positions=np.arange(len(names)), widths=0.5, patch_artist=True)

            # Make boxes transparent
            for patch in bp['boxes']:
                patch.set_facecolor("skyblue")
                patch.set_alpha(0.3)

            # Scatter overlay for each class
            for i, arr in enumerate(data):
                x = np.random.normal(i, 0.05, size=len(arr))  # jitter
                ax.scatter(x, arr, alpha=0.6, s=20)

            ax.set_xticks(np.arange(len(names)))
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=9)
            ax.set_title(group)
            ax.set_ylabel("Accuracy")

        plt.tight_layout()
        plt.show()

    # Plot the ablation results
    plot_grouped_ablation_subplots(ablation_results, ablation_names)

    return


if __name__ == '__main__':
    # plot_embeddings()
    # plot_average_embeddings()
    # calculations_for_poster()
    # plot_ablation_on_folds()
    plot_classification_ablation()