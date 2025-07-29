import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb

from cache_handler import get_model_config_str
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap


MODEL_PREDICTION_BATCHED_EXTRA_SAMPLE_SIZE = 40000


def kde_normalizer(kde):
    for line in kde.get_lines():
        y_data = line.get_ydata()
        line.set_ydata(y_data / y_data.max())  # Normalize to max value of 1


def plot_output_distributions_claude(trained_model, valid_patient_inds, unique_patient_ids,
                                     valid_masks, positive_seqs, df_bld, patient_id_masks,
                                     train_patient_inds, train_inds, df_hlt, model_type, log_wandb, args, device='cuda'):
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
        neg_seqs = np.unique(neg_seqs)
        # make sure that neg_seqs does not contain any pos_seqs
        neg_seqs = np.setdiff1d(neg_seqs, pos_seqs, assume_unique=True)
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
        kde = sns.kdeplot(pos_probs, ax=ax1, color=validation_colors[i], label=f'Pos Set {i}', common_norm=True)
        kde_normalizer(kde)
        kde = sns.kdeplot(neg_probs, ax=ax1, color=validation_colors[i], linestyle='--', label=f'Neg Set {i}', common_norm=True)
        kde_normalizer(kde)
    ax1.set_xlabel("Predicted Probability for Positive Class")
    ax1.set_ylabel("Density")
    ax1.set_ylim(0, 1.1)
    ax1.legend()
    # Subplot 2: Training Set Distributions
    ax2.set_title("Training Set")
    for i, patient_ind in enumerate(train_patient_inds[:3]):  # Limit to first 3 for visibility
        mask_i = patient_id_masks[patient_ind]
        mask = ((mask_i == 1) & train_inds)
        pos_seqs = np.array(positive_seqs)[mask]
        # Negative sequences for this patient
        neg_seqs = df_bld.loc[df_bld["patient_id"] == unique_patient_ids[patient_ind], "AASeq"].values
        neg_seqs = np.unique(neg_seqs)
        # make sure that neg_seqs does not contain any pos_seqs
        neg_seqs = np.setdiff1d(neg_seqs, pos_seqs, assume_unique=True)
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
        kde = sns.kdeplot(pos_probs, ax=ax2, color=train_colors[i], label=f'Pos Set {i}', common_norm=True)
        kde_normalizer(kde)
        kde = sns.kdeplot(neg_probs, ax=ax2, color=train_colors[i], linestyle='--', label=f'Neg Set {i}', common_norm=True)
        kde_normalizer(kde)
    ax2.set_xlabel("Predicted Probability for Positive Class")
    ax2.set_ylabel("Density")
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    # Subplot 3: Healthy Patients Distributions
    ax3.set_title("Healthy Patients")
    healthy_patients = df_hlt["patient_id"].unique()
    for i in range(3):
        patient = healthy_patients[i]
        seqs = df_hlt.loc[df_hlt["patient_id"] == patient, "AASeq"].values
        seqs = np.unique(seqs)
        seqs = np.random.choice(seqs, size=min(10000, len(seqs)), replace=False)
        # Get model outputs
        trained_model.to(device)
        trained_model.eval()
        with torch.no_grad():
            logits = trained_model(seqs)
        # Convert to probabilities
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        # Plot KDE for healthy patient samples
        kde = sns.kdeplot(probs, ax=ax3, color=healthy_colors[i], label=f'Healthy Set {i}', common_norm=True)
        kde_normalizer(kde)
    ax3.set_xlabel("Predicted Probability for Positive Class")
    ax3.set_ylabel("Density")
    ax3.set_ylim(0, 1.1)
    ax3.legend()
    # Save the plot
    os.makedirs(f"plots/{model_type}_model/dist_model_output", exist_ok=True)
    plt.savefig(f"plots/{model_type}_model/dist_model_output/comprehensive_distribution_{get_model_config_str(args)}.png")
    plt.tight_layout()

    # save the figure in wandb:
    if log_wandb:
        wandb.log({"output_distributions": wandb.Image(plt)})
    else:
        plt.show()


def get_model_predictions_batched(model, sequences, sample_plots, batch_size=50000, device='cuda', threshold=None,
                                  extra_sample_size=MODEL_PREDICTION_BATCHED_EXTRA_SAMPLE_SIZE):
    """Helper function to get model predictions in batches"""
    sample_size = MODEL_PREDICTION_BATCHED_SAMPLE_SIZE
    model.to(device)
    model.eval()

    all_probs = []

    if sample_plots == 1:
        # sample sample_size sequences for plotting
        if len(sequences) > sample_size:
            indices = np.random.choice(len(sequences), size=sample_size, replace=False)
            sequences = [sequences[i] for i in indices]
    if sample_plots == 2:
        # sample sample_count sequences for plotting
        if len(sequences) > sample_size:
            extra_factor = (len(sequences) - sample_size) // extra_sample_size + 1
            sample_count = sample_size * extra_factor
            sample_count = min(sample_count, len(sequences))  # Safety cap
            indices = np.random.choice(len(sequences), size=sample_count, replace=False)
            sequences = [sequences[i] for i in indices]
    if sample_plots == 3:
        # sample sample_size times extra factor sequences for plotting
        if len(sequences) > sample_size:
            extra_factor = (len(sequences) - sample_size) // extra_sample_size + 1
            sequences_list = []
            for i in range(extra_factor):
                indices = np.random.choice(len(sequences), size=sample_size, replace=False)
                sequences_list.extend([sequences[i] for i in indices])
            sequences = sequences_list

    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch_seqs = sequences[i:i + batch_size]
            batch_logits = model(batch_seqs)
            batch_probs = torch.softmax(batch_logits, dim=1)[:, 1].cpu().numpy()
            all_probs.append(batch_probs)

    probs = np.concatenate(all_probs)
    if threshold is None:
        return probs
    else:
        return probs[probs >= threshold]

def plot_output_distributions_per_patient(trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
                                          test_masks, valid_masks, positive_seqs, df_bld,
                                          df_hlt, model_type, log_wandb, sample_plots, args,
                                          device='cuda', threshold=None, framealpha=0.1, legend_fontsize=6):
    """
    Create a comprehensive visualization of model output distributions across validation,
    training, and healthy patient sets.
    Parameters:
    - trained_model: The trained neural network model
    - Various data-related parameters to extract sequences and patient sets
    """
    global MODEL_PREDICTION_BATCHED_SAMPLE_SIZE
    if args.top_n_seqs is not None:
        MODEL_PREDICTION_BATCHED_SAMPLE_SIZE = args.top_n_seqs * 1000
    else:
        MODEL_PREDICTION_BATCHED_SAMPLE_SIZE = 20000

    def create_kde_from_histogram(probs, ax, color, label):
        """Helper function to create KDE plot from histogram of rounded probabilities"""
        if sample_plots == 3:
            sample_size = MODEL_PREDICTION_BATCHED_SAMPLE_SIZE
            extra_sample_size = MODEL_PREDICTION_BATCHED_EXTRA_SAMPLE_SIZE
            if len(probs) <= sample_size:
                extra_factor = 1
            else:
                extra_factor = (len(probs) - sample_size) // extra_sample_size + 1

            expanded_probs_list = []
            # Cut the probs into sample_size chunks:
            for i in range(extra_factor):
                start = i * sample_size
                end = start + sample_size
                if end > len(probs):
                    end = len(probs)
                chunk_probs = probs[start:end]

                # Round probabilities to 2 decimal places
                rounded_probs = np.round(chunk_probs, 2)

                # Create histogram (counts for each unique rounded value)
                unique_vals, counts = np.unique(rounded_probs, return_counts=True)

                # Create expanded array based on counts for KDE
                expanded_probs = np.repeat(unique_vals, counts)
                expanded_probs_list.append(expanded_probs)

            # Plot KDE
            expanded_probs = np.concatenate(expanded_probs_list)
        else:
            # Round probabilities to 2 decimal places
            rounded_probs = np.round(probs, 2)

            # Create histogram (counts for each unique rounded value)
            unique_vals, counts = np.unique(rounded_probs, return_counts=True)

            # Create expanded array based on counts for KDE
            expanded_probs = np.repeat(unique_vals, counts)

        # Plot KDE
        kde = sns.kdeplot(expanded_probs, ax=ax, color=color, label=label)
        kde_normalizer(kde)
        # Extract the KDE line and compute max y after x > 0.5
        line = kde.get_lines()[-1]  # Most recent line
        x_data, y_data = line.get_data()
        mask = x_data > 0.5
        if np.any(mask):
            max_y_after_05 = y_data[mask].max()
            label += f" (max={max_y_after_05:.3f})"
        line.set_label(label)  # Set label with updated string
        return kde

    # Set up the figure with three subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Model Output Distributions for {model_type} Model", fontsize=16)

    # Create new test_inds which is all test_patient_inds and one from valid_patient_inds
    test_inds = np.concatenate([test_patient_inds, valid_patient_inds])
    # And create a new test_mask which contains both
    test_masks = np.concatenate((test_masks, valid_masks), axis=0)

    # Color palettes for different sets
    min_color, max_color = 0.4, 0.8
    test_pos_colors = plt.cm.Greens(np.linspace(min_color, max_color, len(test_inds)))
    test_neg_colors = plt.cm.Reds(np.linspace(min_color, max_color, len(test_inds)))
    test_colors = plt.cm.Purples(np.linspace(min_color, max_color, len(test_inds)))
    healthy_colors = plt.cm.Oranges(np.linspace(min_color, max_color, len(test_inds)))

    # Subplot 1: Validation Set Distributions
    ax1.set_title("Test Set Positives and Negatives")
    for i, patient_ind in enumerate(test_inds):
        mask = (test_masks[i] == 1)
        pos_seqs = np.array(positive_seqs)[mask]
        # Negative sequences for this patient
        neg_seqs = df_bld.loc[df_bld["patient_id"] == unique_patient_ids[patient_ind], "AASeq"].values
        neg_seqs = np.unique(neg_seqs)
        # make sure that all pos_seqs are in neg_seqs
        assert np.all(np.isin(pos_seqs, neg_seqs)), f"Positives are not in negatives for patient {patient_ind}"
        # Make sure that there are no sequences in the negative set that are in the positive set
        neg_seqs = np.setdiff1d(neg_seqs, pos_seqs, assume_unique=True)

        # Get model outputs using batched prediction
        pos_probs = get_model_predictions_batched(trained_model, pos_seqs, sample_plots, device=device, threshold=threshold)
        neg_probs = get_model_predictions_batched(trained_model, neg_seqs, sample_plots, device=device, threshold=threshold)

        # # Get model outputs
        # trained_model.to(device)
        # trained_model.eval()
        # with torch.no_grad():
        #     pos_logits = trained_model(pos_seqs)
        #     neg_logits = trained_model(neg_seqs)
        # # Convert to probabilities
        # pos_probs = torch.softmax(pos_logits, dim=1)[:, 1].cpu().numpy()
        # neg_probs = torch.softmax(neg_logits, dim=1)[:, 1].cpu().numpy()
        # Plot KDE for positive and negative samples
        create_kde_from_histogram(pos_probs, ax1, test_pos_colors[i], f'Pos Set {i} ({unique_patient_ids[patient_ind]}) - {len(pos_seqs)}')
        create_kde_from_histogram(neg_probs, ax1, test_neg_colors[i], f'Neg Set {i} ({unique_patient_ids[patient_ind]}) - {len(neg_seqs)}')
        # kde = sns.kdeplot(pos_probs, ax=ax1, color=test_pos_colors[i], label=f'Pos Set {i}')
        # kde_normalizer(kde)
        # kde = sns.kdeplot(neg_probs, ax=ax1, color=test_neg_colors[i], label=f'Neg Set {i}', common_norm=True)
        # kde_normalizer(kde)
    ax1.set_xlabel("Predicted Probability for Positive Class")
    ax1.set_ylabel("Density")
    ax1.set_ylim(0, 1.1)
    legend = ax1.legend(framealpha=framealpha, fontsize=legend_fontsize)
    for text in legend.get_texts():
        text.set_alpha(0.5)

    # Subplot 2: Healthy Patients Distributions
    ax2.set_title("Healthy vs Ill Patients")
    healthy_patients = df_hlt["patient_id"].unique()
    rng = np.random.default_rng(42)  # For reproducibility
    healthy_patients = sorted(healthy_patients)
    healthy_patients = rng.permutation(healthy_patients)
    for i, patient_ind in enumerate(test_inds):
        patient = healthy_patients[i]
        healthy_seqs = df_hlt.loc[df_hlt["patient_id"] == patient, "AASeq"].values
        healthy_seqs = np.unique(healthy_seqs)
        disease_seqs = df_bld.loc[df_bld["patient_id"] == unique_patient_ids[patient_ind], "AASeq"].values
        disease_seqs = np.unique(disease_seqs)
        # Get model outputs using batched prediction
        healthy_probs = get_model_predictions_batched(trained_model, healthy_seqs, sample_plots, device=device, threshold=threshold)
        disease_probs = get_model_predictions_batched(trained_model, disease_seqs, sample_plots, device=device, threshold=threshold)
        # # Get model outputs
        # trained_model.to(device)
        # trained_model.eval()
        # with torch.no_grad():
        #     healthy_logits = trained_model(healthy_seqs)
        #     disease_logits = trained_model(disease_seqs)
        # # Convert to probabilities
        # healthy_probs = torch.softmax(healthy_logits, dim=1)[:, 1].cpu().numpy()
        # disease_probs = torch.softmax(disease_logits, dim=1)[:, 1].cpu().numpy()
        # Plot KDE for healthy patient samples
        create_kde_from_histogram(healthy_probs, ax2, healthy_colors[i], f'Healthy Set {i} ({patient}) - {len(healthy_seqs)}')
        create_kde_from_histogram(disease_probs, ax2, test_colors[i], f'Ill Set {i} ({unique_patient_ids[patient_ind]}) - {len(disease_seqs)}')
        # kde = sns.kdeplot(healthy_probs, ax=ax2, color=healthy_colors[i], label=f'Healthy Set {i}', common_norm=True)
        # kde_normalizer(kde)
        # kde = sns.kdeplot(disease_probs, ax=ax2, color=test_colors[i], label=f'Ill Set {i}', common_norm=True)
        # kde_normalizer(kde)
    # More Healthy patients
    for i in range(len(test_inds), len(test_inds) + 6):
        patient = healthy_patients[i]
        healthy_seqs = df_hlt.loc[df_hlt["patient_id"] == patient, "AASeq"].values
        healthy_seqs = np.unique(healthy_seqs)
        # Get model outputs using batched prediction
        healthy_probs = get_model_predictions_batched(trained_model, healthy_seqs, sample_plots, device=device, threshold=threshold)
        # Get model outputs
        # trained_model.to(device)
        # trained_model.eval()
        # with torch.no_grad():
        #     healthy_logits = trained_model(healthy_seqs)
        # # Convert to probabilities
        # healthy_probs = torch.softmax(healthy_logits, dim=1)[:, 1].cpu().numpy()
        # Plot KDE for healthy patient samples
        create_kde_from_histogram(healthy_probs, ax2, healthy_colors[i - len(test_inds)], f'Healthy Set {i} ({patient}) - {len(healthy_seqs)}')
        # kde = sns.kdeplot(healthy_probs, ax=ax2, color=healthy_colors[i - len(test_inds)], label=f'Healthy Set {i}', common_norm=True)
        # kde_normalizer(kde)
    ax2.set_xlabel("Predicted Probability for Positive Class")
    ax2.set_ylabel("Density")
    ax2.set_ylim(0, 1.1)
    # loc = 'upper center' if threshold != 0.5 else 'best'
    legend = ax2.legend(framealpha=framealpha, fontsize=legend_fontsize)
    for text in legend.get_texts():
        text.set_alpha(0.5)
    # Save the plot
    os.makedirs(f"plots/{model_type}_model/dist_model_output", exist_ok=True)
    extra = f"_threshold_{threshold}" if threshold is not None else ""
    plt.savefig(f"plots/{model_type}_model/dist_model_output/comprehensive_distribution_per_patients_{get_model_config_str(args)}{extra}.png")
    plt.tight_layout()

    # save the figure in wandb:
    if log_wandb:
        title = "output_distributions_per_patient"
        if threshold is not None:
            title += f"_threshold_{threshold}"
        wandb.log({title: wandb.Image(plt)})
    else:
        plt.show()

    # Add plot for cumulative distributions instead of KDE:
    # Add plot for cumulative distributions instead of KDE:
    # Create a new figure for CDF plots with two subplots
    fig_cdf, (cdf_ax1, cdf_ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig_cdf.suptitle(f"Cumulative Distribution Functions for {model_type} Model", fontsize=16)

    # Subplot 1: Test Set Positives and Negatives CDFs
    cdf_ax1.set_title("Test Set Positives and Negatives (CDF)")
    for i, patient_ind in enumerate(test_inds):
        mask = (test_masks[i] == 1)
        pos_seqs = np.array(positive_seqs)[mask]
        # Negative sequences for this patient
        neg_seqs = df_bld.loc[df_bld["patient_id"] == unique_patient_ids[patient_ind], "AASeq"].values
        neg_seqs = np.unique(neg_seqs)
        # Make sure that there are no sequences in the negative set that are in the positive set
        neg_seqs = np.setdiff1d(neg_seqs, pos_seqs, assume_unique=True)
        # Get model outputs using batched prediction
        pos_probs = get_model_predictions_batched(trained_model, pos_seqs, sample_plots, device=device, threshold=threshold)
        neg_probs = get_model_predictions_batched(trained_model, neg_seqs, sample_plots, device=device, threshold=threshold)
        # # Get model outputs
        # trained_model.to(device)
        # trained_model.eval()
        # with torch.no_grad():
        #     pos_logits = trained_model(pos_seqs)
        #     neg_logits = trained_model(neg_seqs)
        # # Convert to probabilities
        # pos_probs = torch.softmax(pos_logits, dim=1)[:, 1].cpu().numpy()
        # neg_probs = torch.softmax(neg_logits, dim=1)[:, 1].cpu().numpy()

        # Plot CDFs for positive and negative samples
        # For positive sequences
        x = np.sort(pos_probs)
        y = np.arange(1, len(x) + 1) / len(x)
        cdf_ax1.plot(x, y, color=test_pos_colors[i], label=f'Pos Set {i} - {len(pos_seqs)}')

        # For negative sequences
        x = np.sort(neg_probs)
        y = np.arange(1, len(x) + 1) / len(x)
        cdf_ax1.plot(x, y, color=test_neg_colors[i], label=f'Neg Set {i} - {len(neg_seqs)}')

    cdf_ax1.set_xlabel("Predicted Probability for Positive Class")
    cdf_ax1.set_ylabel("Cumulative Probability")
    cdf_ax1.set_ylim(0, 1.05)
    legend = cdf_ax1.legend(framealpha=framealpha, fontsize=legend_fontsize)
    for text in legend.get_texts():
        text.set_alpha(0.5)

    # Subplot 2: Healthy vs Ill Patients CDFs
    cdf_ax2.set_title("Healthy vs Ill Patients (CDF)")
    for i, patient_ind in enumerate(test_inds):
        patient = healthy_patients[i]
        healthy_seqs = df_hlt.loc[df_hlt["patient_id"] == patient, "AASeq"].values
        healthy_seqs = np.unique(healthy_seqs)
        disease_seqs = df_bld.loc[df_bld["patient_id"] == unique_patient_ids[patient_ind], "AASeq"].values
        disease_seqs = np.unique(disease_seqs)
        # Get model outputs using batched prediction
        healthy_probs = get_model_predictions_batched(trained_model, healthy_seqs, sample_plots, device=device, threshold=threshold)
        disease_probs = get_model_predictions_batched(trained_model, disease_seqs, sample_plots, device=device, threshold=threshold)
        # # Get model outputs
        # trained_model.to(device)
        # trained_model.eval()
        # with torch.no_grad():
        #     healthy_logits = trained_model(healthy_seqs)
        #     disease_logits = trained_model(disease_seqs)
        # # Convert to probabilities
        # healthy_probs = torch.softmax(healthy_logits, dim=1)[:, 1].cpu().numpy()
        # disease_probs = torch.softmax(disease_logits, dim=1)[:, 1].cpu().numpy()

        # Plot CDFs for healthy and disease samples
        # For healthy sequences
        x = np.sort(healthy_probs)
        y = np.arange(1, len(x) + 1) / len(x)
        cdf_ax2.plot(x, y, color=healthy_colors[i], label=f'Healthy Set {i} - {len(healthy_seqs)}')

        # For disease sequences
        x = np.sort(disease_probs)
        y = np.arange(1, len(x) + 1) / len(x)
        cdf_ax2.plot(x, y, color=test_colors[i], label=f'Ill Set {i} - {len(disease_seqs)}')

    # More Healthy patients
    for i in range(len(test_inds), len(test_inds) + 6):
        patient = healthy_patients[i]
        healthy_seqs = df_hlt.loc[df_hlt["patient_id"] == patient, "AASeq"].values
        healthy_seqs = np.unique(healthy_seqs)
        # Get model outputs using batched prediction
        healthy_probs = get_model_predictions_batched(trained_model, healthy_seqs, sample_plots, device=device, threshold=threshold)
        # # Get model outputs
        # trained_model.to(device)
        # trained_model.eval()
        # with torch.no_grad():
        #     healthy_logits = trained_model(healthy_seqs)
        # # Convert to probabilities
        # healthy_probs = torch.softmax(healthy_logits, dim=1)[:, 1].cpu().numpy()

        # Plot CDF for healthy patient samples
        x = np.sort(healthy_probs)
        y = np.arange(1, len(x) + 1) / len(x)
        cdf_ax2.plot(x, y, color=healthy_colors[i - len(test_inds)], label=f'Healthy Set {i} - {len(healthy_seqs)}')

    cdf_ax2.set_xlabel("Predicted Probability for Positive Class")
    cdf_ax2.set_ylabel("Cumulative Probability")
    cdf_ax2.set_ylim(0, 1.05)
    legend = cdf_ax2.legend(framealpha=framealpha, fontsize=legend_fontsize)
    for text in legend.get_texts():
        text.set_alpha(0.5)

    # Save the CDF plot
    plt.tight_layout()
    extra = f"_threshold_{threshold}" if threshold is not None else ""
    plt.savefig(f"plots/{model_type}_model/dist_model_output/cumulative_distribution_per_patients_{get_model_config_str(args)}{extra}.png")

    # Save the figure in wandb:
    if log_wandb:
        title = "cumulative_distributions_per_patient"
        if threshold is not None:
            title += f"_threshold_{threshold}"
        wandb.log({title: wandb.Image(fig_cdf)})
    else:
        plt.show()


def plot_output_distributions_per_patient_new(trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
                                              test_masks, valid_masks, positive_seqs, df_bld,
                                              df_hlt, model_type, log_wandb, args, device='cuda'):
    """
    Create a comprehensive visualization of model output distributions across validation,
    training, and healthy patient sets.
    Parameters:
    - trained_model: The trained neural network model
    - Various data-related parameters to extract sequences and patient sets
    """
    # Get 10 random healthy patients for average healthy distribution
    healthy_patients = df_hlt["patient_id"].unique()
    np.random.shuffle(healthy_patients)
    healthy_patients = healthy_patients[:10]

    # Prepare combined test and validation indices
    test_inds = np.concatenate([test_patient_inds, valid_patient_inds])

    # Collect distributions
    healthy_dists = []
    disease_dists = []
    individual_disease_dists = []
    individual_disease_labels = []

    # Process healthy patients to create average distribution
    for patient in healthy_patients:
        healthy_seqs = df_hlt.loc[df_hlt["patient_id"] == patient, "AASeq"].values
        healthy_seqs = np.unique(healthy_seqs)

        # Get model outputs
        trained_model.to(device)
        trained_model.eval()
        with torch.no_grad():
            healthy_logits = trained_model(healthy_seqs)

        # Convert to probabilities
        healthy_probs = torch.softmax(healthy_logits, dim=1)[:, 1].cpu().numpy()
        healthy_dists.append(healthy_probs)

    # Process disease patients
    for i, patient_ind in enumerate(test_inds):
        # Get disease sequences for this patient
        disease_seqs = df_bld.loc[df_bld["patient_id"] == unique_patient_ids[patient_ind], "AASeq"].values
        disease_seqs = np.unique(disease_seqs)

        # Get model outputs
        trained_model.to(device)
        trained_model.eval()
        with torch.no_grad():
            disease_logits = trained_model(disease_seqs)

        # Convert to probabilities
        disease_probs = torch.softmax(disease_logits, dim=1)[:, 1].cpu().numpy()

        # Add to collections
        disease_dists.append(disease_probs)

        # Store individual distributions with labels (test or validation)
        individual_disease_dists.append(disease_probs)
        if i < len(test_patient_inds):
            individual_disease_labels.append(f"Test Patient {i}")
        else:
            individual_disease_labels.append(f"Valid Patient {i - len(test_patient_inds)}")

    # FIGUREs variables:
    xlim = (-0.15, 1.05)
    ylim = (-0.02, 1.2)
    dpi = 600

    # FIGURE 1: Average distributions with std
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    for spine in ax1.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.75)
    fig1.suptitle(f"Average Output Distributions for {model_type} Model", fontsize=16)
    # Plot average healthy distribution with std
    plot_average_dist_with_std(ax1, healthy_dists, color="#7BC8F6", label="Healthy Distribution")
    # Plot average disease distribution with std
    plot_average_dist_with_std(ax1, disease_dists, color="#FFA500", label="Patient Distribution")
    ax1.set_xlabel("Predicted Probability for Positive Class")
    ax1.set_ylabel("Density")
    ax1.set_xlim(xlim[0], xlim[1])  # Focus on the set range
    ax1.set_ylim(ylim[0], ylim[1])  # Focus on the set range
    ax1.set_xticks(np.arange(0.0, 1.1, 0.2))  # Ticks from 0.0 to 1.0 with step 0.1
    ax1.set_yticks(np.arange(0.0, 1.1, 0.2))  # Ticks from 0.0 to 1.0 with step 0.2
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Save figure 1
    fig1_path = f"plots/{model_type}_model/dist_model_output/average_distributions_nolegend_{get_model_config_str(args)}.png"
    os.makedirs(os.path.dirname(fig1_path), exist_ok=True)
    plt.savefig(fig1_path, dpi=dpi)
    ax1.legend(loc='upper right', framealpha=1.0)
    fig1_path = f"plots/{model_type}_model/dist_model_output/average_distributions_{get_model_config_str(args)}.png"
    plt.savefig(fig1_path, dpi=dpi)

    # FIGURE 2: Individual patient distributions vs healthy average
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    for spine in ax2.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.75)
    fig2.suptitle(f"Individual Patient Distributions for {model_type} Model", fontsize=16)
    # Plot average healthy distribution with std (same as figure 1)
    plot_average_dist_with_std(ax2, healthy_dists, color="#7BC8F6", label="Healthy Distribution")
    # Plot each individual disease patient
    # Use colormap for differentiation between patients
    for i, (dist, label) in enumerate(zip(individual_disease_dists, individual_disease_labels)):
        if i == 0:
            kde = sns.kdeplot(
                dist,
                ax=ax2,
                color="#FFA500",
                label="Patients Distribution",
                common_norm=True
            )
        else:
            kde = sns.kdeplot(
                dist,
                ax=ax2,
                color="#FFA500",
                common_norm=True
            )
        # Get the last line (the KDE curve just added)
        line = kde.get_lines()[-1]
        # Get the data from the KDE curve
        y_data = line.get_ydata()
        # Normalize by dividing by max value
        max_value = np.max(y_data)
        if max_value > 0:
            normalized_y = y_data / max_value
            # Update the line with normalized values
            line.set_ydata(normalized_y)
    ax2.set_xlabel("Predicted Probability for Positive Class")
    ax2.set_ylabel("Density")
    ax2.set_xlim(xlim[0], xlim[1])  # Focus on the set range
    ax2.set_ylim(ylim[0], ylim[1])  # Focus on the set range
    ax2.set_xticks(np.arange(0.0, 1.1, 0.2))  # Ticks from 0.0 to 1.0 with step 0.1
    ax2.set_yticks(np.arange(0.0, 1.1, 0.2))  # Ticks from 0.0 to 1.0 with step 0.2
    # Create a custom legend with better spacing
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Save figure 2
    plt.tight_layout()
    fig2_path = f"plots/{model_type}_model/dist_model_output/individual_distributions_nolegend_{get_model_config_str(args)}.png"
    plt.savefig(fig2_path, dpi=dpi)
    ax2.legend(loc='upper right', framealpha=1.0)
    fig2_path = f"plots/{model_type}_model/dist_model_output/individual_distributions_{get_model_config_str(args)}.png"
    plt.savefig(fig2_path, dpi=dpi)

    # Log to wandb if requested
    if log_wandb:
        wandb.log({
            "average_distributions": wandb.Image(fig1),
            "individual_distributions": wandb.Image(fig2)
        })
    else:
        plt.show()

    plt.close(fig1)
    plt.close(fig2)


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
            alpha=0.3
        )

    # Set the label after normalization
    line.set_label(label)

    return line


def plot_output_distributions_unseen_ms(trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
                                        test_masks, valid_masks, positive_seqs, df_bld,
                                        df_hlt, model_type, log_wandb, args, device='cuda'):
    # read the .xlsx file "db/ms_related/IEDB_MS_AB.xlsx"
    df_ms = pd.read_excel("db/ms_related/IEDB_MS_AB.xlsx")
    # read the .csv file "db/ms_related/McPAS-TCR_MS_seqs.csv"
    df_mcpas = pd.read_csv("db/ms_related/McPAS-TCR_MS_seqs.csv")
    seqs1 = df_ms["Chain 1 CDR3"]  # TODO: I think chain 1 is not necessary related to our sequences.
    seqs2 = df_ms["Chain 2 CDR3"]
    seqs3 = df_mcpas["CDR3.beta.aa"]
    AA_LETTERS = "ACDEFGHIKLMNPQRSTVWY"
    all_seqs = np.concatenate([seqs1, seqs2, seqs3])
    # Filter out sequences that are not in the AA_LETTERS
    all_seqs = [seq for seq in all_seqs if all(letter in AA_LETTERS for letter in seq)]
    # If the start of a sequence does not start with C add it
    all_seqs = [seq if seq.startswith("C") else "C" + seq for seq in all_seqs]
    all_seqs = list(set([seq if seq.endswith("F") else seq + "F" for seq in all_seqs]))

    # Get 10 random healthy patients for average healthy distribution
    healthy_patients = df_hlt["patient_id"].unique()
    np.random.shuffle(healthy_patients)
    healthy_patients = healthy_patients[:10]

    # Collect distributions
    healthy_dists = []
    disease_dists = []
    individual_disease_dists = []
    individual_disease_labels = []

    # Process healthy patients to create average distribution
    for patient in healthy_patients:
        healthy_seqs = df_hlt.loc[df_hlt["patient_id"] == patient, "AASeq"].values
        healthy_seqs = np.unique(healthy_seqs)

        # Get model outputs
        trained_model.to(device)
        trained_model.eval()
        with torch.no_grad():
            healthy_logits = trained_model(healthy_seqs)

        # Convert to probabilities
        healthy_probs = torch.softmax(healthy_logits, dim=1)[:, 1].cpu().numpy()
        healthy_dists.append(healthy_probs)

    # Process disease patients
    # for i, patient_ind in enumerate(test_inds):
    # Get disease sequences for this patient
    disease_seqs = np.array(all_seqs)

    # Get model outputs
    trained_model.to(device)
    trained_model.eval()
    with torch.no_grad():
        disease_logits = trained_model(disease_seqs)

    # Convert to probabilities
    disease_probs = torch.softmax(disease_logits, dim=1)[:, 1].cpu().numpy()

    # Add to collections
    disease_dists.append(disease_probs)

    # Store individual distributions with labels (test or validation)
    individual_disease_dists.append(disease_probs)
    individual_disease_labels.append(f"Test Patient {0}")

    # FIGUREs variables:
    xlim = (-0.15, 1.05)
    ylim = (-0.02, 1.2)
    dpi = 600

    # FIGURE 1: Average distributions with std
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    for spine in ax1.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.75)
    fig1.suptitle(f"Average Output Distributions for {model_type} Model", fontsize=16)
    # Plot average healthy distribution with std
    plot_average_dist_with_std(ax1, healthy_dists, color="#7BC8F6", label="Healthy Distribution")
    # Plot average disease distribution with std
    plot_average_dist_with_std(ax1, disease_dists, color="#FFA500", label="Patient Distribution")
    ax1.set_xlabel("Predicted Probability for Positive Class")
    ax1.set_ylabel("Density")
    ax1.set_xlim(xlim[0], xlim[1])  # Focus on the set range
    ax1.set_ylim(ylim[0], ylim[1])  # Focus on the set range
    ax1.set_xticks(np.arange(0.0, 1.1, 0.2))  # Ticks from 0.0 to 1.0 with step 0.1
    ax1.set_yticks(np.arange(0.0, 1.1, 0.2))  # Ticks from 0.0 to 1.0 with step 0.2
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Save figure 1
    fig1_path = f"plots/{model_type}_model/dist_model_output/ms_data/distributions_nolegend_{get_model_config_str(args)}.png"
    os.makedirs(os.path.dirname(fig1_path), exist_ok=True)
    plt.savefig(fig1_path, dpi=dpi)
    ax1.legend(loc='upper right', framealpha=1.0)
    fig1_path = f"plots/{model_type}_model/dist_model_output/ms_data/distributions_{get_model_config_str(args)}.png"
    plt.savefig(fig1_path, dpi=dpi)

    # Plot a histogram of the disease distributions
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    for spine in ax2.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.75)
    fig2.suptitle(f"Histogram of Disease Distributions for {model_type} Model", fontsize=16)
    # Plot histogram
    ax2.hist(disease_probs, bins=50, color="#FFA500", alpha=0.7, label="Disease Distribution")
    ax2.set_xlabel("Predicted Probability for Positive Class")
    ax2.set_ylabel("Count")
    ax2.set_xticks(np.arange(0.0, 1.1, 0.2))  # Ticks from 0.0 to 1.0 with step 0.1
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # Save figure 2
    fig2_path = f"plots/{model_type}_model/dist_model_output/ms_data/histogram_{get_model_config_str(args)}.png"
    os.makedirs(os.path.dirname(fig2_path), exist_ok=True)
    plt.savefig(fig2_path, dpi=dpi)
    ax2.legend(loc='upper right', framealpha=1.0)
    fig2_path = f"plots/{model_type}_model/dist_model_output/ms_data/histogram_{get_model_config_str(args)}.png"
    plt.savefig(fig2_path, dpi=dpi)

    # Log to wandb if requested
    if log_wandb:
        wandb.log({
            "average_distributions": wandb.Image(fig1),
            "individual_distributions": wandb.Image(fig2)
        })
    else:
        plt.show()

    plt.close(fig1)
    plt.close(fig2)


def display_ratio_figures(df_bld, positive_seqs, neg_seqs, aaseq_to_ratio, dataset_type, dpi=600, bin_num=200, threshold_steps=1000):
    # Pick 5 random patients
    random_patients = np.random.choice(df_bld['patient_id'].unique(), size=5, replace=False)
    # Pick random sequences from neg_seqs in the length of positive_seqs
    neg_seqs = np.random.choice(neg_seqs, size=min(len(neg_seqs), len(positive_seqs)), replace=False)

    # Prepare data for both plots
    ratios_before = {}
    ratios_after = {}

    for patient_id in random_patients:
        aaseqs = df_bld[df_bld['patient_id'] == patient_id]['AASeq'].unique()
        ratios_before[patient_id] = aaseq_to_ratio(aaseqs, dont_use_function=True)  # before applying f
        ratios_after[patient_id] = aaseq_to_ratio(aaseqs, dont_use_function=False)  # after applying f

    # Ratios for positive sequences
    pos_ratios_before = aaseq_to_ratio(positive_seqs, dont_use_function=True)
    pos_ratios_after = aaseq_to_ratio(positive_seqs, dont_use_function=False)

    # Ratios for negative sequences
    neg_ratios_before = aaseq_to_ratio(neg_seqs, dont_use_function=True)
    neg_ratios_after = aaseq_to_ratio(neg_seqs, dont_use_function=False)

    # Plotting (high-res)
    fig, axes = plt.subplots(3, 2, figsize=(18, 18), sharey='row', dpi=dpi)

    # --- Top row: Random patients ---
    # Normalized KDE for top-left plot
    for pid in random_patients:
        kde = sns.kdeplot(ratios_before[pid], ax=axes[0, 0], label=f"Patient {pid}", common_norm=False)
        # Get the y data for the last line that was plotted
        line = axes[0, 0].get_lines()[-1]
        y_data = line.get_ydata()
        # Normalize
        line.set_ydata(y_data / np.max(y_data))

    axes[0, 0].set_title("KDE of Ratios for 5 Patients (BEFORE f)")
    axes[0, 0].set_xlabel("Ratios")
    axes[0, 0].set_ylabel("Normalized Density")
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1.05)  # Set y-limit to ensure max is 1

    # Normalized KDE for top-right plot
    for pid in random_patients:
        kde = sns.kdeplot(ratios_after[pid], ax=axes[0, 1], label=f"Patient {pid}", common_norm=False)
        # Get the y data for the last line that was plotted
        line = axes[0, 1].get_lines()[-1]
        y_data = line.get_ydata()
        # Normalize
        line.set_ydata(y_data / np.max(y_data))

    axes[0, 1].set_title("KDE of f(Ratios) for 5 Patients (AFTER f)")
    axes[0, 1].set_xlabel("f(Ratios)")
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1.05)  # Set y-limit to ensure max is 1

    # --- Bottom row: Histograms for Positive and negative sequences ---
    # Histogram for bottom-left plot (BEFORE f)
    # Calculate simple means directly
    pos_mean = np.mean(pos_ratios_before)
    neg_mean = np.mean(neg_ratios_before)

    # Add vertical lines at the means
    axes[1, 0].axvline(x=pos_mean, color='tab:orange', linestyle='-', alpha=0.9,
                       label=f"Pos Mean: {pos_mean:.3f}")
    axes[1, 0].axvline(x=neg_mean, color='tab:blue', linestyle='-', alpha=0.9,
                       label=f"Neg Mean: {neg_mean:.3f}")
    axes[1, 0].legend()

    # Calculate bins based on the range of data
    all_data_before = np.concatenate([pos_ratios_before, neg_ratios_before])
    bins = np.linspace(min(all_data_before), max(all_data_before), bin_num)

    # Plot histograms with transparency for overlap visibility
    axes[1, 0].hist(pos_ratios_before, bins=bins, color='tab:orange', alpha=0.5, label="Positive Sequences", density=True)
    axes[1, 0].hist(neg_ratios_before, bins=bins, color='tab:blue', alpha=0.5, label="Negative Sequences", density=True)

    axes[1, 0].set_title("Histogram of Ratios for Positive vs Negative Sequences (BEFORE f)")
    axes[1, 0].set_xlabel("Ratios")
    axes[1, 0].set_ylabel("Normalized Frequency")
    axes[1, 0].legend()

    # Set ticks of x-axis for the bottom-left plot
    ticks_range = np.arange(0, 1.01, 0.01)
    axes[1, 0].set_xticks(ticks_range)
    ticks_labels = [f"{tick:.2f}" for i, tick in enumerate(ticks_range)]
    # ticks_labels = [f"{tick:.2f}" if i % 2 == 0 else "" for i, tick in enumerate(ticks_range)]
    axes[1, 0].set_xticklabels(ticks_labels, rotation=90)
    axes[1, 0].set_xlim(0, 0.2)

    # Histogram for bottom-right plot (AFTER f)
    # Calculate means for after f
    pos_mean_after = torch.mean(pos_ratios_after)
    neg_mean_after = torch.mean(neg_ratios_after)

    # Add vertical lines at the means
    axes[1, 1].axvline(x=pos_mean_after, color='tab:orange', linestyle='-', alpha=0.9,
                       label=f"Pos Mean: {pos_mean_after:.3f}")
    axes[1, 1].axvline(x=neg_mean_after, color='tab:blue', linestyle='-', alpha=0.9,
                       label=f"Neg Mean: {neg_mean_after:.3f}")
    axes[1, 1].legend()

    # Calculate bins based on the range of data
    all_data_after = np.concatenate([pos_ratios_after, neg_ratios_after])
    bins = np.linspace(min(all_data_after), max(all_data_after), bin_num)

    # Plot histograms with transparency for overlap visibility
    axes[1, 1].hist(pos_ratios_after, bins=bins, color='tab:orange', alpha=0.5, label="Positive Sequences", density=True)
    axes[1, 1].hist(neg_ratios_after, bins=bins, color='tab:blue', alpha=0.5, label="Negative Sequences", density=True)

    axes[1, 1].set_title("Histogram of f(Ratios) for Positive vs Negative Sequences (AFTER f)")
    axes[1, 1].set_xlabel("f(Ratios)")
    axes[1, 1].legend()

    # --- Bottomest row: Positive sequences and negatives ---
    # Normalized KDE for bottom-left plot
    kde_pos = sns.kdeplot(pos_ratios_before, ax=axes[2, 0], color='tab:orange', label="Positive Sequences",
                          common_norm=False)
    line_pos = axes[2, 0].get_lines()[-1]
    y_data_pos = line_pos.get_ydata()
    line_pos.set_ydata(y_data_pos / np.max(y_data_pos))

    kde_neg = sns.kdeplot(neg_ratios_before, ax=axes[2, 0], color='tab:blue', linestyle='--',
                          label="Negative Sequences", common_norm=False)
    line_neg = axes[2, 0].get_lines()[-1]
    y_data_neg = line_neg.get_ydata()
    line_neg.set_ydata(y_data_neg / np.max(y_data_neg))

    axes[2, 0].set_title("KDE of Ratios for Positive Sequences (BEFORE f)")
    axes[2, 0].set_xlabel("Ratios")
    axes[2, 0].set_ylabel("Normalized Density")
    axes[2, 0].legend()
    axes[2, 0].set_ylim(0, 1.05)  # Set y-limit to ensure max is 1

    # Calculate simple means directly
    pos_mean = np.mean(pos_ratios_before)
    neg_mean = np.mean(neg_ratios_before)

    # Add vertical lines at the means
    axes[2, 0].axvline(x=pos_mean, color='tab:orange', linestyle='-', alpha=0.7,
                       label=f"Pos Mean: {pos_mean:.3f}")
    axes[2, 0].axvline(x=neg_mean, color='tab:blue', linestyle='-', alpha=0.7,
                       label=f"Neg Mean: {neg_mean:.3f}")
    axes[2, 0].legend()
    axes[2, 0].set_xlim(0, 0.2)

    # Normalized KDE for bottom-right plot
    kde_pos = sns.kdeplot(pos_ratios_after, ax=axes[2, 1], color='tab:orange', label="Positive Sequences",
                          common_norm=False)
    line_pos = axes[2, 1].get_lines()[-1]
    y_data_pos = line_pos.get_ydata()
    line_pos.set_ydata(y_data_pos / np.max(y_data_pos))

    kde_neg = sns.kdeplot(neg_ratios_after, ax=axes[2, 1], color='tab:blue', linestyle='--', label="Negative Sequences",
                          common_norm=False)
    line_neg = axes[2, 1].get_lines()[-1]
    y_data_neg = line_neg.get_ydata()
    line_neg.set_ydata(y_data_neg / np.max(y_data_neg))

    axes[2, 1].set_title("KDE of f(Ratios) for Positive Sequences (AFTER f)")
    axes[2, 1].set_xlabel("f(Ratios)")
    axes[2, 1].legend()
    axes[2, 1].set_ylim(0, 1.05)  # Set y-limit to ensure max is 1

    plt.tight_layout()
    os.makedirs("plots/ratio_figures", exist_ok=True)
    plt.savefig(f"plots/ratio_figures/ratios_{dataset_type}.png", dpi=dpi)
    plt.show()

    # --- Threshold of Ratio Figure ---
    # Create a new figure for threshold analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), dpi=dpi)

    # Sort the data to create thresholds
    sorted_data = np.sort(np.concatenate([pos_ratios_before, neg_ratios_before]))
    # Create threshold values from min to max with configurable steps
    thresholds = np.linspace(min(sorted_data), max(sorted_data), threshold_steps)

    # Calculate counts above threshold for each value
    pos_above_threshold = [np.sum(pos_ratios_before >= t) for t in thresholds]
    neg_above_threshold = [np.sum(neg_ratios_before >= t) for t in thresholds]

    # Calculate percentages
    pos_percent = [count / len(pos_ratios_before) * 100 for count in pos_above_threshold]
    neg_percent = [count / len(neg_ratios_before) * 100 for count in neg_above_threshold]

    # Plot counts above threshold
    ax1.plot(thresholds, pos_above_threshold, 'g-', label='Positive Sequences', linewidth=2)
    ax1.plot(thresholds, neg_above_threshold, 'b-', label='Negative Sequences', linewidth=2)
    ax1.set_title('Number of Sequences Above Threshold')
    ax1.set_xlabel('Threshold Value')
    ax1.set_ylabel('Count')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot percentage above threshold
    ax2.plot(thresholds, pos_percent, 'g-', label='Positive Sequences (%)', linewidth=2)
    ax2.plot(thresholds, neg_percent, 'b-', label='Negative Sequences (%)', linewidth=2)
    ax2.set_title('Percentage of Sequences Above Threshold')
    ax2.set_xlabel('Threshold Value')
    ax2.set_ylabel('Percentage (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Calculate and plot the difference between positive and negative percentages
    diff_percent = [p - n for p, n in zip(pos_percent, neg_percent)]
    ax3 = ax2.twinx()
    ax3.plot(thresholds, diff_percent, 'r-', label='Difference (Pos% - Neg%)', linewidth=1.5)
    ax3.set_ylabel('Percentage Difference (%)', color='r')
    ax3.tick_params(axis='y', labelcolor='r')
    ax3.legend(loc='lower right')

    # Find the threshold where the difference is maximum
    max_diff_idx = np.argmax(diff_percent)
    max_diff_threshold = thresholds[max_diff_idx]
    max_diff = diff_percent[max_diff_idx]

    # Mark the maximum difference point
    ax3.scatter([max_diff_threshold], [max_diff], color='r', s=100, zorder=5)
    ax3.annotate(f'Max Diff: {max_diff:.2f}%\nThreshold: {max_diff_threshold:.4f}',
                 xy=(max_diff_threshold, max_diff),
                 xytext=(max_diff_threshold + (max(thresholds) - min(thresholds)) * 0.05,
                         max_diff - 5),
                 arrowprops=dict(arrowstyle="->", color='r'))

    plt.tight_layout()
    plt.savefig(f"plots/ratio_figures/threshold_analysis_{dataset_type}.png", dpi=dpi)
    plt.show()
    pass


def display_ratio_figures_kde(df_bld, positive_seqs, neg_seqs, aaseq_to_ratio, dataset_type, dpi=600):
    # Pick 5 random patients
    random_patients = np.random.choice(df_bld['patient_id'].unique(), size=5, replace=False)
    # Pick random sequences from neg_seqs in the length of positive_seqs
    neg_seqs = np.random.choice(neg_seqs, size=min(len(neg_seqs), len(positive_seqs)), replace=False)

    # Prepare data for both plots
    ratios_before = {}
    ratios_after = {}

    for patient_id in random_patients:
        aaseqs = df_bld[df_bld['patient_id'] == patient_id]['AASeq'].unique()
        ratios_before[patient_id] = aaseq_to_ratio(aaseqs, dont_use_function=True)  # before applying f
        ratios_after[patient_id] = aaseq_to_ratio(aaseqs, dont_use_function=False)  # after applying f

    # Ratios for positive sequences
    pos_ratios_before = aaseq_to_ratio(positive_seqs, dont_use_function=True)
    pos_ratios_after = aaseq_to_ratio(positive_seqs, dont_use_function=False)

    # Ratios for negative sequences
    neg_ratios_before = aaseq_to_ratio(neg_seqs, dont_use_function=True)
    neg_ratios_after = aaseq_to_ratio(neg_seqs, dont_use_function=False)

    # Plotting (high-res)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey='row', dpi=dpi)

    # --- Top row: Random patients ---
    # Normalized KDE for top-left plot
    for pid in random_patients:
        kde = sns.kdeplot(ratios_before[pid], ax=axes[0, 0], label=f"Patient {pid}", common_norm=False)
        # Get the y data for the last line that was plotted
        line = axes[0, 0].get_lines()[-1]
        y_data = line.get_ydata()
        # Normalize
        line.set_ydata(y_data / np.max(y_data))

    axes[0, 0].set_title("KDE of Ratios for 5 Patients (BEFORE f)")
    axes[0, 0].set_xlabel("Ratios")
    axes[0, 0].set_ylabel("Normalized Density")
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1.05)  # Set y-limit to ensure max is 1

    # Normalized KDE for top-right plot
    for pid in random_patients:
        kde = sns.kdeplot(ratios_after[pid], ax=axes[0, 1], label=f"Patient {pid}", common_norm=False)
        # Get the y data for the last line that was plotted
        line = axes[0, 1].get_lines()[-1]
        y_data = line.get_ydata()
        # Normalize
        line.set_ydata(y_data / np.max(y_data))

    axes[0, 1].set_title("KDE of f(Ratios) for 5 Patients (AFTER f)")
    axes[0, 1].set_xlabel("f(Ratios)")
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1.05)  # Set y-limit to ensure max is 1

    # --- Bottom row: Positive sequences and negatives ---
    # Normalized KDE for bottom-left plot
    kde_pos = sns.kdeplot(pos_ratios_before, ax=axes[1, 0], color='tab:green', label="Positive Sequences",
                          common_norm=False)
    line_pos = axes[1, 0].get_lines()[-1]
    y_data_pos = line_pos.get_ydata()
    line_pos.set_ydata(y_data_pos / np.max(y_data_pos))

    kde_neg = sns.kdeplot(neg_ratios_before, ax=axes[1, 0], color='tab:blue', linestyle='--',
                          label="Negative Sequences", common_norm=False)
    line_neg = axes[1, 0].get_lines()[-1]
    y_data_neg = line_neg.get_ydata()
    line_neg.set_ydata(y_data_neg / np.max(y_data_neg))

    axes[1, 0].set_title("KDE of Ratios for Positive Sequences (BEFORE f)")
    axes[1, 0].set_xlabel("Ratios")
    axes[1, 0].set_ylabel("Normalized Density")
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1.05)  # Set y-limit to ensure max is 1

    # Calculate simple means directly
    pos_mean = np.mean(pos_ratios_before)
    neg_mean = np.mean(neg_ratios_before)

    # Add vertical lines at the means
    axes[1, 0].axvline(x=pos_mean, color='tab:green', linestyle='-', alpha=0.7,
                       label=f"Pos Mean: {pos_mean:.3f}")
    axes[1, 0].axvline(x=neg_mean, color='tab:blue', linestyle='-', alpha=0.7,
                       label=f"Neg Mean: {neg_mean:.3f}")
    axes[1, 0].legend()

    # Normalized KDE for bottom-right plot
    kde_pos = sns.kdeplot(pos_ratios_after, ax=axes[1, 1], color='tab:green', label="Positive Sequences",
                          common_norm=False)
    line_pos = axes[1, 1].get_lines()[-1]
    y_data_pos = line_pos.get_ydata()
    line_pos.set_ydata(y_data_pos / np.max(y_data_pos))

    kde_neg = sns.kdeplot(neg_ratios_after, ax=axes[1, 1], color='tab:blue', linestyle='--', label="Negative Sequences",
                          common_norm=False)
    line_neg = axes[1, 1].get_lines()[-1]
    y_data_neg = line_neg.get_ydata()
    line_neg.set_ydata(y_data_neg / np.max(y_data_neg))

    axes[1, 1].set_title("KDE of f(Ratios) for Positive Sequences (AFTER f)")
    axes[1, 1].set_xlabel("f(Ratios)")
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1.05)  # Set y-limit to ensure max is 1

    plt.tight_layout()
    os.makedirs("plots/ratio_figures", exist_ok=True)
    plt.savefig(f"plots/ratio_figures/ratios_{dataset_type}.png", dpi=dpi)
    plt.show()


# Helper function to create embeddings plot
def create_embedding_plot(trained_model, sequences, labels, colors, model_config, device, title_suffix="", verbose=True):
    if len(sequences) == 0:
        print(f"Warning: No sequences found for {title_suffix}")
        return None

    trained_model.to(device)
    trained_model.eval()
    with torch.no_grad():
        embeddings = trained_model.get_embeddings(sequences).cpu()

    # Convert to numpy for sklearn compatibility
    embeddings_np = embeddings.numpy()

    # Apply dimensionality reduction techniques
    if verbose:
        print(f"Applying PCA for {title_suffix}...")
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(embeddings_np)

    if verbose:
        print(f"Applying t-SNE for {title_suffix}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings_np) - 1), n_iter=1000)
    tsne_result = tsne.fit_transform(embeddings_np)

    if verbose:
        print(f"Applying UMAP for {title_suffix}...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings_np) - 1),
                             min_dist=0.1)
    umap_result = umap_reducer.fit_transform(embeddings_np)

    # Create the figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot PCA
    for i, (label, color) in enumerate(zip(np.unique(labels), colors)):
        mask = labels == label
        axes[0].scatter(pca_result[mask, 0], pca_result[mask, 1], c=color, alpha=0.6, s=20, label=label)
    axes[0].set_title(f'PCA - {title_suffix}\nExplained Variance: {pca.explained_variance_ratio_.sum():.3f}')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot t-SNE
    for i, (label, color) in enumerate(zip(np.unique(labels), colors)):
        mask = labels == label
        axes[1].scatter(tsne_result[mask, 0], tsne_result[mask, 1], c=color, alpha=0.6, s=20, label=label)
    axes[1].set_title(f't-SNE - {title_suffix}')
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot UMAP
    for i, (label, color) in enumerate(zip(np.unique(labels), colors)):
        mask = labels == label
        axes[2].scatter(umap_result[mask, 0], umap_result[mask, 1], c=color, alpha=0.6, s=20, label=label)
    axes[2].set_title(f'UMAP - {title_suffix}')
    axes[2].set_xlabel('UMAP 1')
    axes[2].set_ylabel('UMAP 2')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    os.makedirs(f"plots/embedding_mapping/model_config_{model_config}", exist_ok=True)
    plt.savefig(f"plots/embedding_mapping/model_config_{model_config}/{title_suffix.lower().replace(' ', '_')}.png")
    plt.show()

    # Print some statistics
    if verbose:
        print(f"\nEmbedding Statistics for {title_suffix}:")
        print(f"Original embedding dimension: {embeddings_np.shape[1]}")
        print(f"Number of samples: {embeddings_np.shape[0]}")
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")

    return {
        'pca': pca_result,
        'tsne': tsne_result,
        'umap': umap_result,
        'labels': labels
    }


def plot_embedding_mappings(trained_model, df_bld, valid_pos_seqs, test_pos_seqs, valid_patient_ids, test_patient_ids, model_config,
                            max_samples_per_patient=1000, max_samples_per_class=5000, device='cuda', use_only_val=True, verbose=True):
    # Convert to sets for faster lookup
    pos_seqs_set = set(valid_pos_seqs) if use_only_val else set(np.concatenate([valid_pos_seqs, test_pos_seqs]))

    # Select 2 patients from validation and 2 from test
    selected_valid_patients = np.random.choice(valid_patient_ids, size=min(2, len(valid_patient_ids)), replace=False)
    selected_test_patients = np.random.choice(test_patient_ids, size=min(2, len(test_patient_ids)), replace=False) if not use_only_val else []

    all_selected_patients = list(selected_valid_patients) + list(selected_test_patients)

    results = {}

    # Create plots for each selected patient (plots 1-4)
    for i, patient_id in enumerate(all_selected_patients):
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Processing Patient {patient_id} (Plot {i + 1}/4)")
            print(f"{'=' * 50}")

        # Get all sequences for this patient
        patient_seqs = df_bld[df_bld['patient_id'] == patient_id]['AASeq'].unique()

        # Separate positive and negative sequences
        pos_patient_seqs = np.array(list(set(patient_seqs) & pos_seqs_set))
        neg_patient_seqs = np.array(list(set(patient_seqs) - pos_seqs_set))

        # Sample sequences (adjust sample size based on availability)
        pos_sample_size = min(max_samples_per_patient, len(pos_patient_seqs))
        neg_sample_size = min(max_samples_per_patient, len(neg_patient_seqs))

        if pos_sample_size > 0:
            pos_patient_seqs_sampled = np.random.choice(pos_patient_seqs, size=pos_sample_size, replace=False)
        else:
            pos_patient_seqs_sampled = np.array([])

        if neg_sample_size > 0:
            neg_patient_seqs_sampled = np.random.choice(neg_patient_seqs, size=neg_sample_size, replace=False)
        else:
            neg_patient_seqs_sampled = np.array([])

        # Combine sequences and create labels
        combined_seqs = np.concatenate([pos_patient_seqs_sampled, neg_patient_seqs_sampled])
        to_concatenate = [['Positive'] * len(pos_patient_seqs_sampled), ['Negative'] * len(neg_patient_seqs_sampled)]
        labels = np.concatenate(to_concatenate)

        if len(combined_seqs) > 0:
            result = create_embedding_plot(
                trained_model,
                combined_seqs,
                labels,
                ['red', 'blue'],
                model_config,
                device,
                f"Patient {patient_id}",
                verbose
            )
            results[f'patient_{patient_id}'] = result
        else:
            print(f"Warning: No sequences found for patient {patient_id}")

    # Create the 5th plot: All positive and negative sequences from all patients
    if verbose:
        print(f"\n{'=' * 50}")
        print(f"Processing All Patients Combined (Plot 5/5)")
        print(f"{'=' * 50}")

    # Get all sequences from all patients
    all_patient_ids = np.concatenate([valid_patient_ids, test_patient_ids]) if not use_only_val else valid_patient_ids
    all_sequences = []

    for patient_id in all_patient_ids:
        patient_seqs = df_bld[df_bld['patient_id'] == patient_id]['AASeq'].unique()
        all_sequences.extend(patient_seqs)

    all_sequences = np.array(list(set(all_sequences)))  # Remove duplicates

    # Separate into positive and negative
    all_pos_seqs = np.array(list(set(all_sequences) & pos_seqs_set))
    all_neg_seqs = np.array(list(set(all_sequences) - pos_seqs_set))

    to_sample = min(max_samples_per_class, len(all_pos_seqs))
    if len(all_pos_seqs) > to_sample:
        all_pos_seqs = np.random.choice(all_pos_seqs, size=to_sample, replace=False)

    if len(all_neg_seqs) > to_sample:
        all_neg_seqs = np.random.choice(all_neg_seqs, size=to_sample, replace=False)

    # Combine all sequences and create labels
    all_combined_seqs = np.concatenate([all_pos_seqs, all_neg_seqs])
    to_concat = [['Positive'] * len(all_pos_seqs), ['Negative'] * len(all_neg_seqs)]
    all_labels = np.concatenate(to_concat)

    result = create_embedding_plot(
        trained_model,
        all_combined_seqs,
        all_labels,
        ['red', 'blue'],
        model_config,
        device,
        "All Patients Combined",
        verbose
    )
    results['all_patients'] = result
    return results
