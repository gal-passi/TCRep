import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb


def kde_normalizer(kde):
    for line in kde.get_lines():
        y_data = line.get_ydata()
        line.set_ydata(y_data / y_data.max())  # Normalize to max value of 1


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
    plt.savefig(f"plots/{model_type}_model/dist_model_output/comprehensive_distribution.png")
    plt.tight_layout()

    # save the figure in wandb:
    if log_wandb:
        wandb.log({"output_distributions": wandb.Image(plt)})
    else:
        plt.show()


def plot_output_distributions_per_patient(trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
                                          test_masks, valid_masks, positive_seqs, df_bld,
                                          df_hlt, model_type, log_wandb, device='cuda'):
    """
    Create a comprehensive visualization of model output distributions across validation,
    training, and healthy patient sets.
    Parameters:
    - trained_model: The trained neural network model
    - Various data-related parameters to extract sequences and patient sets
    """
    # Set up the figure with three subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f"Model Output Distributions for {model_type} Model", fontsize=16)

    # Create new test_inds which is all test_patient_inds and one from valid_patient_inds
    test_inds = np.concatenate([test_patient_inds, [valid_patient_inds[0]]])
    # And create a new test_mask which contains both
    test_masks = np.concatenate((test_masks, valid_masks[0].reshape(1, -1)), axis=0)

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
        kde = sns.kdeplot(pos_probs, ax=ax1, color=test_pos_colors[i], label=f'Pos Set {i}')
        kde_normalizer(kde)
        kde = sns.kdeplot(neg_probs, ax=ax1, color=test_neg_colors[i], label=f'Neg Set {i}', common_norm=True)
        kde_normalizer(kde)
    ax1.set_xlabel("Predicted Probability for Positive Class")
    ax1.set_ylabel("Density")
    ax1.set_ylim(0, 1.1)
    ax1.legend()

    # Subplot 2: Healthy Patients Distributions
    ax2.set_title("Healthy vs Ill Patients")
    healthy_patients = df_hlt["patient_id"].unique()
    np.random.shuffle(healthy_patients)
    for i, patient_ind in enumerate(test_inds):
        patient = healthy_patients[i]
        healthy_seqs = df_hlt.loc[df_hlt["patient_id"] == patient, "AASeq"].values
        healthy_seqs = np.unique(healthy_seqs)
        disease_seqs = df_bld.loc[df_bld["patient_id"] == unique_patient_ids[patient_ind], "AASeq"].values
        disease_seqs = np.unique(disease_seqs)
        # Get model outputs
        trained_model.to(device)
        trained_model.eval()
        with torch.no_grad():
            healthy_logits = trained_model(healthy_seqs)
            disease_logits = trained_model(disease_seqs)
        # Convert to probabilities
        healthy_probs = torch.softmax(healthy_logits, dim=1)[:, 1].cpu().numpy()
        disease_probs = torch.softmax(disease_logits, dim=1)[:, 1].cpu().numpy()
        # Plot KDE for healthy patient samples
        kde = sns.kdeplot(healthy_probs, ax=ax2, color=healthy_colors[i], label=f'Healthy Set {i}', common_norm=True)
        kde_normalizer(kde)
        kde = sns.kdeplot(disease_probs, ax=ax2, color=test_colors[i], label=f'Ill Set {i}', common_norm=True)
        kde_normalizer(kde)
    ax2.set_xlabel("Predicted Probability for Positive Class")
    ax2.set_ylabel("Density")
    ax2.set_ylim(0, 1.1)
    ax2.legend()
    # Save the plot
    os.makedirs(f"plots/{model_type}_model/dist_model_output", exist_ok=True)
    plt.savefig(f"plots/{model_type}_model/dist_model_output/comprehensive_distribution.png")
    plt.tight_layout()

    # save the figure in wandb:
    if log_wandb:
        wandb.log({"output_distributions_per_patient": wandb.Image(plt)})
    else:
        plt.show()
