import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import wandb
from cache_handler import get_model_config_str
import pandas as pd


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


def plot_output_distributions_per_patient(trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
                                          test_masks, valid_masks, positive_seqs, df_bld,
                                          df_hlt, model_type, log_wandb, args, device='cuda'):
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
    plt.savefig(f"plots/{model_type}_model/dist_model_output/comprehensive_distribution_per_patients_{get_model_config_str(args)}.png")
    plt.tight_layout()

    # save the figure in wandb:
    if log_wandb:
        wandb.log({"output_distributions_per_patient": wandb.Image(plt)})
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
    test_masks = np.concatenate((test_masks, valid_masks), axis=0)

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


def display_ratio_figures(df_bld, positive_seqs, aaseq_to_ratio, dataset_type, dpi=600):
    # Pick 5 random patients
    random_patients = np.random.choice(df_bld['patient_id'].unique(), size=5, replace=False)

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

    # Plotting (high-res)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12), sharey='row', dpi=dpi)

    # --- Top row: Random patients ---
    for pid in random_patients:
        sns.kdeplot(ratios_before[pid], ax=axes[0, 0], label=f"Patient {pid}")
    axes[0, 0].set_title("KDE of Ratios for 5 Patients (BEFORE f)")
    axes[0, 0].set_xlabel("Ratios")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].legend()

    for pid in random_patients:
        sns.kdeplot(ratios_after[pid], ax=axes[0, 1], label=f"Patient {pid}")
    axes[0, 1].set_title("KDE of f(Ratios) for 5 Patients (AFTER f)")
    axes[0, 1].set_xlabel("f(Ratios)")
    axes[0, 1].legend()

    # --- Bottom row: Positive sequences ---
    sns.kdeplot(pos_ratios_before, ax=axes[1, 0], color='tab:green')
    axes[1, 0].set_title("KDE of Ratios for Positive Sequences (BEFORE f)")
    axes[1, 0].set_xlabel("Ratios")
    axes[1, 0].set_ylabel("Density")

    sns.kdeplot(pos_ratios_after, ax=axes[1, 1], color='tab:green')
    axes[1, 1].set_title("KDE of f(Ratios) for Positive Sequences (AFTER f)")
    axes[1, 1].set_xlabel("f(Ratios)")

    plt.tight_layout()
    os.makedirs("plots/ratio_figures", exist_ok=True)
    plt.savefig(f"plots/ratio_figures/ratios_{dataset_type}.png", dpi=dpi)
    plt.show()


# def kde_normalizer(kde, max_density=1.0):
#     """
#     Normalize KDE plot to a maximum density.
#     """
#     # Get current axis limits
#     x, y = kde.get_lines()[0].get_data()
#
#     # Get current maximum y value
#     current_max = np.max(y)
#
#     # Calculate scaling factor
#     scale_factor = max_density / current_max if current_max > 0 else 1.0
#
#     # Update y values with scaling
#     for line in kde.get_lines():
#         x, y = line.get_data()
#         line.set_ydata(y * scale_factor)
