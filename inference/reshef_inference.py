import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from cache_handler import get_model_config_str
import seaborn as sns


def reshef_inference(train_pos_seqs, train_neg_seqs, valid_pos_seqs, valid_neg_seqs, reshef_negative_part, args, to_save_train_data=True):
    # Load the full data and labels
    model_config_string = get_model_config_str(args)
    if reshef_negative_part > 0:
        reshef_cache_folder = os.path.join("cache", "reshef_inference", f"partition_{reshef_negative_part}", model_config_string)
    else:
        reshef_cache_folder = os.path.join("cache", "reshef_inference", model_config_string)
    full_data = np.load(os.path.join(reshef_cache_folder, "full_data.npy"))
    full_labels = np.load(os.path.join(reshef_cache_folder, "full_labels.npy"))

    # Load the parameters for Reshef inference
    n_swap, negative_part = load_parameters(reshef_cache_folder)

    assert reshef_negative_part == negative_part, f"Reshef negative part mismatch: {reshef_negative_part} != {negative_part}"
    assert len(full_data) > 0, "No data found for Reshef inference."
    assert len(full_data) == len(full_labels), "Mismatch between full data and labels lengths."

    # Find all model output files per epoch
    model_output_files = [f for f in os.listdir(reshef_cache_folder) if f.startswith("model_outputs_epoch-") and f.endswith(".npy")]
    model_output_files = sorted(model_output_files, key=lambda x: int(x.split('epoch-')[1].split('.npy')[0]))  # Sort by epoch number
    assert model_output_files, "No model output files found for Reshef inference: " + reshef_cache_folder

    # Load all model outputs
    model_outputs = []
    for model_output_file in model_output_files:
        model_output_path = os.path.join(reshef_cache_folder, model_output_file)
        epoch_outputs = np.load(model_output_path)
        model_outputs.append(epoch_outputs)
    num_epochs = len(model_outputs)

    labels = np.array([[1, 0] if label == 0 else [0, 1] for label in full_labels])

    prob_list = []
    for model_output in model_outputs:
        # apply softmax to the model output logits
        # Assuming model_output is a numpy array of shape (num_samples, 2 classes)
        model_output = torch.tensor(model_output, dtype=torch.float32)
        model_output = torch.softmax(model_output, dim=1)  # Apply log softmax to logits
        # Convert logits to probabilities
        prob = probability_for_confidence(model_output, torch.tensor(labels))
        prob_list.append(prob)

    confidence, variability = probability_list_to_confidence_and_var(prob_list, len(full_data), num_epochs)
    confidence, variability = confidence.detach().numpy(), variability.detach().numpy()

    # Create indices for the scatter plot data
    inds = [0, len(train_pos_seqs[:n_swap]), len(train_pos_seqs[n_swap:]),
            len(train_neg_seqs[:n_swap]), len(train_neg_seqs[n_swap:]),
            len(valid_pos_seqs), len(valid_neg_seqs)]
    for i in range(1, len(inds)):
        inds[i] = inds[i - 1] + inds[i]

    # Function for picking random indices for the scatter plot
    rng = np.random.default_rng(seed=42)  # For reproducibility
    def pick_random_indices(start, end, amount=None):
        """Pick random indices from the given range."""
        indices_range = np.arange(start, end)
        if amount is None:
            return indices_range
        else:
            return rng.choice(indices_range, size=min(amount, len(indices_range)), replace=False)

    plot_var_conv(variability, confidence, inds, pick_random_indices)

    if to_save_train_data:
        save_data_for_training(full_data, confidence, inds, reshef_cache_folder, pick_random_indices)
    pass


def probability_for_confidence(y_pred, y_true):
    """
    Calculates the probability that the predicted class is correct, given the predicted class probabilities and the true class labels.

    Args:
        y_pred (torch.Tensor): Tensor of shape (batch_size, num_classes) with predicted class probabilities.
        y_true (torch.Tensor): Tensor of shape (batch_size, num_classes) with true class labels.

    Returns:
        torch.Tensor: Tensor of shape (batch_size,) with the probability that the predicted class is correct for each sample.
    """
    return torch.sum(y_pred * y_true, axis=1)


def probability_list_to_confidence_and_var(prob_list, n_obs, epoch_num):
    """
    Calculates the confidence and variability of the predicted class probabilities, given a list of predicted class probabilities.

    Args:
        prob_list (list): List of tensors of shape (batch_size,) with predicted class probabilities.
        n_obs (int): Number of observations.
        epoch_num (int): Number of epochs.

    Returns:
        tuple: Tensor of shape (batch_size,) with the confidence of the predicted class probabilities, and a tensor of shape (batch_size,) with the variability of the predicted class probabilities.
    """
    confidence = torch.zeros(n_obs)
    for i in range(epoch_num):
        confidence += (prob_list[i])
    confidence = confidence / epoch_num
    variability = torch.zeros(n_obs)
    for i in range(epoch_num):
        variability += torch.square(confidence - (prob_list[i]))
    variability = variability / epoch_num
    variability = torch.sqrt(variability)
    return confidence, variability


def plot_var_conv(variability, confidence, inds, pick_random_indices, reshef_cache_folder=None):
    # === PLOT 1: Originally Positive (Train Positive + Miss-labeled as Negative) ===
    chosen_pos_indices = pick_random_indices(inds[1], inds[2])  # Train Positive
    chosen_miss_neg_indices = pick_random_indices(inds[0], inds[1])  # Miss-labeled (negative originally, labeled positive)

    x_pos = variability[chosen_pos_indices]
    y_pos = confidence[chosen_pos_indices]

    x_miss_neg = variability[chosen_miss_neg_indices]
    y_miss_neg = confidence[chosen_miss_neg_indices]

    x1 = np.concatenate([x_pos, x_miss_neg])
    y1 = np.concatenate([y_pos, y_miss_neg])
    hue1 = ['Train Positive'] * len(x_pos) + ['Miss-labeled'] * len(x_miss_neg)

    g1 = sns.jointplot(
        x=x1,
        y=y1,
        hue=hue1,
        kind="scatter",
        alpha=0.5,
        marginal_kws=dict(common_norm=False, fill=True, alpha=0.4),
        palette={'Train Positive': 'blue', 'Miss-labeled': 'magenta'},
        height=8
    )

    g1.fig.suptitle("Originally Negative Samples", fontsize=14)
    g1.ax_joint.set_xlabel('Variability', fontsize=12)
    g1.ax_joint.set_ylabel('Confidence', fontsize=12)
    g1.fig.tight_layout()
    g1.fig.subplots_adjust(top=0.95)
    if reshef_cache_folder:
        plot_path = os.path.join(reshef_cache_folder, "reshef_inference_plot_positive_labels.png")
        g1.savefig(plot_path)
        print(f"Saved Reshef inference plot at: {plot_path}")
    plt.show()

    # === PLOT 2: Originally Negative (Train Negative + Miss-labeled) ===
    chosen_neg_indices = pick_random_indices(inds[3], inds[4])  # Train Negative
    chosen_miss_pos_indices = pick_random_indices(inds[2], inds[3])  # Miss-labeled (positive originally, labeled negative)

    x_neg = variability[chosen_neg_indices]
    y_neg = confidence[chosen_neg_indices]

    x_miss_pos = variability[chosen_miss_pos_indices]
    y_miss_pos = confidence[chosen_miss_pos_indices]

    x2 = np.concatenate([x_neg, x_miss_pos])
    y2 = np.concatenate([y_neg, y_miss_pos])
    hue2 = ['Train Negative'] * len(x_neg) + ['Miss-labeled'] * len(x_miss_pos)

    g2 = sns.jointplot(
        x=x2,
        y=y2,
        hue=hue2,
        kind="scatter",
        alpha=0.5,
        marginal_kws=dict(common_norm=False, fill=True, alpha=0.4),
        palette={'Train Negative': 'red', 'Miss-labeled': 'green'},
        height=8
    )

    g2.fig.suptitle("Originally Positive Samples", fontsize=14)
    g2.ax_joint.set_xlabel('Variability', fontsize=12)
    g2.ax_joint.set_ylabel('Confidence', fontsize=12)
    g2.fig.tight_layout()
    g2.fig.subplots_adjust(top=0.95)
    # save plot
    if reshef_cache_folder:
        plot_path = os.path.join(reshef_cache_folder, "reshef_inference_plot_negative_labels.png")
        g2.savefig(plot_path)
        print(f"Saved Reshef inference plot at: {plot_path}")
    plt.show()


def old_plot_var_conf(variability, confidence, inds, labels, pick_random_indices):

    # Create a scatter plot
    plt.figure(figsize=(10, 6))
    # chosen_indices = np.concatenate([pick_random_indices(inds[1], inds[2]), pick_random_indices(inds[3], inds[4])])
    # plt.scatter(variability[chosen_indices], confidence[chosen_indices], color='blue', label='Train Data points', alpha=0.5)
    chosen_indices = pick_random_indices(inds[1], inds[2])
    plt.scatter(variability[chosen_indices], confidence[chosen_indices], color='blue', label='Train Positive points', alpha=0.5)
    chosen_indices = pick_random_indices(inds[3], inds[4])  # Negative samples
    plt.scatter(variability[chosen_indices], confidence[chosen_indices], color='red', label='Train Negative points', alpha=0.5)
    # Miss-labeled data points (positives originally)
    chosen_indices = pick_random_indices(inds[0], inds[1])
    colors = ["magenta" if label[0] == 0 else "green" for label in labels[chosen_indices]]
    plt.scatter(variability[chosen_indices], confidence[chosen_indices], color=colors, label='Miss-labeled Data points (original pos)', alpha=0.5)
    # Miss-labeled data points (negatives originally)
    chosen_indices = pick_random_indices(inds[2], inds[3])
    colors2 = ["magenta" if label[0] == 0 else "green" for label in labels[chosen_indices]]
    plt.scatter(variability[chosen_indices], confidence[chosen_indices], color=colors2, label='Miss-labeled Data points (original neg)', alpha=0.5)
    # # Validation data points
    # chosen_indices = pick_random_indices(inds[4], inds[6])
    # plt.scatter(variability[chosen_indices], confidence[chosen_indices], color='purple', label='Validation Data points', alpha=0.5)

    # plt.scatter(variability[20:-20], confidence[20:-20], color='blue', label='Data points')
    # colors = ["magenta" if label[0] == 0 else "green" for label in labels[:20]]
    # plt.scatter(variability[:20], confidence[:20], color=colors, label='Miss-labeled Data points (original pos)')
    # colors2 = ["magenta" if label[0] == 0 else "green" for label in labels[-20:]]
    # plt.scatter(variability[-20:], confidence[-20:], color=colors2, label='Miss-labeled Data points (original neg)')
    plt.title('Scatter Plot of Confidence vs. Variability')
    plt.xlabel('Variability')
    plt.ylabel('Confidence')
    plt.legend()
    plt.show()


def save_data_for_training(full_data, confidence, inds, reshef_cache_folder, pick_random_indices):
    chosen_indices = pick_random_indices(inds[3], inds[4])  # Negative samples
    chosen_indices2 = pick_random_indices(inds[2], inds[3])  # Miss-labeled data points (positive originally)

    # confidence_red = confidence[chosen_indices]
    confidence_green = confidence[chosen_indices2]
    confidence_green_p = np.percentile(confidence_green, 90)  # get chosen_indices_green 90% percentile

    conf_mask = np.where(confidence[chosen_indices] > confidence_green_p, True,
                         False)  # Get boolean array where confidence is greater than 95% percentile
    neg_seqs_to_train = full_data[chosen_indices][conf_mask]

    seqs_return_to_positives_green = full_data[chosen_indices2]
    all_blue = full_data[inds[1]:inds[2]]  # Train Positive
    all_blus_plus_seqs_return_to_positives_green = np.concatenate((all_blue, seqs_return_to_positives_green),
                                                                  axis=0)

    # save to npz file
    np.savez(os.path.join(reshef_cache_folder, "reshef_inference_data.npz"),
             neg_seqs_to_train=neg_seqs_to_train,
             pos_seqs_to_train=all_blus_plus_seqs_return_to_positives_green)


def load_parameters(reshef_cache_folder, n_swap=100, negative_part=0):
    # try to load n_swap.npy
    n_swap_path = os.path.join(reshef_cache_folder, "n_swap.npy")
    if os.path.exists(n_swap_path):
        n_swap = np.load(n_swap_path).item()
    parameters_path = os.path.join(reshef_cache_folder, "parameters.npz")
    if os.path.exists(parameters_path):
        parameters = np.load(parameters_path)
        n_swap = parameters['n_swap']
        negative_part = parameters['reshef_negative_part']
    return n_swap, negative_part
