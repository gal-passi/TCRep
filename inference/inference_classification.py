import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from models.cvc_cacheing_model import CVCCachingModel
from cache_handler import get_model_config_str
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


INFERENCE_BASE_PLOT_DIR = "plots/cvc_model/inference_plots/"
INFERENCE_CONFUSION_MATRIX_DIR = os.path.join(INFERENCE_BASE_PLOT_DIR, "confusion_matrices/")
INFERENCE_VECTOR_PLOTS_DIR = os.path.join(INFERENCE_BASE_PLOT_DIR, "vector_plots/")


def create_binned_disease_probs(patient_ratio, disease_probs, num_bins=20):
    """
    Create bins based on patient_ratio and compute average disease_probs for each bin.

    Parameters:
    patient_ratio (ndarray): Array of patient ratios to determine binning
    disease_probs (ndarray): Array of disease probabilities (same shape as patient_ratio)
    num_bins (int): Number of bins to create (default: 20)

    Returns:
    ndarray: Array of size num_bins with average disease probabilities for each bin
    """

    # Flatten arrays in case they're multi-dimensional
    patient_ratio_flat = patient_ratio.flatten()
    disease_probs_flat = disease_probs.flatten()

    # Create bin edges based on patient_ratio range
    min_ratio = np.min(patient_ratio_flat)
    max_ratio = np.max(patient_ratio_flat)
    bin_edges = np.linspace(min_ratio, max_ratio, num_bins + 1)

    # Assign each patient_ratio to a bin
    bin_indices = np.digitize(patient_ratio_flat, bin_edges) - 1
    # Handle edge case where values equal to max fall into bin num_bins
    bin_indices = np.clip(bin_indices, 0, num_bins - 1)

    # Initialize result array
    binned_avg_probs = np.zeros(num_bins)

    # Calculate average disease_probs for each bin
    for i in range(num_bins):
        mask = bin_indices == i
        if np.any(mask):  # If there are values in this bin
            binned_avg_probs[i] = np.mean(disease_probs_flat[mask])
        else:  # If bin is empty, you might want to set to 0 or NaN
            binned_avg_probs[i] = 0  # or np.nan

    return binned_avg_probs


def calc_patient_vectors(df, caching_model, patient_inds, vector_representation_bins, add_ratio_to_vector,
                         aaseq_to_ratio, possible_seqs=None, unique_patient_ids=None):
    patient_vectors = []
    for patient_ind in patient_inds:
        # Extract and process the patient sequences
        if unique_patient_ids is not None and possible_seqs is not None:
            patient_seqs = df.loc[df["patient_id"] == unique_patient_ids[patient_ind], "AASeq"].values
            patient_seqs = np.array([x for x in np.unique(patient_seqs) if x in possible_seqs])
        else:
            patient_seqs = df.loc[df["patient_id"] == patient_ind, "AASeq"].values

        # Get model outputs
        with torch.no_grad():
            disease_logits = caching_model(patient_seqs)

        # Convert to probabilities
        disease_probs = torch.softmax(disease_logits, dim=1)[:, 1].cpu().numpy()

        # bin using np into x bins (min value is 0 and max is 1)
        disease_probs_dig = np.digitize(disease_probs, bins=np.linspace(0, 1, vector_representation_bins + 1)) - 1
        # Vectorized bin counting
        patient_vector = np.bincount(disease_probs_dig, minlength=10).astype(np.float32)
        patient_vector /= patient_vector.sum()
        # patient_vector /= patient_vector.max()

        if add_ratio_to_vector:
            patient_ratio = aaseq_to_ratio(patient_seqs, dont_use_function=True).values
            patient_vector_ratio = create_binned_disease_probs(patient_ratio, disease_probs, num_bins=vector_representation_bins)
            patient_vector_ratio /= patient_vector_ratio.max()
            patient_vectors.append(np.concatenate([patient_vector, patient_vector_ratio], axis=0))
            pass
            # bin patient ratio
            # patient_ratio = aaseq_to_ratio(patient_seqs, dont_use_function=True).values ** 0.2
            # patient_ratio = np.digitize(patient_ratio, bins=np.linspace(0, 1, vector_representation_bins + 1)) - 1
            # # Vectorized bin counting
            # patient_vector_ratio = np.bincount(patient_ratio, minlength=10).astype(np.float32)
            # patient_vector_ratio /= patient_vector_ratio.sum()
            # patient_vectors.append(np.concatenate([patient_vector, patient_vector_ratio], axis=0))
        else:
            patient_vectors.append(patient_vector)
    patient_vectors = np.array(patient_vectors)
    return patient_vectors


def inference_classification_model(trained_model, args, df_bld, df_hlt, test_patient_inds, valid_patient_inds, unique_patient_ids,
                                   valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs, aaseq_to_ratio, device,
                                   add_ratio_to_vector=False,
                                   vector_representation_bins=20, num_of_healthy_patients=68, num_of_healthy_test_patients=28):
    # make sure that plot dirs exists
    os.makedirs(INFERENCE_CONFUSION_MATRIX_DIR, exist_ok=True)
    os.makedirs(INFERENCE_VECTOR_PLOTS_DIR, exist_ok=True)

    # Creating a caching model of the trained model
    trained_model.eval()
    caching_model = CVCCachingModel(trained_model, args, device)
    caching_model.to(device)
    caching_model.eval()

    # patient vectors helping data
    possible_seqs = set(np.concatenate([valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs]))
    patient_valid_test_inds = np.concatenate([test_patient_inds, valid_patient_inds])

    # get the patient vectors
    patient_vectors = calc_patient_vectors(df_bld, caching_model, patient_valid_test_inds, vector_representation_bins,
                                           add_ratio_to_vector, aaseq_to_ratio, unique_patient_ids=unique_patient_ids, possible_seqs=possible_seqs)
    # patient_vectors = []
    # for patient_ind in patient_valid_test_inds:
    #     # Extract and process the patient sequences
    #     patient_seqs = df_bld.loc[df_bld["patient_id"] == unique_patient_ids[patient_ind], "AASeq"].values
    #     patient_seqs = np.array([x for x in np.unique(patient_seqs) if x in possible_seqs])
    #
    #     # Get model outputs
    #     with torch.no_grad():
    #         disease_logits = caching_model(patient_seqs)
    #
    #     # Convert to probabilities
    #     disease_probs = torch.softmax(disease_logits, dim=1)[:, 1].cpu().numpy()
    #
    #     # bin using np into x bins (min value is 0 and max is 1)
    #     disease_probs = np.digitize(disease_probs, bins=np.linspace(0, 1, vector_representation_bins + 1)) - 1
    #     # Vectorized bin counting
    #     patient_vector = np.bincount(disease_probs, minlength=10).astype(np.float32)
    #     patient_vector /= patient_vector.sum()
    #
    #     if add_ratio_to_vector:
    #         # bin patient ratio
    #         patient_ratio = aaseq_to_ratio(patient_seqs, dont_use_function=True).values ** 0.2
    #         patient_ratio = np.digitize(patient_ratio, bins=np.linspace(0, 1, vector_representation_bins + 1)) - 1
    #         # Vectorized bin counting
    #         patient_vector_ratio = np.bincount(patient_ratio, minlength=10).astype(np.float32)
    #         patient_vector_ratio /= patient_vector_ratio.sum()
    #         patient_vectors.append(np.concatenate([patient_vector, patient_vector_ratio], axis=0))
    #     else:
    #         patient_vectors.append(patient_vector)

    patient_vectors = np.array(patient_vectors)

    # healthy vectors helping data
    healthy_patients = df_hlt["patient_id"].unique()
    np.random.shuffle(healthy_patients)
    healthy_patients = healthy_patients[:num_of_healthy_patients]

    # get the healthy patient vectors
    # healthy_vectors = []
    # for patient_ind in healthy_patients:
    #     # Extract and process the patient sequences
    #     patient_seqs = df_hlt.loc[df_hlt["patient_id"] == patient_ind, "AASeq"].values
    #
    #     # Get model outputs
    #     with torch.no_grad():
    #         healthy_logits = caching_model(patient_seqs)
    #
    #     # Convert to probabilities
    #     healthy_probs = torch.softmax(healthy_logits, dim=1)[:, 1].cpu().numpy()
    #
    #     # bin using np into x bins (min value is 0 and max is 1)
    #     healthy_probs = np.digitize(healthy_probs, bins=np.linspace(0, 1, vector_representation_bins + 1)) - 1
    #     # Vectorized bin counting
    #     healthy_vector = np.bincount(healthy_probs, minlength=vector_representation_bins).astype(np.float32)
    #     healthy_vector /= healthy_vector.sum()
    #
    #     if add_ratio_to_vector:
    #         # bin patient ratio
    #         healthy_ratio = aaseq_to_ratio(patient_seqs, dont_use_function=True).values ** 0.2
    #         healthy_ratio = np.digitize(healthy_ratio, bins=np.linspace(0, 1, vector_representation_bins + 1)) - 1
    #         # Vectorized bin counting
    #         healthy_vector_ratio = np.bincount(healthy_ratio, minlength=10).astype(np.float32)
    #         healthy_vector_ratio /= healthy_vector_ratio.sum()
    #         healthy_vectors.append(np.concatenate([healthy_vector, healthy_vector_ratio], axis=0))
    #     else:
    #         healthy_vectors.append(healthy_vector)
    # healthy_vectors = np.array(healthy_vectors)

    healthy_vectors = calc_patient_vectors(df_hlt, caching_model, healthy_patients, vector_representation_bins,
                                           add_ratio_to_vector, aaseq_to_ratio)
    healthy_vectors, healthy_test_vectors = (healthy_vectors[:num_of_healthy_patients - num_of_healthy_test_patients],
                                             healthy_vectors[num_of_healthy_patients - num_of_healthy_test_patients:])

    total_cm_knn = []
    total_cm_rf = []

    # Do k-fold on patients and each time leave 2 different patients out (using combinations)
    possible_patients = list(range(len(patient_valid_test_inds)))
    for patient1, patient2 in itertools.combinations(possible_patients, 2):
        train_patients = np.delete(possible_patients, [patient1, patient2])
        test_patients = np.array([patient1, patient2])
        train_vectors = patient_vectors[train_patients]
        test_vectors = patient_vectors[test_patients]

        # Calculating statistics:
        # Combine training data
        X_train = np.vstack([train_vectors, healthy_vectors])
        y_train = np.hstack([np.ones(train_vectors.shape[0]), np.zeros(healthy_vectors.shape[0])])

        # Combine test data
        X_test = np.vstack([test_vectors, healthy_test_vectors])
        y_test = np.hstack([np.ones(test_vectors.shape[0]), np.zeros(healthy_test_vectors.shape[0])])

        # 1. KNN Classifier
        knn = KNeighborsClassifier(n_neighbors=2)  # You might want to tune this parameter
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        cm_knn = confusion_matrix(y_test, y_pred_knn)

        # 2. Balanced Random Forest Classifier
        # Using class_weight='balanced' to handle class imbalance
        rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        cm_rf = confusion_matrix(y_test, y_pred_rf)

        # Append the confusion matrices to the total list
        total_cm_knn.append(cm_knn)
        total_cm_rf.append(cm_rf)

    # Display final results
    def calculate_average_cm_rates(cm_list):
        """Calculate the average confusion matrix rates from a list of confusion matrices"""
        rates_list = []

        for cm in cm_list:
            # Calculate rates for each confusion matrix
            # cm structure: [[TN, FP], [FN, TP]]
            TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

            # Calculate rates (avoiding division by zero)
            total_actual_positive = TP + FN
            total_actual_negative = TN + FP

            if total_actual_positive > 0:
                TPR = TP / total_actual_positive  # Sensitivity/Recall
                FNR = FN / total_actual_positive  # Miss Rate
            else:
                TPR = FNR = 0

            if total_actual_negative > 0:
                TNR = TN / total_actual_negative  # Specificity
                FPR = FP / total_actual_negative  # Fall-out
            else:
                TNR = FPR = 0

            # Create rate matrix in same structure as confusion matrix
            rate_matrix = np.array([[TNR, FPR], [FNR, TPR]])
            rates_list.append(rate_matrix)

        # Calculate average rates across all folds
        rates_array = np.array(rates_list)
        avg_rates = np.mean(rates_array, axis=0)

        return np.round(avg_rates, 4)

    def display_average_results(total_cm_knn, total_cm_rf):
        """Display the average results from multiple k-fold runs"""
        # Calculate average confusion matrices
        avg_cm_knn = calculate_average_cm_rates(total_cm_knn)
        avg_cm_rf = calculate_average_cm_rates(total_cm_rf)

        # Number of folds
        n_folds = len(total_cm_knn)

        # Print results
        print(f"Average results across {n_folds} folds:")

        print("\nKNN - Average Confusion Matrix:")
        # rearrange to: [[TP, FP], [FN, TN]]
        avg_cm_knn_rearranged = np.array([[avg_cm_knn[1, 1], avg_cm_knn[1, 0]],
                                          [avg_cm_knn[0, 1], avg_cm_knn[0, 0]]])
        print(avg_cm_knn_rearranged)
        # print(f"Average Disease correctly classified: {avg_cm_knn[1, 1]:.2f}")
        # print(f"Average Healthy correctly classified: {avg_cm_knn[0, 0]:.2f}")

        print("\nRandom Forest - Average Confusion Matrix:")
        # rearrange to: [[TP, FP], [FN, TN]]
        avg_cm_rf_rearranged = np.array([[avg_cm_rf[1, 1], avg_cm_rf[1, 0]],
                                         [avg_cm_rf[0, 1], avg_cm_rf[0, 0]]])
        print(avg_cm_rf_rearranged)
        # print(f"Average Disease correctly classified: {avg_cm_rf[1, 1]:.2f}")
        # print(f"Average Healthy correctly classified: {avg_cm_rf[0, 0]:.2f}")

        # Visualize average confusion matrices
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        sns.heatmap(avg_cm_knn, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=['Predicted Healthy', 'Predicted Disease'],
                    yticklabels=['Actual Healthy', 'Actual Disease'])
        plt.title('Average Confusion Matrix - KNN')

        plt.subplot(1, 2, 2)
        sns.heatmap(avg_cm_rf, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=['Predicted Healthy', 'Predicted Disease'],
                    yticklabels=['Actual Healthy', 'Actual Disease'])
        plt.title('Average Confusion Matrix - Random Forest')

        plt.tight_layout()
        # save under plot dirs with name including the get_model_config_str
        model_config_str = get_model_config_str(args)
        plt.savefig(os.path.join(INFERENCE_CONFUSION_MATRIX_DIR, f"avg_confusion_matrices_{model_config_str}.png"))
        plt.show()

    # Execute the function to display average results
    display_average_results(total_cm_knn, total_cm_rf)

    # You could also calculate standard deviations to show variability across folds
    def calculate_std_cm_rates(cm_list):
        """Calculate standard deviation of confusion matrix rates across folds"""
        rates_list = []

        for cm in cm_list:
            # Calculate rates for each confusion matrix
            TN, FP, FN, TP = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

            total_actual_positive = TP + FN
            total_actual_negative = TN + FP

            if total_actual_positive > 0:
                TPR = TP / total_actual_positive
                FNR = FN / total_actual_positive
            else:
                TPR = FNR = 0

            if total_actual_negative > 0:
                TNR = TN / total_actual_negative
                FPR = FP / total_actual_negative
            else:
                TNR = FPR = 0

            rate_matrix = np.array([[TNR, FPR], [FNR, TPR]])
            rates_list.append(rate_matrix)

        rates_array = np.array(rates_list)
        std_rates = np.std(rates_array, axis=0)
        return np.round(std_rates, 4)

    std_rates_knn = calculate_std_cm_rates(total_cm_knn)
    std_rates_rf = calculate_std_cm_rates(total_cm_rf)

    print("\nStandard deviations of rates across folds:")
    print("\nKNN - Standard Deviation of Confusion Matrix Rates:")
    print("      Pred_Healthy  Pred_Disease")
    print(f"Act_Healthy    {std_rates_knn[0, 0]:.4f}    {std_rates_knn[0, 1]:.4f}  (TNR, FPR)")
    print(f"Act_Disease    {std_rates_knn[1, 0]:.4f}    {std_rates_knn[1, 1]:.4f}  (FNR, TPR)")

    print("\nRandom Forest - Standard Deviation of Confusion Matrix Rates:")
    print("      Pred_Healthy  Pred_Disease")
    print(f"Act_Healthy    {std_rates_rf[0, 0]:.4f}    {std_rates_rf[0, 1]:.4f}  (TNR, FPR)")
    print(f"Act_Disease    {std_rates_rf[1, 0]:.4f}    {std_rates_rf[1, 1]:.4f}  (FNR, TPR)")

    # TODO: Check the following code!

    # Plot dimensionality reduction visualizations
    all_healthy_vectors = np.vstack([healthy_vectors, healthy_test_vectors])
    plot_vectors(patient_vectors, all_healthy_vectors, args, " - All Patients")


def plot_vectors(patient_vectors, healthy_vectors, args, title_suffix=""):
    """
    Plot t-SNE and PCA visualizations of patient vectors, colored by disease/healthy status

    Args:
        patient_vectors: numpy array of disease patient vectors
        healthy_vectors: numpy array of healthy patient vectors
        title_suffix: string to add to plot titles
    """
    # Combine all vectors and create labels
    all_vectors = np.vstack([patient_vectors, healthy_vectors])
    labels = np.hstack([np.ones(len(patient_vectors)), np.zeros(len(healthy_vectors))])
    label_names = ['Healthy', 'Disease']
    # Blue for healthy, Red/Pink for disease
    colors = ['#2E86AB', '#A23B72']

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # PCA
    pca = PCA(n_components=2, random_state=42)
    pca_result = pca.fit_transform(all_vectors)

    # Plot PCA
    for i, (label, color, name) in enumerate(zip([0, 1], colors, label_names)):
        mask = labels == label
        axes[0].scatter(pca_result[mask, 0], pca_result[mask, 1],
                        c=color, label=name, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    axes[0].set_title(f'PCA of Patient Vectors{title_suffix}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_vectors) - 1))
    tsne_result = tsne.fit_transform(all_vectors)

    # Plot t-SNE
    for i, (label, color, name) in enumerate(zip([0, 1], colors, label_names)):
        mask = labels == label
        axes[1].scatter(tsne_result[mask, 0], tsne_result[mask, 1],
                        c=color, label=name, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)

    axes[1].set_xlabel('t-SNE Component 1')
    axes[1].set_ylabel('t-SNE Component 2')
    axes[1].set_title(f't-SNE of Patient Vectors{title_suffix}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    # save under plot dirs with name including the get_model_config_str
    model_config_str = get_model_config_str(args)
    plt.savefig(os.path.join(INFERENCE_VECTOR_PLOTS_DIR, f"patient_vectors_{model_config_str}{title_suffix}.png"))
    plt.show()

    # Print some statistics
    print(f"Vector dimensionality: {all_vectors.shape[1]}")
    print(f"Number of disease patients: {len(patient_vectors)}")
    print(f"Number of healthy patients: {len(healthy_vectors)}")
    print(
        f"PCA explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
    print(f"Total explained variance by first 2 PCs: {sum(pca.explained_variance_ratio_):.3f}")

