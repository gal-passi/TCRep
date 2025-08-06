import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from models.cvc_cacheing_model import CVCCachingModel
from models.cvc_df_caching_model import CVCDFCachingModel
from models.cvc_basic_cacheing_model import CVCBasicCachingModel
from cache_handler import get_model_config_str
from sklearn.manifold import TSNE
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp
from multiprocessing import Pool, cpu_count
from cache_handler import load_model_state
from models.cvc_ensemble_model import CVCEnsembleModel
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pickle


INFERENCE_BASE_PLOT_DIR = "plots/cvc_model/inference_plots/"
INFERENCE_V2_BASE_PLOT_DIR = "plots/cvc_model/inference_plots_v2/"
INFERENCE_CONFUSION_MATRIX_DIR = os.path.join(INFERENCE_BASE_PLOT_DIR, "confusion_matrices/")
INFERENCE_VECTOR_PLOTS_DIR = os.path.join(INFERENCE_BASE_PLOT_DIR, "vector_plots/")
INFERENCE_CONFUSION_MATRIX_V2_DIR = os.path.join(INFERENCE_V2_BASE_PLOT_DIR, "confusion_matrices_v2/")
INFERENCE_GMM_MEAN_COV_PLOTS_V2_DIR = os.path.join(INFERENCE_V2_BASE_PLOT_DIR, "gmm_mean_cov_plots_v2/")
INFERENCE_GMM_MEAN_WEIGHT_SCATTER_V2_DIR = os.path.join(INFERENCE_V2_BASE_PLOT_DIR, "gmm_mean_weight_scatter_v2/")
INFERENCE_GMM_BIC_PLOTS_V2_DIR = os.path.join(INFERENCE_V2_BASE_PLOT_DIR, "gmm_plots_v2/")
INFERENCE_VECTOR_PLOTS_V2_DIR = os.path.join(INFERENCE_V2_BASE_PLOT_DIR, "vector_plots_v2/")
INFERENCE_UNCERTAINTY_THRESHOLD_V2_DIR = os.path.join(INFERENCE_V2_BASE_PLOT_DIR, "uncertainty_threshold_plots_v2/")
INFERENCE_UNCERTAINTY_THRESHOLD_CACHE_V2_DIR = os.path.join(INFERENCE_V2_BASE_PLOT_DIR, "uncertainty_threshold_plots_v2/cache/")
INFERENCE_OUTLIER_GMMS_PLOTS_V2_DIR = os.path.join(INFERENCE_V2_BASE_PLOT_DIR, "outlier_gmms_plots_v2/")
INFERENCE_SVM_EXTRA_PLOTS_V2_DIR = os.path.join(INFERENCE_V2_BASE_PLOT_DIR, "svm_plots_v2/")

# Models Configurations:
DISABLE_BAD_MODELS = True
DISABLE_DIST_MODELS = True


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
                         aaseq_to_ratio, possible_seqs=None, unique_patient_ids=None, start_vec_from=0, samples=None, bad_seqs=None):
    patient_vectors = []
    patient_probs = []
    for patient_ind in patient_inds:
        # Extract and process the patient sequences
        if unique_patient_ids is not None and possible_seqs is not None:
            patient_seqs = df.loc[df["patient_id"] == unique_patient_ids[patient_ind], "AASeq"].values
            patient_seqs = np.array([x for x in np.unique(patient_seqs) if x in possible_seqs])
        else:
            patient_seqs = df.loc[df["patient_id"] == patient_ind, "AASeq"].values

        patient_seqs = sorted(patient_seqs)
        rng = np.random.RandomState(42)  # fixed seed, legacy & portable RNG
        if samples is not None and len(patient_seqs) > samples:
            patient_seqs = rng.choice(patient_seqs, size=samples, replace=False)

        # Filter patient_seqs to only include those in ok_seqs
        if bad_seqs is not None:
            patient_seqs = np.array(patient_seqs)
            mask = np.isin(patient_seqs, bad_seqs)
            patient_seqs_new = patient_seqs[~mask]
            if len(patient_seqs_new) == 0:
                print(f"Patient {patient_ind} has no valid sequences after filtering bad sequences.")
                continue
            patient_seqs = patient_seqs_new

        # Get model outputs
        with torch.no_grad():
            disease_logits = caching_model(patient_seqs)

        # Convert to probabilities
        disease_probs = torch.softmax(disease_logits, dim=1)[:, 1].cpu().numpy()
        patient_probs.append(disease_probs)

        # bin using np into x bins (min value is 0 and max is 1)
        disease_probs[disease_probs == 1] = 0.9999  # Avoid binning issues with 1.0
        disease_probs_dig = np.digitize(disease_probs, bins=np.linspace(0, 1, vector_representation_bins + 1)) - 1
        # Vectorized bin counting
        patient_vector = np.bincount(disease_probs_dig, minlength=vector_representation_bins).astype(np.float32)
        # patient_vector /= patient_vector.sum()
        patient_vector /= patient_vector.max()
        patient_vector = patient_vector[start_vec_from:]

        if add_ratio_to_vector:
            patient_ratio = aaseq_to_ratio(patient_seqs, dont_use_function=True).values
            patient_vector_ratio = create_binned_disease_probs(patient_ratio, disease_probs, num_bins=vector_representation_bins)
            patient_vector_ratio /= patient_vector_ratio.max()
            patient_vector_ratio = patient_vector_ratio[start_vec_from:]
            patient_vectors.append(np.concatenate([patient_vector, patient_vector_ratio], axis=0))
        else:
            patient_vectors.append(patient_vector)
    patient_vectors = np.array(patient_vectors)
    return patient_vectors, patient_probs


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


# Calculate standard deviations to show variability across folds
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


def ks_statistic(x, y):
    return ks_2samp(x, y).statistic


def _predict_single(args):
    metric_func, disease_base, healthy_base, target = args
    disease_dist = metric_func(disease_base, target)
    healthy_dist = metric_func(healthy_base, target)
    return 1 if disease_dist < healthy_dist else 0


def dist_predictor(metric_func, disease_base, healthy_base, targets, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()

    args_list = [(metric_func, disease_base, healthy_base, target) for target in targets]

    with Pool(processes=num_workers) as pool:
        y_pred = pool.map(_predict_single, args_list)

    return np.array(y_pred)


def plot_feature_importances(rf_feature_importance, start_vec_from, add_ratio_to_vector, args):
    print("\nRandom Forest Sorted Feature Importances (averaged across folds):")
    rf_feature_importance /= rf_feature_importance.max()
    if add_ratio_to_vector:
        feature_names = [f"Feature {i + start_vec_from}" for i in range(len(rf_feature_importance) // 2)] + \
                        [f"Ratio Feature {i + start_vec_from + len(rf_feature_importance) // 2}" for i in range(len(rf_feature_importance) // 2)]
    else:
        feature_names = [f"Feature {i + start_vec_from}" for i in range(len(rf_feature_importance))]
    indices = np.argsort(rf_feature_importance)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Normalized)")
    plt.bar(range(len(rf_feature_importance)), rf_feature_importance[indices], align="center")
    plt.xticks(range(len(rf_feature_importance)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    # save under plot dirs with name including the get_model_config_str
    model_config_str = get_model_config_str(args)
    plt.savefig(os.path.join(INFERENCE_VECTOR_PLOTS_DIR, f"rf_feature_importances_{model_config_str}.png"))
    plt.show()


def inference_classification_model(trained_model, args, df_bld, df_hlt, test_patient_inds, valid_patient_inds, unique_patient_ids,
                                   valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs, aaseq_to_ratio, to_ensemble, device,
                                   add_ratio_to_vector=False, start_vec_from=0,
                                   # vector_representation_bins=20, num_of_healthy_patients=8, num_of_healthy_test_patients=2):
                                   vector_representation_bins=40, num_of_healthy_patients=68, num_of_healthy_test_patients=28, only_all_classifiers=False):
    # TODO: Hot fix for now when using different set of num of healthy patients:
    if len(df_hlt["patient_id"].unique()) != num_of_healthy_patients:
        num_of_healthy_patients = len(df_hlt["patient_id"].unique())
        num_of_healthy_test_patients = int(num_of_healthy_patients * 0.25)

    np.random.seed(42)
    # make sure that plot dirs exists
    os.makedirs(INFERENCE_CONFUSION_MATRIX_DIR, exist_ok=True)
    os.makedirs(INFERENCE_VECTOR_PLOTS_DIR, exist_ok=True)

    # Creating a caching model of the trained model
    if to_ensemble:
        caching_model = trained_model
    else:
        trained_model.eval()
        caching_model = CVCCachingModel(trained_model, args, device)
        caching_model.to(device)
        caching_model.eval()

    # patient vectors helping data
    possible_seqs = set(np.concatenate([valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs]))
    patient_valid_test_inds = np.concatenate([test_patient_inds, valid_patient_inds])

    # get the patient vectors
    patient_vectors, patient_probs = calc_patient_vectors(df_bld, caching_model, patient_valid_test_inds, vector_representation_bins,
                                                          add_ratio_to_vector, aaseq_to_ratio, unique_patient_ids=unique_patient_ids,
                                                          possible_seqs=possible_seqs, start_vec_from=start_vec_from)

    # healthy vectors helping data
    healthy_patients = df_hlt["patient_id"].unique()
    np.random.shuffle(healthy_patients)
    healthy_patients = healthy_patients[:num_of_healthy_patients]

    healthy_vectors, healthy_probs = calc_patient_vectors(df_hlt, caching_model, healthy_patients, vector_representation_bins,
                                           add_ratio_to_vector, aaseq_to_ratio, start_vec_from=start_vec_from)
    healthy_vectors, healthy_test_vectors = (healthy_vectors[:num_of_healthy_patients - num_of_healthy_test_patients],
                                             healthy_vectors[num_of_healthy_patients - num_of_healthy_test_patients:])
    healthy_probs, healthy_test_probs = (healthy_probs[:num_of_healthy_patients - num_of_healthy_test_patients],
                                         healthy_probs[num_of_healthy_patients - num_of_healthy_test_patients:])

    total_cm_knn = []
    total_cm_rf = []
    total_cm_svm_rbf = []
    total_cm_knn_rbf = []
    total_cm_svm_linear = []
    total_cm_logistic = []
    total_cm_mlp = []
    total_cm_nb = []
    total_cm_jsd = []
    total_cm_wsd = []
    total_cm_ks = []
    rf_feature_importance = np.zeros(len(patient_vectors[0]))

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

        # Generate model scores for score-based classifiers
        # TODO: Find a way to deal with cases where the lengths are not equal!!!
        # min_len = 10000  # min([len(x) for x in patient_probs + healthy_probs])
        # train_patient_scores = get_model_scores(patient_probs, train_patients, min_len)  # patient_probs[train_patients]
        # train_healthy_scores = get_model_scores(healthy_probs, list(range(len(healthy_probs))), min_len)  # healthy_probs
        # test_patient_scores = get_model_scores(patient_probs, test_patients, min_len) # patient_probs[test_patients]
        # test_healthy_scores = get_model_scores(healthy_test_probs, list(range(len(healthy_test_probs))), min_len)  # healthy_test_probs
        # # Combine scores for training and testing
        # train_scores = np.vstack([train_patient_scores, train_healthy_scores])
        # test_scores = np.vstack([test_patient_scores, test_healthy_scores])

        # # Reshape scores to be 2D if they're 1D
        # if len(train_scores.shape) == 1:
        #     train_scores = train_scores.reshape(-1, 1)
        # if len(test_scores.shape) == 1:
        #     test_scores = test_scores.reshape(-1, 1)

        if not DISABLE_DIST_MODELS:
            disease_base_bins = train_vectors.mean(axis=0)
            healthy_base_bins = healthy_vectors.mean(axis=0)
            distribution_test_bins = np.concatenate([test_vectors, healthy_test_vectors])

            disease_base = np.concatenate([patient_probs[i] for i in train_patients])
            healthy_base = np.concatenate(healthy_probs)
            distribution_test = [patient_probs[i] for i in test_patients] + list(healthy_test_probs)

            # 0. Distance-based predictions
            y_pred_jsd = dist_predictor(jensenshannon, disease_base_bins, healthy_base_bins, distribution_test_bins)
            cm_jsd = confusion_matrix(y_test, y_pred_jsd)
            y_pred_wsd = dist_predictor(wasserstein_distance, disease_base, healthy_base, distribution_test)
            cm_wsd = confusion_matrix(y_test, y_pred_wsd)
            y_pred_ks = dist_predictor(ks_statistic, disease_base, healthy_base, distribution_test)
            cm_ks = confusion_matrix(y_test, y_pred_ks)

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

        importances = rf.feature_importances_
        rf_feature_importance += importances

        # 3. SVM with RBF kernel (Score-based)
        svm_rbf = SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=True)
        svm_rbf.fit(X_train, y_train)
        y_pred_svm_rbf = svm_rbf.predict(X_test)
        cm_svm_rbf = confusion_matrix(y_test, y_pred_svm_rbf)

        # TODO: Figure out how to run KNN with RBF kernel correctly!
        # KNN with RBF kernel
        from sklearn.metrics.pairwise import rbf_kernel
        gamma = 1.0 / (2 * np.var(X_train))  # A common heuristic
        # Compute RBF kernel matrix
        X_train_rbf = rbf_kernel(X_train, X_train, gamma=gamma)
        X_test_rbf = rbf_kernel(X_test, X_train, gamma=gamma)
        # Apply KNN on the transformed features
        knn_rbf = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
        knn_rbf.fit(X_train_rbf, y_train)
        y_pred_knn_rbf = knn_rbf.predict(X_test_rbf)
        cm_knn_rbf = confusion_matrix(y_test, y_pred_knn_rbf)

        # knn_rbf = KNeighborsClassifier(n_neighbors=2, metric='rbf')  # Not directly supported, using default metric
        # knn_rbf.fit(X_train, y_train)
        # y_pred_knn_rbf = knn_rbf.predict(X_test)
        # cm_knn_rbf = confusion_matrix(y_test, y_pred_knn_rbf)

        if not DISABLE_BAD_MODELS:
            # 4. SVM with Linear kernel (Score-based)
            svm_linear = SVC(kernel='linear', class_weight='balanced', random_state=42, probability=True)
            svm_linear.fit(X_train, y_train)
            y_pred_svm_linear = svm_linear.predict(X_test)
            cm_svm_linear = confusion_matrix(y_test, y_pred_svm_linear)

            # 5. Logistic Regression (Score-based)
            logistic = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
            logistic.fit(X_train, y_train)
            y_pred_logistic = logistic.predict(X_test)
            cm_logistic = confusion_matrix(y_test, y_pred_logistic)

            # 6. Multi-layer Perceptron (Score-based)
            mlp = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42)
            mlp.fit(X_train, y_train)
            y_pred_mlp = mlp.predict(X_test)
            cm_mlp = confusion_matrix(y_test, y_pred_mlp)

            # 7. Naive Bayes (Score-based)
            nb = GaussianNB()
            nb.fit(X_train, y_train)
            y_pred_nb = nb.predict(X_test)
            cm_nb = confusion_matrix(y_test, y_pred_nb)

        # Append the confusion matrices to the total list
        total_cm_knn.append(cm_knn)
        total_cm_rf.append(cm_rf)
        total_cm_svm_rbf.append(cm_svm_rbf)
        total_cm_knn_rbf.append(cm_knn_rbf)
        if not DISABLE_BAD_MODELS:
            total_cm_svm_linear.append(cm_svm_linear)
            total_cm_logistic.append(cm_logistic)
            total_cm_mlp.append(cm_mlp)
            total_cm_nb.append(cm_nb)
        if not DISABLE_DIST_MODELS:
            total_cm_jsd.append(cm_jsd)
            total_cm_wsd.append(cm_wsd)
            total_cm_ks.append(cm_ks)

    # Print feature importance for Random Forest
    if not only_all_classifiers:
        plot_feature_importances(rf_feature_importance, start_vec_from, add_ratio_to_vector, args)

    # Dictionary of all classifiers and their results
    all_classifiers = {
        'KNN (Vector-based)': total_cm_knn,
        'Random Forest (Vector-based)': total_cm_rf,
        'SVM RBF (Score-based)': total_cm_svm_rbf,
        'KNN RBF Kernel (Vector-based)': total_cm_knn_rbf
    }
    if not DISABLE_BAD_MODELS:
        all_classifiers.update({
            'SVM Linear (Score-based)': total_cm_svm_linear,
            'Logistic Regression (Score-based)': total_cm_logistic,
            'MLP (Score-based)': total_cm_mlp,
            'Naive Bayes (Score-based)': total_cm_nb
        })
    if not DISABLE_DIST_MODELS:
        all_classifiers.update({
            'Jensen-Shannon Distance': total_cm_jsd,
            'Wasserstein Distance': total_cm_wsd,
            'Kolmogorov-Smirnov Distance': total_cm_ks
        })

    if only_all_classifiers:
        return all_classifiers

    # Display final results for all classifiers
    display_enhanced_results(all_classifiers, args)
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

    # Plot dimensionality reduction visualizations
    all_healthy_vectors = np.vstack([healthy_vectors, healthy_test_vectors])
    plot_vectors(patient_vectors, all_healthy_vectors, args, " - All Patients")


def plot_vectors(patient_vectors, healthy_vectors, args, title_suffix="", inference_save_dir=INFERENCE_VECTOR_PLOTS_DIR):
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
    plt.savefig(os.path.join(inference_save_dir, f"patient_vectors_{model_config_str}{title_suffix}.png"))
    plt.show()

    # Print statistics
    print(f"Vector dimensionality: {all_vectors.shape[1]}")
    print(f"Number of disease patients: {len(patient_vectors)}")
    print(f"Number of healthy patients: {len(healthy_vectors)}")
    print(f"PCA explained variance ratio: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
    print(f"Total explained variance by first 2 PCs: {sum(pca.explained_variance_ratio_):.3f}")


def get_model_scores(patient_probs, train_patients, min_len=None):
    min_array_len = min([len(x) for x in patient_probs]) if min_len is None else min_len
    # Ensure all patient_probs have the same length
    # patient_probs = [x[:min_array_len] for x in patient_probs]
    modified_patient_probs = []
    for i in range(len(patient_probs)):
        prob = patient_probs[i]
        if len(prob) < min_array_len:
            difference_len = min_len - len(prob)
            # sample that many out of prob
            prob = np.concatenate([prob, np.random.choice(prob, size=difference_len, replace=True)])
        else:
            prob = np.random.choice(prob, size=min_len, replace=False)
        modified_patient_probs.append(prob)
    # Stack the probabilities for the training patients
    patient_probs_stacked = np.vstack([modified_patient_probs[i] for i in train_patients])
    return patient_probs_stacked


def display_enhanced_results(all_classifiers, args):
    """Display results for all classifiers"""

    n_folds = len(list(all_classifiers.values())[0])
    print(f"Average results across {n_folds} folds for all classifiers:")
    print("=" * 80)

    # Calculate and display results for each classifier
    results_summary = {}

    for classifier_name, cm_list in all_classifiers.items():
        print(f"\n{classifier_name}:")
        print("-" * 50)

        # Calculate average and std
        avg_rates = calculate_average_cm_rates(cm_list)
        std_rates = calculate_std_cm_rates(cm_list)

        # Store results
        results_summary[classifier_name] = {
            'avg_rates': avg_rates,
            'std_rates': std_rates
        }

        # Display confusion matrix
        print("Average Confusion Matrix (Rates):")
        print("      Pred_Healthy  Pred_Disease")
        print(f"Act_Healthy    {avg_rates[0, 0]:.4f}    {avg_rates[0, 1]:.4f}  (TNR, FPR)")
        print(f"Act_Disease    {avg_rates[1, 0]:.4f}    {avg_rates[1, 1]:.4f}  (FNR, TPR)")

        print("\nStandard Deviations:")
        print("      Pred_Healthy  Pred_Disease")
        print(f"Act_Healthy    {std_rates[0, 0]:.4f}    {std_rates[0, 1]:.4f}  (TNR, FPR)")
        print(f"Act_Disease    {std_rates[1, 0]:.4f}    {std_rates[1, 1]:.4f}  (FNR, TPR)")

        # Calculate key metrics
        TPR = avg_rates[1, 1]  # Sensitivity
        TNR = avg_rates[0, 0]  # Specificity
        PPV = TPR / (TPR + avg_rates[0, 1]) if (TPR + avg_rates[0, 1]) > 0 else 0  # Precision
        F1 = 2 * (PPV * TPR) / (PPV + TPR) if (PPV + TPR) > 0 else 0

        print(f"\nKey Metrics:")
        print(f"Sensitivity (TPR): {TPR:.4f}")
        print(f"Specificity (TNR): {TNR:.4f}")
        print(f"Precision (PPV):   {PPV:.4f}")
        print(f"F1-Score:          {F1:.4f}")

    # Create comprehensive visualization
    create_comprehensive_visualization(all_classifiers, results_summary, args)

    # Create summary comparison
    # create_summary_comparison(results_summary, args)


def create_comprehensive_visualization(all_classifiers, results_summary, args):
    """Create comprehensive visualization of all classifiers"""

    classifiers = list(results_summary.keys())
    metrics = ['Sensitivity-TPR', 'Specificity-TNR', 'Precision', 'F1-Score', 'Accuracy']

    # Calculate metrics for each classifier
    metric_values = {metric: [] for metric in metrics}

    for classifier_name in classifiers:
        avg_rates = results_summary[classifier_name]['avg_rates']

        TPR = avg_rates[1, 1]  # Sensitivity
        TNR = avg_rates[0, 0]  # Specificity
        PPV = TPR / (TPR + avg_rates[0, 1]) if (TPR + avg_rates[0, 1]) > 0 else 0  # Precision
        F1 = 2 * (PPV * TPR) / (PPV + TPR) if (PPV + TPR) > 0 else 0
        accuracy = (avg_rates[0, 0] + avg_rates[1, 1]) / np.sum(avg_rates) if np.sum(avg_rates) > 0 else 0

        metric_values['Sensitivity-TPR'].append(TPR)
        metric_values['Specificity-TNR'].append(TNR)
        metric_values['Precision'].append(PPV)
        metric_values['F1-Score'].append(F1)
        metric_values['Accuracy'].append(accuracy)

    n_classifiers = len(all_classifiers)
    cols = 3
    rows = (n_classifiers + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif n_classifiers == 1:
        axes = np.array([[axes]])

    axes = axes.flatten()

    for idx, (classifier_name, cm_list) in enumerate(all_classifiers.items()):
        avg_rates = calculate_average_cm_rates(cm_list)

        sns.heatmap(avg_rates, annot=True, fmt='.3f', cmap='Blues',
                    xticklabels=['Predicted Healthy', 'Predicted Disease'],
                    yticklabels=['Actual Healthy', 'Actual Disease'],
                    ax=axes[idx])

        # Clean title with just classifier name
        axes[idx].set_title(f'{classifier_name}', fontsize=12, fontweight='bold')

        # Option 1: Add metrics as text box in corner
        metrics_text = (f'Acc: {metric_values["Accuracy"][idx]:.3f}\n'
                        f'F1: {metric_values["F1-Score"][idx]:.3f}\n'
                        f'Prec: {metric_values["Precision"][idx]:.3f}')

        axes[idx].text(0.02, 0.98, metrics_text, transform=axes[idx].transAxes,
                       fontsize=9, verticalalignment='top', horizontalalignment='left',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Hide unused subplots
    for idx in range(n_classifiers, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()

    # Save plot
    model_config_str = get_model_config_str(args)
    plt.savefig(os.path.join(INFERENCE_CONFUSION_MATRIX_DIR,
                             f"all_classifiers_confusion_matrices_{model_config_str}.png"))
    plt.show()


def create_summary_comparison(results_summary, args):
    """Create a summary comparison of all classifiers"""

    classifiers = list(results_summary.keys())
    metrics = ['Sensitivity-TPR', 'Specificity-TNR', 'Precision', 'F1-Score', 'Accuracy']

    # Calculate metrics for each classifier
    metric_values = {metric: [] for metric in metrics}

    for classifier_name in classifiers:
        avg_rates = results_summary[classifier_name]['avg_rates']

        TPR = avg_rates[1, 1]  # Sensitivity
        TNR = avg_rates[0, 0]  # Specificity
        PPV = TPR / (TPR + avg_rates[0, 1]) if (TPR + avg_rates[0, 1]) > 0 else 0  # Precision
        F1 = 2 * (PPV * TPR) / (PPV + TPR) if (PPV + TPR) > 0 else 0
        accuracy = (avg_rates[0, 0] + avg_rates[1, 1]) / np.sum(avg_rates) if np.sum(avg_rates) > 0 else 0

        metric_values['Sensitivity-TPR'].append(TPR)
        metric_values['Specificity-TNR'].append(TNR)
        metric_values['Precision'].append(PPV)
        metric_values['F1-Score'].append(F1)
        metric_values['Accuracy'].append(accuracy)

    # Create comparison plot
    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(classifiers))
    width = 0.2

    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, metric_values[metric], width, label=metric, alpha=0.8)

    ax.set_xlabel('Classifiers')
    ax.set_ylabel('Score')
    ax.set_title('Classifier Performance Comparison')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(classifiers, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    model_config_str = get_model_config_str(args)
    plt.savefig(os.path.join(INFERENCE_CONFUSION_MATRIX_DIR, f"classifier_performance_comparison_{model_config_str}.png"))
    plt.show()

    # Print summary table
    print("\n" + "=" * 100)
    print("CLASSIFIER PERFORMANCE SUMMARY")
    print("=" * 100)
    print(f"{'Classifier':<35} {'Sensitivity':<12} {'Specificity':<12} {'Precision':<12} {'F1-Score':<12}")
    print("-" * 100)

    for i, classifier_name in enumerate(classifiers):
        print(f"{classifier_name:<35} {metric_values['Sensitivity-TPR'][i]:<12.4f} "
              f"{metric_values['Specificity-TNR'][i]:<12.4f} {metric_values['Precision'][i]:<12.4f} "
              f"{metric_values['F1-Score'][i]:<12.4f}")


def inference_classification_model_combined(trained_model, args, to_ensemble, get_data_loader_wrapper, device,
                                   add_ratio_to_vector=False, start_vec_from=20,
                                   # vector_representation_bins=20, num_of_healthy_patients=8, num_of_healthy_test_patients=2):
                                   vector_representation_bins=40, num_of_healthy_patients=68, num_of_healthy_test_patients=28):
    all_classifiers = []
    for fold_ind in range(1, 6):
        # Load the dataset for the current fold
        args.k_fold = fold_ind
        dataset_loader = get_data_loader_wrapper(fold_ind)
        df_bld, df_hlt = dataset_loader.get_dfs()
        train_pos_seqs, neg_seqs, valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs = dataset_loader.get_seqs()
        train_patient_inds, valid_patient_inds, test_patient_inds = dataset_loader.get_patient_inds()
        unique_patient_ids = dataset_loader.unique_patient_ids
        aaseq_to_ratio = dataset_loader.get_aaseq_to_ratio_func()

        # Load the model state for the current fold
        if to_ensemble:
            from cache_handler import get_model_dir
            cache_dir = get_model_dir(args)
            trained_model = CVCEnsembleModel(args, device, cache_dir=cache_dir, default_to_return='min')
        else:
            trained_model = load_model_state(trained_model, args, args.epochs - 1, device)
            if trained_model is None:
                print(f"Model for fold {fold_ind} not found. Skipping this fold.")
                continue

        # Run inference for the current fold
        fold_classifier = inference_classification_model(
            trained_model, args, df_bld, df_hlt, test_patient_inds, valid_patient_inds,
            unique_patient_ids, valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs,
            aaseq_to_ratio, to_ensemble, device, add_ratio_to_vector=add_ratio_to_vector,
            start_vec_from=start_vec_from, vector_representation_bins=vector_representation_bins,
            num_of_healthy_patients=num_of_healthy_patients,
            num_of_healthy_test_patients=num_of_healthy_test_patients,
            only_all_classifiers=True
        )

        # Append the results of the current fold to the all_classifiers list
        all_classifiers.append(fold_classifier)

    # Combine results from all folds
    combined_classifiers = {}
    for classifier_name in all_classifiers[0].keys():
        combined_classifiers[classifier_name] = []
        for fold_classifier in all_classifiers:
            combined_classifiers[classifier_name].extend(fold_classifier[classifier_name])

    # Display combined results using display_enhanced_results
    args.k_fold = 0
    display_enhanced_results(combined_classifiers, args)


def inference_classification_model_version2(trained_model, args, df_bld, df_hlt, test_patient_ids, valid_patient_ids,
                                            valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs, aaseq_to_ratio, to_ensemble, model_non_trained, device,
                                            add_ratio_to_vector=False, start_vec_from=0,
                                            vector_representation_bins=40, num_of_healthy_patients=68, num_of_healthy_test_patients=28, only_all_classifiers=False,
                                            to_display_mapping=False, samples_size=20000,
                                            k_fold_disease=1, chosen_components=2, to_savefig=True,
                                            covariance_type='tied'):
    np.random.seed(42)
    # Take shuffle and divide the patient 1/3 such that k_fold_disease will choose which 1/3 of patients to take
    patient_valid_test_ids = np.concatenate([test_patient_ids, valid_patient_ids])
    patient_train_ids = [x for x in df_bld["patient_id"].unique() if x not in patient_valid_test_ids]
    np.random.shuffle(patient_train_ids)
    num_of_patients = len(patient_train_ids)
    num_of_patients_per_fold = num_of_patients // 3
    start_ind = (k_fold_disease - 1) * num_of_patients_per_fold
    end_ind = start_ind + num_of_patients_per_fold
    fold_patient_test_ids = patient_train_ids[start_ind:end_ind]
    fold_patient_train_ids = [x for x in patient_train_ids if x not in fold_patient_test_ids]
    df_bld_validation = df_bld[df_bld["patient_id"].isin(fold_patient_test_ids)]
    df_train_ids = np.concatenate([patient_valid_test_ids, fold_patient_train_ids])
    df_bld = df_bld[df_bld["patient_id"].isin(df_train_ids)]

    if len(df_hlt["patient_id"].unique()) >= int(len(df_bld["patient_id"].unique()) * 2.0):
        num_of_healthy_patients = int(len(df_bld["patient_id"].unique()) * 2.0)
        num_of_healthy_test_patients = int(len(df_bld["patient_id"].unique()) * 1.0)
    elif len(df_hlt["patient_id"].unique()) != num_of_healthy_patients:
        num_of_healthy_patients = len(df_hlt["patient_id"].unique())
        num_of_healthy_test_patients = int(num_of_healthy_patients * 0.25)

    np.random.seed(42)
    # make sure that plot dirs exists
    os.makedirs(INFERENCE_CONFUSION_MATRIX_V2_DIR, exist_ok=True)
    # os.makedirs(INFERENCE_VECTOR_PLOTS_V2_DIR, exist_ok=True)  # disabled for now
    os.makedirs(INFERENCE_GMM_MEAN_COV_PLOTS_V2_DIR, exist_ok=True)
    os.makedirs(INFERENCE_GMM_MEAN_WEIGHT_SCATTER_V2_DIR, exist_ok=True)
    os.makedirs(INFERENCE_GMM_BIC_PLOTS_V2_DIR, exist_ok=True)
    os.makedirs(INFERENCE_UNCERTAINTY_THRESHOLD_V2_DIR, exist_ok=True)
    os.makedirs(INFERENCE_UNCERTAINTY_THRESHOLD_CACHE_V2_DIR, exist_ok=True)
    os.makedirs(INFERENCE_SVM_EXTRA_PLOTS_V2_DIR, exist_ok=True)
    os.makedirs(INFERENCE_OUTLIER_GMMS_PLOTS_V2_DIR, exist_ok=True)
    model_config_str = get_model_config_str(args)

    bad_seqs = None
    filter_uncertain_seqs = False
    if filter_uncertain_seqs:
        reshef_cache_folder = os.path.join("cache", "reshef_inference")
        reshef_inference_data_path = os.path.join(reshef_cache_folder, "reshef_inference_low_confidence_neg_seqs.npy")
        if not os.path.exists(reshef_inference_data_path):
            print(f"Reshef inference data not found at {reshef_inference_data_path}.")
        bad_seqs = np.load(reshef_inference_data_path)

    # Creating a caching model of the trained model
    if to_ensemble:  # TODO: This does not work currently! Raising NotImplementedError
        caching_model = trained_model
        raise NotImplementedError("Ensemble model inference with non-trained model is not implemented yet.")
    else:
        trained_model.eval()
        if samples_size == 100:
            caching_model = trained_model
        else:
            caching_model = CVCBasicCachingModel(trained_model, args, device, verbose=True)
            # caching_model = CVCCachingModel(trained_model, args, device)
            # caching_model = CVCDFCachingModel(trained_model, args, device)  # TODO: This is very very slow for some reason...
        caching_model.to(device)
        caching_model.eval()

    # EXTRA - calculate model outputs on valid and test positive sequences
    if to_display_mapping and model_non_trained is not None:
        part_pos_valid = np.random.choice(valid_pos_seqs, size=len(valid_pos_seqs), replace=False)
        part_pos_test = np.random.choice(test_pos_seqs, size=len(test_pos_seqs), replace=False)
        pos_seqs = np.concatenate([part_pos_valid, part_pos_test])
        neg_seqs = np.random.choice(df_hlt['AASeq'].unique(), size=len(pos_seqs), replace=False)
        to_save_plots_data = False

        available_models = [trained_model, model_non_trained]
        models_names = ['Trained Model', 'Non-Trained Model']
        for model, model_name in zip(available_models, models_names):
            # get model outputs for pos and neg sequences:
            model.eval()
            with torch.no_grad():
                pos_embeds = model.get_embeddings(pos_seqs)
                neg_embeds = model.get_embeddings(neg_seqs)

                pos_outputs = torch.softmax(model.linear(pos_embeds.to(torch.float32)), dim=1)
                neg_outputs = torch.softmax(model.linear(neg_embeds.to(torch.float32)), dim=1)

            # save the embeddings and outputs in "plots/plots_for_posters/data/embedding_plot"
            if to_save_plots_data:
                # save all to one file
                np.savez("plots/plots_for_posters/data/embedding_plots_data.npz",
                         pos_embeds=pos_embeds.cpu().numpy(), neg_embeds=neg_embeds.cpu().numpy(),
                         pos_outputs=pos_outputs.cpu().numpy(), neg_outputs=neg_outputs.cpu().numpy())

            confident_pos_indices = pos_outputs[:, 1] > 0.75
            confident_neg_indices = neg_outputs[:, 1] > 0.75
            # confident_pos_indices = pos_outputs[:, 1] > 0.5
            # confident_neg_indices = neg_outputs[:, 1] <= 0.5
            pos_alphas = np.where(confident_pos_indices, 0.5, 0.035)
            neg_alphas = np.where(confident_neg_indices, 0.5, 0.035)
            # alphas = np.concatenate([pos_alphas, neg_alphas])
            from matplotlib.lines import Line2D

            custom_legend = [
                Line2D([0], [0], marker='o', color='w', label='MS Sequences',
                       markerfacecolor='darkorange', markersize=6),
                Line2D([0], [0], marker='o', color='w', label='Healthy Sequences',
                       markerfacecolor='blue', markersize=6),
            ]

            all_embeds = torch.cat([pos_embeds, neg_embeds], dim=0).cpu().numpy()
            labels = np.array([1] * len(pos_embeds) + [0] * len(neg_embeds))
            # all_embeds = torch.cat([pos_embeds, neg_embeds], dim=0).cpu().numpy()
            # labels = np.array([1] * len(pos_embeds) + [0] * len(neg_embeds))
            # Create the figure with subplots
            fig, axes = plt.subplots(1, 3, figsize=(6*3, 6), dpi=600)
            spine_thickness = 0.75
            # PCA
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(all_embeds)
            axes[0].scatter(pca_result[labels == 0, 0], pca_result[labels == 0, 1], c='blue', label='Healthy Sequences', alpha=neg_alphas)
            axes[0].scatter(pca_result[labels == 1, 0], pca_result[labels == 1, 1], c='darkorange', label='MS Sequences', alpha=pos_alphas)
            axes[0].set_title("PCA of Embeddings")
            # axes[0].set_xlabel("Dim 1")
            # axes[0].set_ylabel("Dim 2")
            axes[0].legend(handles=custom_legend, loc='upper left')
            axes[0].set_xticks([])
            axes[0].set_yticks([])
            for spine in axes[0].spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(spine_thickness)
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
            tsne_result = tsne.fit_transform(all_embeds)
            axes[1].scatter(tsne_result[labels == 0, 0], tsne_result[labels == 0, 1], c='blue', label='Healthy Sequences', alpha=neg_alphas)
            axes[1].scatter(tsne_result[labels == 1, 0], tsne_result[labels == 1, 1], c='darkorange', label='MS Sequences', alpha=pos_alphas)
            axes[1].set_title("t-SNE of Embeddings")
            # axes[1].set_xlabel("Dim 1")
            # axes[1].set_ylabel("Dim 2")
            axes[1].legend(handles=custom_legend, loc='upper left')
            axes[1].set_xticks([])
            axes[1].set_yticks([])
            for spine in axes[1].spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(spine_thickness)
            # UMAP
            umap_model = umap.UMAP(n_components=2, random_state=42)
            umap_result = umap_model.fit_transform(all_embeds)
            axes[2].scatter(umap_result[labels == 0, 0], umap_result[labels == 0, 1], c='blue', label='Healthy Sequences', alpha=neg_alphas)
            axes[2].scatter(umap_result[labels == 1, 0], umap_result[labels == 1, 1], c='darkorange', label='MS Sequences', alpha=pos_alphas)
            axes[2].set_title("UMAP of Embeddings")
            # axes[2].set_xlabel("Dim 1")
            # axes[2].set_ylabel("Dim 2")
            axes[2].legend(handles=custom_legend, loc='upper left')
            plt.suptitle(f"{model_name} - Embeddings Visualization", fontsize=16)
            axes[2].set_xticks([])
            axes[2].set_yticks([])
            for spine in axes[2].spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(spine_thickness)
            plt.tight_layout()
            plt.show()

    # get the patient vectors
    # possible_seqs = set(np.concatenate([valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs]))
    patient_test_vectors, patient_test_probs = calc_patient_vectors(df_bld, caching_model, patient_valid_test_ids, vector_representation_bins,
                                                          add_ratio_to_vector, aaseq_to_ratio, unique_patient_ids=None,
                                                          possible_seqs=None, start_vec_from=start_vec_from, samples=samples_size, bad_seqs=bad_seqs)

    # get the train patient vectors
    patient_train_vectors, patient_train_probs = calc_patient_vectors(df_bld, caching_model, fold_patient_train_ids, vector_representation_bins,
                                                                      add_ratio_to_vector, aaseq_to_ratio, unique_patient_ids=None,
                                                                      possible_seqs=None, start_vec_from=start_vec_from, samples=samples_size, bad_seqs=bad_seqs)

    # get the validation patient vectors
    disease_validation_vectors, disease_validation_probs = calc_patient_vectors(df_bld_validation, caching_model, fold_patient_test_ids, vector_representation_bins,
                                                                                add_ratio_to_vector, aaseq_to_ratio, unique_patient_ids=None,
                                                                                possible_seqs=None, start_vec_from=start_vec_from, samples=samples_size, bad_seqs=bad_seqs)

    # healthy vectors helping data
    healthy_patients = df_hlt["patient_id"].unique()
    np.random.shuffle(healthy_patients)  # TODO: Consider sorting (and even using rng.permutation) to make it more deterministic!
    healthy_patients = healthy_patients[:num_of_healthy_patients]

    healthy_vectors, healthy_probs = calc_patient_vectors(df_hlt, caching_model, healthy_patients, vector_representation_bins,
                                           add_ratio_to_vector, aaseq_to_ratio, start_vec_from=start_vec_from, samples=samples_size, bad_seqs=bad_seqs)
    healthy_vectors, healthy_test_vectors = (healthy_vectors[:num_of_healthy_patients - num_of_healthy_test_patients],
                                             healthy_vectors[num_of_healthy_patients - num_of_healthy_test_patients:])
    healthy_probs, healthy_test_probs = (healthy_probs[:num_of_healthy_patients - num_of_healthy_test_patients],
                                         healthy_probs[num_of_healthy_patients - num_of_healthy_test_patients:])

    # ============= GMM ADDITION STARTS HERE =============

    print("=" * 60)
    print("GAUSSIAN MIXTURE MODEL ANALYSIS ON PROBABILITY DISTRIBUTIONS")
    print("=" * 60)

    # Prepare probability data for GMM fitting
    # Flatten all probability arrays for each group
    patient_train_probs_flat = np.concatenate([probs.flatten() for probs in patient_train_probs])
    healthy_train_probs_flat = np.concatenate([probs.flatten() for probs in healthy_probs])

    # Reshape for sklearn (needs 2D input)
    patient_train_data = patient_train_probs_flat.reshape(-1, 1)
    healthy_train_data = healthy_train_probs_flat.reshape(-1, 1)

    print(f"Patient training probability values: {patient_train_data.shape[0]}")
    print(f"Healthy training probability values: {healthy_train_data.shape[0]}")
    print(f"Patient probability range: [{patient_train_probs_flat.min():.3f}, {patient_train_probs_flat.max():.3f}]")
    print(f"Healthy probability range: [{healthy_train_probs_flat.min():.3f}, {healthy_train_probs_flat.max():.3f}]")

    # Choose a compromise number of components that works for both
    # Choosing chosen_components manually for now:  # Strategy: choose the minimum of the two optimal values, but at least 2
    # chosen_components = max(2, min(optimal_patient_components, optimal_healthy_components))
    print(f"Chosen number of components for both distributions: {chosen_components}")

    # Train final GMMs with chosen number of components
    print(f"\n--- Training Final GMMs with {chosen_components} components - With covariance type: {covariance_type}---")

    final_patient_gmm = GaussianMixture(n_components=chosen_components, random_state=42, covariance_type=covariance_type)
    final_patient_gmm.fit(patient_train_data)
    # full: [[0.00183163], [0.27300646]]
    # diag: [[0.00183163], [0.27300646]]
    # tied: [[0.04703453], [0.68540329]]

    final_healthy_gmm = GaussianMixture(n_components=chosen_components, random_state=42, covariance_type=covariance_type)
    final_healthy_gmm.fit(healthy_train_data)

    print(f"Patient GMM - BIC: {final_patient_gmm.bic(patient_train_data):.2f}")
    print(f"Healthy GMM - BIC: {final_healthy_gmm.bic(healthy_train_data):.2f}")

    gmms = plot_bic_scores_per_models(patient_train_probs, healthy_probs, final_patient_gmm,
                                      final_healthy_gmm, patient_train_data, healthy_train_data,
                                      patient_train_probs_flat, healthy_train_probs_flat, covariance_type,
                                      chosen_components, k_fold_disease, model_config_str, to_savefig=to_savefig)
    patient_gmm_models, healthy_gmm_models = gmms

    recalc_gmms = plot_gmms_per_person_outliers(patient_gmm_models, healthy_gmm_models, patient_train_probs, healthy_probs,
                                                patient_test_probs, healthy_test_probs, covariance_type, chosen_components,
                                                k_fold_disease, model_config_str, to_savefig=to_savefig)
    final_patient_gmm, final_healthy_gmm, patient_gmm_models, healthy_gmm_models = recalc_gmms

    plot_gmm_means_weights_plot(patient_gmm_models, healthy_gmm_models, chosen_components, k_fold_disease, model_config_str, to_savefig=to_savefig)

    plot_gmm_components_per_patient_and_final(patient_gmm_models, healthy_gmm_models, final_patient_gmm,
                                              final_healthy_gmm, chosen_components, k_fold_disease, model_config_str, to_savefig=to_savefig)

    plot_per_person_gmm_components(patient_gmm_models, healthy_gmm_models, patient_test_probs,
                                   healthy_test_probs, covariance_type, chosen_components, size=20)

    cm_svm_rbf = get_svm_rbf_classification_cm(patient_train_vectors, healthy_vectors, patient_test_vectors, healthy_test_vectors,
                                               vector_representation_bins, k_fold_disease, model_config_str, to_savefig=to_savefig)

    plot_gmm_classification_multi_threshold(patient_test_probs, healthy_test_probs, final_patient_gmm, final_healthy_gmm, cm_svm_rbf,
                                            chosen_components, k_fold_disease, model_config_str, to_savefig=to_savefig)


    # Finding uncertainty threshold and such:
    # Process patient test subjects
    patient_lls = []
    healthy_lls = []
    test_true_labels = []
    for i, patient_probs in enumerate(disease_validation_probs):
        patient_probs_reshaped = patient_probs.flatten().reshape(-1, 1)

        # Calculate average log-likelihood for this patient across all their probability values
        patient_ll = np.mean(final_patient_gmm.score_samples(patient_probs_reshaped))
        healthy_ll = np.mean(final_healthy_gmm.score_samples(patient_probs_reshaped))
        patient_lls.append(patient_ll)
        healthy_lls.append(healthy_ll)

        test_true_labels.append(1)
    for i, healthy_test_prob in enumerate(healthy_test_probs[:len(disease_validation_probs)]):
        healthy_test_prob_reshaped = healthy_test_prob.flatten().reshape(-1, 1)

        # Calculate average log-likelihood for this healthy subject across all their probability values
        patient_ll = np.mean(final_patient_gmm.score_samples(healthy_test_prob_reshaped))
        healthy_ll = np.mean(final_healthy_gmm.score_samples(healthy_test_prob_reshaped))
        patient_lls.append(patient_ll)
        healthy_lls.append(healthy_ll)

        test_true_labels.append(0)  # True label is healthy
    find_and_display_uncertainty_threshold(patient_lls, healthy_lls, test_true_labels, chosen_components, k_fold_disease, model_config_str, to_savefig=to_savefig)

    return


def plot_bic_scores_per_models(patient_train_probs, healthy_probs, final_patient_gmm, final_healthy_gmm, patient_train_data, healthy_train_data,
                               patient_train_probs_flat, healthy_train_probs_flat, covariance_type, chosen_components, k_fold_disease, model_config_str, to_savefig=True):
    # Define range of components to test (fewer components for 1D data)
    max_components = min(6, min(len(patient_train_probs), len(healthy_probs)) // 2)
    n_components_range = range(1, max_components + 1)

    # BIC analysis for patient probability data
    print("\n--- BIC Analysis for Patient Probability Data ---")
    patient_bic_scores = []
    patient_gmm_models = {}

    def calc_mean_bic_score(n_comps, probabilities):
        bics_scores = []
        gmms = []
        for prob in probabilities:
            prob_reshaped = prob.flatten().reshape(-1, 1)
            gmm = GaussianMixture(n_components=n_comps, random_state=42, covariance_type=covariance_type)
            gmm.fit(prob_reshaped)
            bics_scores.append(gmm.bic(prob_reshaped) / len(prob_reshaped))  # Normalize by number of samples
            gmms.append(gmm)
        return np.mean(bics_scores), gmms

    for n_components in n_components_range:
        bic_score, patient_gmm_model = calc_mean_bic_score(n_components, patient_train_probs)
        patient_bic_scores.append(bic_score)
        patient_gmm_models[n_components] = patient_gmm_model
        # gmm = GaussianMixture(n_components=n_components, random_state=42, covariance_type='full')
        # gmm.fit(patient_train_data)
        # bic_score = gmm.bic(patient_train_data)
        # patient_bic_scores.append(bic_score)
        # patient_gmm_models[n_components] = gmm
        print(f"Components: {n_components:2d}, BIC: {bic_score:.2f}")

    # BIC analysis for healthy probability data
    print("\n--- BIC Analysis for Healthy Probability Data ---")
    healthy_bic_scores = []
    healthy_gmm_models = {}

    for n_components in n_components_range:
        bic_score, healthy_gmm_model = calc_mean_bic_score(n_components, healthy_probs)
        healthy_bic_scores.append(bic_score)
        healthy_gmm_models[n_components] = healthy_gmm_model
        # gmm = GaussianMixture(n_components=n_components, random_state=42, covariance_type='full')
        # gmm.fit(healthy_train_data)
        # bic_score = gmm.bic(healthy_train_data)
        # healthy_bic_scores.append(bic_score)
        # healthy_gmm_models[n_components] = gmm
        print(f"Components: {n_components:2d}, BIC: {bic_score:.2f}")

    # Find optimal number of components for each
    optimal_patient_components = n_components_range[np.argmin(patient_bic_scores)]
    optimal_healthy_components = n_components_range[np.argmin(healthy_bic_scores)]

    print(f"\nOptimal components - Patient: {optimal_patient_components}, Healthy: {optimal_healthy_components}")

    # Plot BIC scores and probability distributions
    plt.figure(figsize=(12, 5))
    # BIC scores plot
    plt.subplot(1, 2, 1)
    plt.plot(n_components_range, patient_bic_scores, 'bo-', label='Patient Data')
    plt.plot(n_components_range, healthy_bic_scores, 'ro-', label='Healthy Data')
    # Add horizontal line for bic score of final GMMs
    plt.axhline(y=final_patient_gmm.bic(patient_train_data) / len(patient_train_data), color='blue', linestyle='--', label='Final Patient GMM BIC')
    plt.axhline(y=final_healthy_gmm.bic(healthy_train_data) / len(healthy_train_data), color='red', linestyle='--', label='Final Healthy GMM BIC')
    plt.xlabel('Number of Components')
    plt.ylabel('Normalized BIC Score')
    plt.title('Normalized BIC Scores for 1D GMM Component Selection')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot fitted GMMs
    plt.subplot(1, 2, 2)
    x_range = np.linspace(0, 1, 1000).reshape(-1, 1)
    patient_gmm_pdf = np.exp(final_patient_gmm.score_samples(x_range))
    healthy_gmm_pdf = np.exp(final_healthy_gmm.score_samples(x_range))
    plt.plot(x_range.flatten(), patient_gmm_pdf, 'b-', linewidth=2, label=f'Patient GMM ({chosen_components} comp.)')
    plt.plot(x_range.flatten(), healthy_gmm_pdf, 'r-', linewidth=2, label=f'Healthy GMM ({chosen_components} comp.)')
    plt.hist(patient_train_probs_flat, bins=50, alpha=0.3, density=True, color='blue')
    plt.hist(healthy_train_probs_flat, bins=50, alpha=0.3, density=True, color='red')
    plt.xlabel('Probability Values')
    plt.ylabel('Density')
    plt.title('Fitted GMM Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if to_savefig:
        plt.savefig(os.path.join(INFERENCE_GMM_BIC_PLOTS_V2_DIR, f'gmm_bic_analysis_fold-{k_fold_disease}_components-{chosen_components}_{model_config_str}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    return patient_gmm_models, healthy_gmm_models


def plot_gmm_components_per_patient_and_final(patient_gmm_models, healthy_gmm_models, final_patient_gmm,
                                              final_healthy_gmm, chosen_components, k_fold_disease, model_config_str, to_savefig=True):
    # Comparing between patient_gmm_models[chosen_components] and final_patient_gmm (and also for healthy)
    def calculate_gmm_means_covs_per_component(gmm_models):
        avg_means_of_components = []
        avg_covs_of_components = []
        avg_weights_of_components = []
        for gmm_model in gmm_models:
            avg_means_of_components.append([comp for comp in gmm_model.means_])
            avg_covs_of_components.append([comp[0] for comp in gmm_model.covariances_])
            avg_weights_of_components.append(gmm_model.weights_)
            if len(avg_covs_of_components[-1]) == 1:
                # make it repeat len(avg_means_of_components[-1]) times
                avg_covs_of_components[-1] = np.repeat(avg_covs_of_components[-1], len(avg_means_of_components[-1]))
        # calculate and sort avg_means_of_components, but keep avg_covs_of_components in the corresponding order
        avg_means_of_components = np.mean(np.array(avg_means_of_components), axis=0)[:, 0]
        try:
            avg_covs_of_components = np.mean(np.array(avg_covs_of_components), axis=0)[:, 0]
        except IndexError:
            avg_covs_of_components = np.mean(np.array(avg_covs_of_components), axis=0)
        avg_weights_of_components = np.mean(np.array(avg_weights_of_components), axis=0)
        # Sort means and covariances together
        sorted_indices = np.argsort(avg_means_of_components)
        avg_means_of_components = avg_means_of_components[sorted_indices]
        avg_covs_of_components = avg_covs_of_components[sorted_indices]
        avg_weights_of_components = avg_weights_of_components[sorted_indices]
        return avg_means_of_components, avg_covs_of_components, avg_weights_of_components

    # Calculate on mean of GMMs per patient/subject
    patient_means, patient_covs, patient_weights = calculate_gmm_means_covs_per_component(patient_gmm_models[chosen_components])
    healthy_means, healthy_covs, healthy_weights = calculate_gmm_means_covs_per_component(healthy_gmm_models[chosen_components])
    final_patient_means, final_patient_covs, final_patient_weights = calculate_gmm_means_covs_per_component([final_patient_gmm])
    final_healthy_means, final_healthy_covs, final_healthy_weights = calculate_gmm_means_covs_per_component([final_healthy_gmm])

    # Display the means and covariances in a figure
    # Plotting on a horizontal axis
    fig, ax = plt.subplots(figsize=(10, 3), dpi=600)
    y_offsets = {
        'patient': 0.3,
        'final_patient': 0.2,
        'healthy': -0.2,
        'final_healthy': -0.3
    }

    # Helper to plot error bars
    def plot_gmm(ax, means, covs, weights, y_offset, label, color, marker='o', linestyle=''):
        y = np.full_like(means, y_offset)
        y += np.linspace(-0.02, 0.02, len(means))  # prevent overlap
        std = np.sqrt(covs)
        sizes = 200 * weights  # Scale factor for visibility (tune as needed)
        # Plot error bars and scatter
        ax.errorbar(means, y, xerr=std, fmt='none', ecolor=color, capsize=3, linestyle=linestyle)
        ax.scatter(means, y, s=sizes, label=label, color=color, marker=marker, alpha=0.8, edgecolors='black')
        # Add weight as text next to each point
        for xi, yi, wi in zip(means, y, weights):
            ax.text(xi, yi + 0.025, f'{wi:.2f}', ha='center', va='bottom', fontsize=8, color='black')

    # Use consistent color scheme
    plot_gmm(ax, patient_means, patient_covs, patient_weights, y_offsets['patient'], 'Avg Patient GMMs', color='blue')
    plot_gmm(ax, final_patient_means, final_patient_covs, final_patient_weights, y_offsets['final_patient'], 'Final Patient GMM', color='dodgerblue', marker='x')
    plot_gmm(ax, healthy_means, healthy_covs, healthy_weights, y_offsets['healthy'], 'Avg Healthy GMMs', color='darkorange')
    plot_gmm(ax, final_healthy_means, final_healthy_covs, final_healthy_weights, y_offsets['final_healthy'], 'Final Healthy GMM', color='goldenrod', marker='x')
    # Styling
    ax.set_xlim(-0.25, 1.25)
    ax.set_yticks([])
    ax.set_xlabel("Component Mean (0–1)")
    ax.set_title("GMM Component Means with Variances (± std)")
    ax.legend(loc='right', ncol=2, framealpha=0.9)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    plt.tight_layout()
    # save plot
    if to_savefig:
        plt.savefig(os.path.join(INFERENCE_GMM_MEAN_COV_PLOTS_V2_DIR, f'gmm_means_covs_fold-{k_fold_disease}_components-{chosen_components}_{model_config_str}.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_per_person_gmm_components(patient_gmm_models, healthy_gmm_models, patient_test_probs,
                                   healthy_test_probs, covariance_type, chosen_components, size=20,
                                   only_per_person_gmms=False, extra_colors=None, savefig_path=None):
    # Same plot but with 20 random samples from each GMM model in patient:
    def display_per_person_mean_and_std(gmm_models, size=size, title_extra=''):
        if len(gmm_models) > size:
            chosen_patient_gmms = np.random.choice(gmm_models, size=size, replace=False)
        else:
            chosen_patient_gmms = gmm_models
        fig, ax = plt.subplots(figsize=(10, 3), dpi=600)
        y_offsets = np.linspace(-0.4, 0.4, len(chosen_patient_gmms))
        for i, gmm in enumerate(chosen_patient_gmms):
            means = gmm.means_[:, 0]
            if covariance_type == 'full':
                stds = np.sqrt(gmm.covariances_[:, 0])[:, 0]
            elif covariance_type == 'diag':
                stds = np.sqrt(gmm.covariances_[:, 0])
            elif covariance_type == 'tied':
                stds = np.sqrt(gmm.covariances_[0, 0])
            else:
                raise ValueError(f"Unsupported covariance type: {covariance_type}")
            y = np.full_like(means, y_offsets[i])
            # y += np.linspace(-0.02, 0.02, len(means))
            if extra_colors is not None:
                color = extra_colors[title_extra][i]
            else:
                color = 'blue'
            ax.errorbar(means, y, xerr=stds, fmt='o', label=f'Patient GMM {i + 1}', color=color, capsize=3)
        # Styling
        ax.set_xlim(-0.25, 1.25)
        ax.set_yticks([])
        ax.set_xlabel("Component Mean (0–1)")
        ax.set_title(f"Random Samples from {title_extra} GMMs with Variances (± std)")
        # ax.legend(loc='upper right', ncol=2, framealpha=0.9)
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        plt.tight_layout()
        if savefig_path is not None:
            plt.savefig(savefig_path.format(title_extra.lower()))
        plt.show()

    display_per_person_mean_and_std(patient_gmm_models[chosen_components], title_extra='Patient')
    display_per_person_mean_and_std(healthy_gmm_models[chosen_components], title_extra='Healthy')

    if only_per_person_gmms:
        return

    # Fit on 4 of the test patients and 4 of the healthy subjects
    patient_test_gmms = []
    healthy_test_gmms = []
    for i in range(4):
        # fit gmm:
        patient_gmm = GaussianMixture(n_components=chosen_components, random_state=42, covariance_type=covariance_type)
        patient_gmm.fit(patient_test_probs[i].reshape(-1, 1))
        healthy_gmm = GaussianMixture(n_components=chosen_components, random_state=42, covariance_type=covariance_type)
        healthy_gmm.fit(healthy_test_probs[i].reshape(-1, 1))
        patient_test_gmms.append(patient_gmm)
        healthy_test_gmms.append(healthy_gmm)
    # Display the means and covariances for test patients
    display_per_person_mean_and_std(patient_test_gmms, title_extra='TEST Patient')
    display_per_person_mean_and_std(healthy_test_gmms, title_extra='TEST Healthy')


def get_svm_rbf_classification_cm(patient_train_vectors, healthy_vectors, patient_test_vectors, healthy_test_vectors,
                                  vector_representation_bins, k_fold_disease, model_config_str, to_savefig=True, plot_confidence=True):
    patient_train_vectors = patient_train_vectors[:, vector_representation_bins // 2:]
    healthy_vectors = healthy_vectors[:, vector_representation_bins // 2:]
    patient_test_vectors = patient_test_vectors[:, vector_representation_bins // 2:]
    healthy_test_vectors = healthy_test_vectors[:, vector_representation_bins // 2:]
    X_train = np.concatenate([patient_train_vectors, healthy_vectors], axis=0)
    X_test = np.concatenate([patient_test_vectors, healthy_test_vectors], axis=0)
    y_train = np.array([1] * len(patient_train_vectors) + [0] * len(healthy_vectors))
    y_test = np.array([1] * len(patient_test_vectors) + [0] * len(healthy_test_vectors))
    # from sklearn.metrics.pairwise import rbf_kernel
    # gamma = 1.0 / (2 * np.var(X_train))  # A common heuristic
    # # Compute RBF kernel matrix
    # X_train_rbf = rbf_kernel(X_train, X_train, gamma=gamma)
    # X_test_rbf = rbf_kernel(X_test, X_train, gamma=gamma)
    # # Apply KNN on the transformed features
    # knn_rbf = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
    # knn_rbf.fit(X_train_rbf, y_train)
    # y_pred_knn_rbf = knn_rbf.predict(X_test_rbf)
    # cm_knn_rbf = confusion_matrix(y_test, y_pred_knn_rbf)
    svm_rbf = SVC(kernel='rbf', class_weight='balanced', random_state=42, probability=True)
    svm_rbf.fit(X_train, y_train)
    y_pred_svm_rbf = svm_rbf.predict(X_test)
    cm_svm_rbf = confusion_matrix(y_test, y_pred_svm_rbf)

    if plot_confidence:
        # === Add Confidence Calculation ===
        probas = svm_rbf.predict_proba(X_test)        # Shape: (n_samples, 2)
        confidences = np.max(probas, axis=1)          # Highest class probability
        confidence_thresholds = sorted(list(set(confidences)))  # Unique confidence values for thresholds
        # calculate accuracies across thresholds
        y_pred_svm_rbf = svm_rbf.predict(X_test)
        accuracies = []
        num_remaining = []
        for threshold in confidence_thresholds:
            # Mask to keep only predictions above the confidence threshold
            keep_mask = confidences >= threshold

            if np.sum(keep_mask) == 0:
                continue  # Skip thresholds that remove all predictions

            filtered_preds = y_pred_svm_rbf[keep_mask]
            filtered_true = y_test[keep_mask]

            acc = np.mean(filtered_preds == filtered_true)
            accuracies.append(acc)
            num_remaining.append(np.sum(keep_mask))

        # === PLOT ===
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(confidence_thresholds[:len(accuracies)], accuracies, color='blue', label='Accuracy')
        ax1.set_xlabel('Confidence Threshold')
        ax1.set_ylabel('Accuracy', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_ylim(0, 1.05)
        ax2 = ax1.twinx()
        ax2.plot(confidence_thresholds[:len(num_remaining)], num_remaining, color='red', linestyle='--',
                 label='Remaining Samples')
        ax2.set_ylabel('Number of Remaining Samples', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        plt.title('Accuracy vs Confidence Threshold (SVM-RBF)')
        fig.tight_layout()
        plt.grid(True)
        if to_savefig:
            plt.savefig(os.path.join(INFERENCE_SVM_EXTRA_PLOTS_V2_DIR, f'svm_rbf_confidence_fold-{k_fold_disease}_{model_config_str}.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # === CHOOSE 4 CONFIDENCE THRESHOLDS ===
        thresholds_to_plot = [0.0,  # No filtering
                              np.percentile(confidences, 25),
                              np.percentile(confidences, 50),
                              np.percentile(confidences, 75)]
        titles = ['No Threshold',
                  'Threshold ≥ 25th percentile',
                  'Threshold ≥ 50th percentile (Median)',
                  'Threshold ≥ 75th percentile']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()  # Flatten for easy indexing
        for i, (threshold, title) in enumerate(zip(thresholds_to_plot, titles)):
            keep_mask = confidences >= threshold
            if np.sum(keep_mask) == 0:
                axes[i].axis('off')
                axes[i].set_title(f"{title}\n(No samples)")
                continue
            filtered_preds = y_pred_svm_rbf[keep_mask]
            filtered_true = y_test[keep_mask]
            cm = confusion_matrix(filtered_true, filtered_preds)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Healthy', 'Patient'],
                        yticklabels=['Healthy', 'Patient'],
                        ax=axes[i], cbar=False)
            axes[i].set_title(f"{title}\n(n={np.sum(keep_mask)})")
            axes[i].set_xlabel("Predicted Label")
            axes[i].set_ylabel("True Label")
        plt.suptitle("Confusion Matrices at Different Confidence Thresholds", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        if to_savefig:
            plt.savefig(os.path.join(INFERENCE_SVM_EXTRA_PLOTS_V2_DIR, f'svm_rbf_confusion_matrices_fold-{k_fold_disease}_{model_config_str}.png'), dpi=300, bbox_inches='tight')
        plt.show()

    return cm_svm_rbf


def plot_gmm_classification_multi_threshold(patient_test_probs, healthy_test_probs, final_patient_gmm, final_healthy_gmm,
                                            cm_svm_rbf, chosen_components, k_fold_disease, model_config_str,
                                            to_savefig=True, thresholds=[0, 0.015, 0.03]):
    """
    Plot GMM classification results for multiple thresholds in a single figure with 4 subplots.
    Also displays percentiles and current thresholds information.
    """

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

    # Calculate percentiles
    all_diffs = calculate_all_differences()
    abs_diffs = np.abs(all_diffs)

    percentiles = [90, 75, 50, 25]
    percentiles = [100 - p for p in percentiles]  # Convert to 100 - percentile
    percentile_values = np.percentile(abs_diffs, percentiles)

    # Combine original thresholds with percentile thresholds
    all_thresholds = thresholds + percentile_values.tolist()

    # Create figure with appropriate number of subplots (original + percentiles + SVM)
    total_plots = len(thresholds) + len(percentiles) + 1  # +1 for SVM
    rows = (total_plots + 3) // 4  # Calculate rows needed (4 columns max)
    cols = min(4, total_plots)

    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    if total_plots == 1:
        axs = [axs]
    else:
        axs = axs.flatten() if rows > 1 else axs

    plot_idx = 0

    # 1. First plot: No threshold (normal)
    no_threshold_idx = [i for i, t in enumerate(thresholds) if t == 0]
    if no_threshold_idx:
        threshold = thresholds[no_threshold_idx[0]]
        gmm_predictions, y_test = get_gmm_predictions(threshold)

        if len(gmm_predictions) == 0:
            cm_gmm = np.zeros((2, 2), dtype=int)
            accuracy = 0.0
        else:
            cm_gmm = confusion_matrix(y_test, gmm_predictions)
            accuracy = np.mean(gmm_predictions == y_test)

        sns.heatmap(cm_gmm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Healthy', 'Patient'],
                    yticklabels=['Healthy', 'Patient'],
                    ax=axs[plot_idx])

        axs[plot_idx].set_title(f'GMM - No Threshold\nACC: {accuracy:.3f}')
        axs[plot_idx].set_ylabel('True Label')
        axs[plot_idx].set_xlabel('Predicted Label')
        plot_idx += 1

    # 2. Then plot: Original thresholds (excluding 0 threshold)
    for threshold in thresholds:
        if threshold > 0:
            gmm_predictions, y_test = get_gmm_predictions(threshold)

            if len(gmm_predictions) == 0:
                cm_gmm = np.zeros((2, 2), dtype=int)
                accuracy = 0.0
            else:
                cm_gmm = confusion_matrix(y_test, gmm_predictions)
                accuracy = np.mean(gmm_predictions == y_test)

            sns.heatmap(cm_gmm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Healthy', 'Patient'],
                        yticklabels=['Healthy', 'Patient'],
                        ax=axs[plot_idx])

            axs[plot_idx].set_title(f'GMM - Threshold: {threshold}\nACC: {accuracy:.3f}')
            axs[plot_idx].set_ylabel('True Label')
            axs[plot_idx].set_xlabel('Predicted Label')
            plot_idx += 1

    # 3. Then plot: SVM results
    svm_accuracy = np.trace(cm_svm_rbf) / np.sum(cm_svm_rbf)
    sns.heatmap(cm_svm_rbf, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Healthy', 'Patient'],
                yticklabels=['Healthy', 'Patient'],
                ax=axs[plot_idx])
    axs[plot_idx].set_title(f'SVM-RBF\nACC: {svm_accuracy:.3f}')
    axs[plot_idx].set_ylabel('True Label')
    axs[plot_idx].set_xlabel('Predicted Label')
    plot_idx += 1

    # 4. Lastly plot: Percentile thresholds
    for percentile, threshold in zip(percentiles, percentile_values):
        gmm_predictions, y_test = get_gmm_predictions(threshold)

        if len(gmm_predictions) == 0:
            cm_gmm = np.zeros((2, 2), dtype=int)
            accuracy = 0.0
        else:
            cm_gmm = confusion_matrix(y_test, gmm_predictions)
            accuracy = np.mean(gmm_predictions == y_test)

        sns.heatmap(cm_gmm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=['Healthy', 'Patient'],
                    yticklabels=['Healthy', 'Patient'],
                    ax=axs[plot_idx])

        axs[plot_idx].set_title(f'GMM - {100 - percentile}th Percentile\nThreshold: {threshold:.4f}, ACC: {accuracy:.3f}')
        axs[plot_idx].set_ylabel('True Label')
        axs[plot_idx].set_xlabel('Predicted Label')
        plot_idx += 1

    # Hide any unused subplots
    for idx in range(plot_idx, len(axs)):
        axs[idx].set_visible(False)

    plt.suptitle(f'Classification Results Comparison (Fold {k_fold_disease}, Components: {chosen_components})',
                 fontsize=16, y=0.98)
    plt.tight_layout()

    if to_savefig:
        plt.savefig(os.path.join(INFERENCE_CONFUSION_MATRIX_V2_DIR,
                                 f'confusion_matrix_multi_threshold_fold-{k_fold_disease}_components-{chosen_components}_{model_config_str}.png'),
                    dpi=300, bbox_inches='tight')
    plt.show()

def find_and_display_uncertainty_threshold(patient_lls, healthy_lls, test_true_labels, chosen_components,
                                           k_fold_disease, model_config_str, to_savefig=True):
    # Convert to numpy arrays for easier manipulation
    patient_lls = np.array(patient_lls)
    healthy_lls = np.array(healthy_lls)
    test_true_labels = np.array(test_true_labels)

    # Save the function arguments in pickle file for caching
    if to_savefig:
        cache_file = os.path.join(INFERENCE_UNCERTAINTY_THRESHOLD_CACHE_V2_DIR, f'params_uncertainty_threshold_fold-{k_fold_disease}_components-{chosen_components}_{model_config_str}.pkl')
        if not os.path.exists(cache_file):
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'patient_lls': patient_lls,
                    'healthy_lls': healthy_lls,
                    'test_true_labels': test_true_labels,
                    'chosen_components': chosen_components,
                    'k_fold_disease': k_fold_disease,
                    'model_config_str': model_config_str
                }, f)

    # Calculate difference scores (patient_ll - healthy_ll)
    # Higher values indicate more likely to be patient
    diff_scores = patient_lls - healthy_lls

    def calculate_metrics(y_true, y_pred):
        """Calculate TPR, TNR, F1-Score, and Accuracy"""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # Sensitivity/Recall
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)

        return tpr, tnr, f1, accuracy

    # Define range of uncertainty thresholds
    # We'll use percentiles of the absolute difference scores
    abs_diff_scores = np.abs(diff_scores)
    uncertainty_thresholds = np.percentile(abs_diff_scores, np.linspace(0, 90, 50))

    # Storage for results
    results = {
        'uncertainty_threshold': [],
        'best_classification_threshold': [],
        'best_tpr': [],
        'best_tnr': [],
        'best_f1': [],
        'best_accuracy': [],
        'samples_remaining_ratio': []
    }

    print("Optimizing thresholds...")
    print("Uncertainty Threshold | Samples Remaining | Best Accuracy | Best F1 | Best TPR | Best TNR")
    print("-" * 90)

    for uncertain_thresh in uncertainty_thresholds:
        # Find samples that are NOT uncertain (above the uncertainty threshold)
        certain_mask = abs_diff_scores >= uncertain_thresh

        if np.sum(certain_mask) < 10:  # Skip if too few samples remain
            continue

        # Get the certain samples
        certain_diff_scores = diff_scores[certain_mask]
        certain_true_labels = test_true_labels[certain_mask]

        # Define range of classification thresholds for the certain samples
        min_score = np.min(certain_diff_scores)
        max_score = np.max(certain_diff_scores)
        classification_thresholds = np.linspace(min_score, max_score, 100)

        best_accuracy = 0
        best_metrics = None
        best_class_thresh = None

        # Find the best classification threshold for this uncertainty threshold
        for class_thresh in classification_thresholds:
            # Predict: if diff_score > class_thresh, predict patient (1), else healthy (0)
            predictions = (certain_diff_scores > class_thresh).astype(int)

            if len(np.unique(predictions)) < 2:  # Skip if all predictions are the same
                continue

            tpr, tnr, f1, accuracy = calculate_metrics(certain_true_labels, predictions)

            # Use accuracy as primary metric for optimization
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_metrics = (tpr, tnr, f1, accuracy)
                best_class_thresh = class_thresh

        if best_metrics is not None:
            samples_remaining_ratio = np.sum(certain_mask) / len(test_true_labels)

            results['uncertainty_threshold'].append(uncertain_thresh)
            results['best_classification_threshold'].append(best_class_thresh)
            results['best_tpr'].append(best_metrics[0])
            results['best_tnr'].append(best_metrics[1])
            results['best_f1'].append(best_metrics[2])
            results['best_accuracy'].append(best_metrics[3])
            results['samples_remaining_ratio'].append(samples_remaining_ratio)

            print(
                f"{uncertain_thresh:17.4f} | {samples_remaining_ratio:15.3f} | {best_metrics[3]:11.3f} | {best_metrics[2]:7.3f} | {best_metrics[0]:8.3f} | {best_metrics[1]:8.3f}")

    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Accuracy vs Uncertainty Threshold
    ax1.plot(results['uncertainty_threshold'], results['best_accuracy'], 'b-o', linewidth=2, markersize=6)
    ax1.set_xlabel('Uncertainty Threshold')
    ax1.set_ylabel('Best Accuracy')
    ax1.set_title('Accuracy vs Uncertainty Threshold After Uncertainty Filtering')
    ax1.grid(True, alpha=0.3)

    # Plot 2: F1-Score vs Uncertainty Threshold
    ax2.plot(results['uncertainty_threshold'], results['best_f1'], 'g-o', linewidth=2, markersize=6)
    ax2.set_xlabel('Uncertainty Threshold')
    ax2.set_ylabel('Best F1-Score')
    ax2.set_title('F1-Score vs Uncertainty Threshold After Uncertainty Filtering')
    ax2.grid(True, alpha=0.3)

    # Plot 3: TPR and TNR vs Uncertainty Threshold
    ax3.plot(results['uncertainty_threshold'], results['best_tpr'], 'r-o', linewidth=2, markersize=6, label='TPR (Sensitivity)')
    ax3.plot(results['uncertainty_threshold'], results['best_tnr'], 'm-o', linewidth=2, markersize=6, label='TNR (Specificity)')
    ax3.set_xlabel('Uncertainty Threshold')
    ax3.set_ylabel('Score')
    ax3.set_title('TPR and TNR vs Uncertainty Threshold After Uncertainty Filtering')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: All metrics together
    ax4.plot(results['uncertainty_threshold'], results['best_accuracy'], 'b-o', linewidth=2, markersize=4, label='Accuracy')
    ax4.plot(results['uncertainty_threshold'], results['best_f1'], 'g-o', linewidth=2, markersize=4, label='F1-Score')
    ax4.plot(results['uncertainty_threshold'], results['best_tpr'], 'r-o', linewidth=2, markersize=4, label='TPR')
    ax4.plot(results['uncertainty_threshold'], results['best_tnr'], 'm-o', linewidth=2, markersize=4, label='TNR')
    ax4.set_xlabel('Uncertainty Threshold')
    ax4.set_ylabel('Score')
    ax4.set_title('All Metrics vs Uncertainty Threshold After Uncertainty Filtering')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    if to_savefig:
        plt.savefig(os.path.join(INFERENCE_UNCERTAINTY_THRESHOLD_V2_DIR, f'uncertainty_threshold_fold-{k_fold_disease}_components-{chosen_components}_{model_config_str}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Create heatmap-style plot with uncertainty threshold vs accuracy, colored by samples remaining ratio
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    # Create scatter plot with color mapping
    scatter = ax.scatter(results['uncertainty_threshold'], results['best_accuracy'],
                         c=results['samples_remaining_ratio'],
                         s=100, cmap='viridis', alpha=0.8, edgecolors='black', linewidth=0.5)
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Samples Remaining Ratio', rotation=270, labelpad=20, fontsize=12)
    # Customize the plot
    ax.set_xlabel('Uncertainty Threshold', fontsize=12)
    ax.set_ylabel('Best Accuracy', fontsize=12)
    ax.set_title('Accuracy vs Uncertainty Threshold\n(Color indicates fraction of samples remaining)', fontsize=14)
    ax.grid(True, alpha=0.3)
    # Add some annotations for key points
    best_accuracy_idx = np.argmax(results['best_accuracy'])
    ax.annotate(f'Best Accuracy\n({results["best_accuracy"][best_accuracy_idx]:.3f})',
                xy=(results['uncertainty_threshold'][best_accuracy_idx], results['best_accuracy'][best_accuracy_idx]),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    plt.tight_layout()
    if to_savefig:
        plt.savefig(os.path.join(INFERENCE_UNCERTAINTY_THRESHOLD_V2_DIR, f'uncertainty_threshold_fold-{k_fold_disease}_components-{chosen_components}_{model_config_str}.png'), dpi=300, bbox_inches='tight')
    plt.show()

    # Find and display optimal points
    best_accuracy_idx = np.argmax(results['best_accuracy'])
    best_f1_idx = np.argmax(results['best_f1'])

    print("\n" + "=" * 80)
    print("OPTIMAL RESULTS:")
    print("=" * 80)
    print(f"Best Accuracy: {results['best_accuracy'][best_accuracy_idx]:.4f}")
    print(f"  - Uncertainty Threshold: {results['uncertainty_threshold'][best_accuracy_idx]:.4f}")
    print(f"  - Classification Threshold: {results['best_classification_threshold'][best_accuracy_idx]:.4f}")
    print(f"  - Samples Remaining: {results['samples_remaining_ratio'][best_accuracy_idx]:.3f}")
    print(f"  - TPR: {results['best_tpr'][best_accuracy_idx]:.4f}")
    print(f"  - TNR: {results['best_tnr'][best_accuracy_idx]:.4f}")
    print(f"  - F1-Score: {results['best_f1'][best_accuracy_idx]:.4f}")

    print(f"\nBest F1-Score: {results['best_f1'][best_f1_idx]:.4f}")
    print(f"  - Uncertainty Threshold: {results['uncertainty_threshold'][best_f1_idx]:.4f}")
    print(f"  - Classification Threshold: {results['best_classification_threshold'][best_f1_idx]:.4f}")
    print(f"  - Samples Remaining: {results['samples_remaining_ratio'][best_f1_idx]:.3f}")
    print(f"  - TPR: {results['best_tpr'][best_f1_idx]:.4f}")
    print(f"  - TNR: {results['best_tnr'][best_f1_idx]:.4f}")
    print(f"  - Accuracy: {results['best_accuracy'][best_f1_idx]:.4f}")

    # Additional analysis: Show the trade-off
    print("\n" + "=" * 80)
    print("TRADE-OFF ANALYSIS:")
    print("=" * 80)
    print("As uncertainty threshold increases (more samples filtered out):")
    print("- Samples remaining decreases")
    print("- Classification accuracy on remaining samples typically increases")
    print("- But overall coverage decreases")

    # Calculate some statistics
    high_coverage_mask = np.array(results['samples_remaining_ratio']) > 0.8
    if np.any(high_coverage_mask):
        high_coverage_acc = np.array(results['best_accuracy'])[high_coverage_mask]
        print(f"\nWith >80% sample coverage:")
        print(f"  - Best accuracy: {np.max(high_coverage_acc):.4f}")
        print(f"  - Average accuracy: {np.mean(high_coverage_acc):.4f}")

    low_coverage_mask = np.array(results['samples_remaining_ratio']) < 0.5
    if np.any(low_coverage_mask):
        low_coverage_acc = np.array(results['best_accuracy'])[low_coverage_mask]
        print(f"\nWith <50% sample coverage:")
        print(f"  - Best accuracy: {np.max(low_coverage_acc):.4f}")
        print(f"  - Average accuracy: {np.mean(low_coverage_acc):.4f}")
    return


def plot_gmms_per_person_outliers(patient_gmm_models, healthy_gmm_models, patient_train_probs, healthy_probs,
                                  patient_test_probs, healthy_test_probs, covariance_type, chosen_components,
                                  k_fold_disease, model_config_str, to_savefig=True):
    from sklearn.ensemble import IsolationForest
    patient_gmms = patient_gmm_models[chosen_components]
    healthy_gmms = healthy_gmm_models[chosen_components]

    def flatten_means(gmm):
        return gmm.means_.flatten()

    patient_means = np.array([flatten_means(gmm) for gmm in patient_gmms])
    healthy_means = np.array([flatten_means(gmm) for gmm in healthy_gmms])

    outlier_type = 1
    if outlier_type == 0:
        iso_patient = IsolationForest(contamination=0.1, random_state=42)
        iso_healthy = IsolationForest(contamination=0.1, random_state=42)
        outlier_labels_patient = iso_patient.fit_predict(patient_means)
        outlier_labels_healthy = iso_healthy.fit_predict(healthy_means)
    elif outlier_type == 1:
        patient_means_r = patient_means.max(axis=1)
        healthy_means_r = healthy_means.max(axis=1)
        def get_outlier_labels(data, lower_q=0.10, upper_q=0.90):
            lower = np.quantile(data, lower_q)
            upper = np.quantile(data, upper_q)
            labels = np.where((data >= lower) & (data <= upper), 1, -1)
            return labels
        # Get label arrays
        outlier_labels_patient = get_outlier_labels(patient_means_r)
        outlier_labels_healthy = get_outlier_labels(healthy_means_r)
    else:
        patient_means_r = patient_means.max(axis=1)
        healthy_means_r = healthy_means.max(axis=1)

        patient_main_mean = patient_means_r.mean()
        healthy_main_mean = healthy_means_r.mean()
        middle_point = (patient_main_mean + healthy_main_mean) / 2

        # define all samples as outlier if they cross that middle point
        outlier_labels_patient = np.where(
            (middle_point < patient_means_r), 1, -1)
            # (middle_point < patient_means_r) & (patient_means_r < 2 * patient_main_mean - middle_point), 1, -1)
        outlier_labels_healthy = np.where(
            (healthy_means_r < middle_point), 1, -1)
            # (2 * healthy_main_mean - middle_point < healthy_means_r) & (healthy_means_r < middle_point), 1, -1)

    color_dict = dict()
    color_dict['Patient'] = ['blue' if x == 1 else 'red' for x in outlier_labels_patient]
    color_dict['Healthy'] = ['blue' if x == 1 else 'red' for x in outlier_labels_healthy]

    if to_savefig:
        savefig_path = os.path.join(INFERENCE_OUTLIER_GMMS_PLOTS_V2_DIR, f'outlier_gmms_{"{}"}_fold-{k_fold_disease}_components-{chosen_components}_{model_config_str}.png')
    else:
        savefig_path = None
    plot_per_person_gmm_components(patient_gmm_models, healthy_gmm_models, patient_test_probs,
                                   healthy_test_probs, covariance_type, chosen_components, size=500, only_per_person_gmms=True,
                                   extra_colors=color_dict, savefig_path=savefig_path)

    filtered_patient_gmms = [gmm for gmm, label in zip(patient_gmms, outlier_labels_patient) if label == 1]
    filtered_healthy_gmms = [gmm for gmm, label in zip(healthy_gmms, outlier_labels_healthy) if label == 1]

    patient_gmm_models[chosen_components] = filtered_patient_gmms
    healthy_gmm_models[chosen_components] = filtered_healthy_gmms

    patient_train_data = np.concatenate([probs.flatten() for probs, label in zip(patient_train_probs, outlier_labels_patient) if label == 1]).reshape(-1, 1)
    healthy_train_data = np.concatenate([probs.flatten() for probs, label in zip(healthy_probs, outlier_labels_healthy) if label == 1]).reshape(-1, 1)

    final_patient_gmm = GaussianMixture(n_components=chosen_components, random_state=42, covariance_type=covariance_type)
    final_patient_gmm.fit(patient_train_data)
    final_healthy_gmm = GaussianMixture(n_components=chosen_components, random_state=42, covariance_type=covariance_type)
    final_healthy_gmm.fit(healthy_train_data)

    return final_patient_gmm, final_healthy_gmm, patient_gmm_models, healthy_gmm_models


def plot_gmm_means_weights_plot(patient_gmm_models, healthy_gmm_models, chosen_components, k_fold_disease, model_config_str, to_savefig=True):
    patient_gmms = patient_gmm_models[chosen_components]
    healthy_gmms = healthy_gmm_models[chosen_components]
    patient_means = np.array([gmm.means_.flatten() for gmm in patient_gmms])
    healthy_means = np.array([gmm.means_.flatten() for gmm in healthy_gmms])

    # make a figure with two sub-figures which are scatter plot of GMM max means and their corresponding weights
    patient_weights = np.array([gmm.weights_.flatten() for gmm in patient_gmms])
    healthy_weights = np.array([gmm.weights_.flatten() for gmm in healthy_gmms])
    # get the max means of the GMMs and their corresponding weights using argmax
    patient_max_indices = np.argmax(patient_means, axis=1)
    healthy_max_indices = np.argmax(healthy_means, axis=1)
    patient_max_means = patient_means[np.arange(len(patient_means)), patient_max_indices]
    healthy_max_means = healthy_means[np.arange(len(healthy_means)), healthy_max_indices]
    patient_max_weights = patient_weights[np.arange(len(patient_weights)), patient_max_indices]
    healthy_max_weights = healthy_weights[np.arange(len(healthy_weights)), healthy_max_indices]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=450)
    axes[0].scatter(patient_max_means, patient_max_weights, c='blue', alpha=0.5, s=50)
    axes[0].set_title("Patient GMM Max Means and Weights")
    axes[0].set_xlabel("Max Mean")
    axes[0].set_ylabel("Weight")
    axes[1].scatter(healthy_max_means, healthy_max_weights, c='blue', alpha=0.5, s=50)
    axes[1].set_title("Healthy GMM Max Means and Weights")
    axes[1].set_xlabel("Max Mean")
    axes[1].set_ylabel("Weight")
    plt.tight_layout()
    if to_savefig:
        plt.savefig(os.path.join(INFERENCE_GMM_MEAN_WEIGHT_SCATTER_V2_DIR, f"gmm_max_means_weights_fold-{k_fold_disease}_components-{chosen_components}_{model_config_str}.png"))
    plt.show()

#
# # TODO: Temp function - remove later
# def inference_classification_model_version2_tmp(trained_model, args, df_bld, df_hlt, test_patient_ids, valid_patient_ids,
#                                             valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs, aaseq_to_ratio, to_ensemble, model_non_trained, device,
#                                             add_ratio_to_vector=False, start_vec_from=0,
#                                             vector_representation_bins=40, num_of_healthy_patients=68, num_of_healthy_test_patients=28, only_all_classifiers=False,
#                                             to_display_mapping=False, samples_size=200,
#                                             k_fold_disease=1, chosen_components=2, to_savefig=False):
#     np.random.seed(42)
#     # Take shuffle and divide the patient 1/3 such that k_fold_disease will choose which 1/3 of patients to take
#     patient_valid_test_ids = np.concatenate([test_patient_ids, valid_patient_ids])
#     patient_train_ids = [x for x in df_bld["patient_id"].unique() if x not in patient_valid_test_ids]
#     np.random.shuffle(patient_train_ids)
#     num_of_patients = len(patient_train_ids)
#     num_of_patients_per_fold = num_of_patients // 3
#     start_ind = (k_fold_disease - 1) * num_of_patients_per_fold
#     end_ind = start_ind + num_of_patients_per_fold
#     fold_patient_test_ids = patient_train_ids[start_ind:end_ind]
#     fold_patient_train_ids = [x for x in patient_train_ids if x not in fold_patient_test_ids]
#     df_bld_validation = df_bld[df_bld["patient_id"].isin(fold_patient_test_ids)]
#     df_train_ids = np.concatenate([patient_valid_test_ids, fold_patient_train_ids])
#     df_bld = df_bld[df_bld["patient_id"].isin(df_train_ids)]
#
#     if len(df_hlt["patient_id"].unique()) >= int(len(df_bld["patient_id"].unique()) * 2.0):
#         num_of_healthy_patients = int(len(df_bld["patient_id"].unique()) * 2.0)
#         num_of_healthy_test_patients = int(len(df_bld["patient_id"].unique()) * 1.0)
#     elif len(df_hlt["patient_id"].unique()) != num_of_healthy_patients:
#         num_of_healthy_patients = len(df_hlt["patient_id"].unique())
#         num_of_healthy_test_patients = int(num_of_healthy_patients * 0.25)
#
#     np.random.seed(42)
#     # make sure that plot dirs exists
#     os.makedirs(INFERENCE_CONFUSION_MATRIX_V2_DIR, exist_ok=True)
#     # os.makedirs(INFERENCE_VECTOR_PLOTS_V2_DIR, exist_ok=True)  # disabled for now
#     os.makedirs(INFERENCE_GMM_MEAN_COV_PLOTS_V2_DIR, exist_ok=True)
#     os.makedirs(INFERENCE_GMM_BIC_PLOTS_V2_DIR, exist_ok=True)
#     os.makedirs(INFERENCE_UNCERTAINTY_THRESHOLD_V2_DIR, exist_ok=True)
#     os.makedirs(INFERENCE_UNCERTAINTY_THRESHOLD_CACHE_V2_DIR, exist_ok=True)
#     model_config_str = get_model_config_str(args)
#
#     # Creating a caching model of the trained model
#     if to_ensemble:  # TODO: This does not work currently! Raising NotImplementedError
#         caching_model = trained_model
#         raise NotImplementedError("Ensemble model inference with non-trained model is not implemented yet.")
#     else:
#         trained_model.eval()
#         if samples_size == 50:
#             caching_model = CVCBasicCachingModel(trained_model, args, device)
#         elif samples_size == 20:
#             caching_model = trained_model
#         else:
#             caching_model = CVCCachingModel(trained_model, args, device)
#             # caching_model = CVCDFCachingModel(trained_model, args, device)  # TODO: This is very very slow for some reason...
#         caching_model.to(device)
#         caching_model.eval()
#
#     # get the patient vectors
#     # possible_seqs = set(np.concatenate([valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs]))
#     patient_test_vectors, patient_test_probs = calc_patient_vectors(df_bld, caching_model, patient_valid_test_ids, vector_representation_bins,
#                                                           add_ratio_to_vector, aaseq_to_ratio, unique_patient_ids=None,
#                                                           possible_seqs=None, start_vec_from=start_vec_from, samples=samples_size)
#
#     # get the train patient vectors
#     patient_train_vectors, patient_train_probs = calc_patient_vectors(df_bld, caching_model, fold_patient_train_ids, vector_representation_bins,
#                                                                       add_ratio_to_vector, aaseq_to_ratio, unique_patient_ids=None,
#                                                                       possible_seqs=None, start_vec_from=start_vec_from, samples=samples_size)
#
#     # get the validation patient vectors
#     disease_validation_vectors, disease_validation_probs = calc_patient_vectors(df_bld_validation, caching_model, fold_patient_test_ids, vector_representation_bins,
#                                                                                 add_ratio_to_vector, aaseq_to_ratio, unique_patient_ids=None,
#                                                                                 possible_seqs=None, start_vec_from=start_vec_from, samples=samples_size)
#
#     # healthy vectors helping data
#     healthy_patients = df_hlt["patient_id"].unique()
#     np.random.shuffle(healthy_patients)
#     healthy_patients = healthy_patients[:num_of_healthy_patients]
#
#     healthy_vectors, healthy_probs = calc_patient_vectors(df_hlt, caching_model, healthy_patients, vector_representation_bins,
#                                            add_ratio_to_vector, aaseq_to_ratio, start_vec_from=start_vec_from, samples=samples_size)
#     healthy_vectors, healthy_test_vectors = (healthy_vectors[:num_of_healthy_patients - num_of_healthy_test_patients],
#                                              healthy_vectors[num_of_healthy_patients - num_of_healthy_test_patients:])
#     healthy_probs, healthy_test_probs = (healthy_probs[:num_of_healthy_patients - num_of_healthy_test_patients],
#                                          healthy_probs[num_of_healthy_patients - num_of_healthy_test_patients:])
#
#     # ============= GMM ADDITION STARTS HERE =============
#
#     print("=" * 60)
#     print("GAUSSIAN MIXTURE MODEL ANALYSIS ON PROBABILITY DISTRIBUTIONS")
#     print("=" * 60)
#
#     # Prepare probability data for GMM fitting
#     # Flatten all probability arrays for each group
#     patient_train_probs_flat = np.concatenate([probs.flatten() for probs in patient_train_probs])
#     healthy_train_probs_flat = np.concatenate([probs.flatten() for probs in healthy_probs])
#
#     # Reshape for sklearn (needs 2D input)
#     patient_train_data = patient_train_probs_flat.reshape(-1, 1)
#     healthy_train_data = healthy_train_probs_flat.reshape(-1, 1)
#
#     print(f"Patient training probability values: {patient_train_data.shape[0]}")
#     print(f"Healthy training probability values: {healthy_train_data.shape[0]}")
#     print(f"Patient probability range: [{patient_train_probs_flat.min():.3f}, {patient_train_probs_flat.max():.3f}]")
#     print(f"Healthy probability range: [{healthy_train_probs_flat.min():.3f}, {healthy_train_probs_flat.max():.3f}]")
#
#     # Choose a compromise number of components that works for both
#     # Choosing chosen_components manually for now:  # Strategy: choose the minimum of the two optimal values, but at least 2
#     # chosen_components = max(2, min(optimal_patient_components, optimal_healthy_components))
#     print(f"Chosen number of components for both distributions: {chosen_components}")
#
#     # Train final GMMs with chosen number of components
#     print(f"\n--- Training Final GMMs with {chosen_components} components ---")
#
#     final_patient_gmm = GaussianMixture(n_components=chosen_components, random_state=42, covariance_type=covariance_type)
#     final_patient_gmm.fit(patient_train_data)
#
#     final_healthy_gmm = GaussianMixture(n_components=chosen_components, random_state=42, covariance_type=covariance_type)
#     final_healthy_gmm.fit(healthy_train_data)
#
#     print(f"Patient GMM - BIC: {final_patient_gmm.bic(patient_train_data):.2f}")
#     print(f"Healthy GMM - BIC: {final_healthy_gmm.bic(healthy_train_data):.2f}")
#
#     gmms = plot_bic_scores_per_models(patient_train_probs, healthy_probs, final_patient_gmm,
#                                       final_healthy_gmm, patient_train_data, healthy_train_data,
#                                       patient_train_probs_flat, healthy_train_probs_flat,
#                                       chosen_components, k_fold_disease, model_config_str, to_savefig=to_savefig)
#     patient_gmm_models, healthy_gmm_models = gmms
#
#     # TODO: Find GMMs out of distribution!
#     from sklearn.ensemble import IsolationForest
#     patient_gmms = patient_gmm_models[chosen_components]
#     healthy_gmms = healthy_gmm_models[chosen_components]
#
#     def flatten_means(gmm):
#         return gmm.means_.flatten()
#     patient_means = np.array([flatten_means(gmm) for gmm in patient_gmms])
#     healthy_means = np.array([flatten_means(gmm) for gmm in healthy_gmms])
#
#     outlier_type = 1
#     if outlier_type == 0:
#         iso_patient = IsolationForest(contamination=0.1, random_state=42)
#         iso_healthy = IsolationForest(contamination=0.1, random_state=42)
#         outlier_labels_patient = iso_patient.fit_predict(patient_means)
#         outlier_labels_healthy = iso_healthy.fit_predict(healthy_means)
#     else:
#         patient_means_r = patient_means.max(axis=1)
#         healthy_means_r = healthy_means.max(axis=1)
#
#         patient_main_mean = patient_means_r.mean()
#         healthy_main_mean = healthy_means_r.mean()
#         middle_point = (patient_main_mean + healthy_main_mean) / 2
#
#         # define all samples as outlier if they cross that middle point
#         outlier_labels_patient = np.where((middle_point < patient_means_r) & (patient_means_r < 2*patient_main_mean-middle_point), 1, -1)
#         outlier_labels_healthy = np.where((2*healthy_main_mean - middle_point < healthy_means_r) & (healthy_means_r < middle_point), 1, -1)
#
#
#     color_dict = dict()
#     color_dict['Patient'] = ['blue' if x == 1 else 'red' for x in outlier_labels_patient]
#     color_dict['Healthy'] = ['blue' if x == 1 else 'red' for x in outlier_labels_healthy]
#
#     plot_per_person_gmm_components(patient_gmm_models, healthy_gmm_models, patient_test_probs,
#                                    healthy_test_probs, chosen_components, size=500, only_per_person_gmms=True, extra_colors=color_dict)
#
#     exit(0)
#     plot_gmm_components_per_patient_and_final(patient_gmm_models, healthy_gmm_models, final_patient_gmm,
#                                               final_healthy_gmm, chosen_components, k_fold_disease, model_config_str, to_savefig=to_savefig)
#
#     plot_per_person_gmm_components(patient_gmm_models, healthy_gmm_models, patient_test_probs,
#                                    healthy_test_probs, chosen_components, size=20)
#
#     cm_svm_rbf = get_svm_rbf_classification_cm(patient_train_vectors, healthy_vectors, patient_test_vectors, healthy_test_vectors, vector_representation_bins)
#
#     plot_gmm_classification_multi_threshold(patient_test_probs, healthy_test_probs, final_patient_gmm, final_healthy_gmm, cm_svm_rbf,
#                                             chosen_components, k_fold_disease, model_config_str, to_savefig=to_savefig)
#
#     # EXTRA - calculate model outputs on valid and test positive sequences
#     if to_display_mapping and model_non_trained is not None:
#         available_models = [trained_model, model_non_trained]
#         models_names = ['Trained Model', 'Non-Trained Model']
#         for model, model_name in zip(available_models, models_names):
#             part_pos_valid = np.random.choice(valid_pos_seqs, size=len(valid_pos_seqs), replace=False)
#             part_pos_test = np.random.choice(test_pos_seqs, size=len(test_pos_seqs), replace=False)
#             pos_seqs = np.concatenate([part_pos_valid, part_pos_test])
#             neg_seqs = np.random.choice(df_hlt['AASeq'].unique(), size=len(pos_seqs), replace=False)
#             # get model outputs for pos and neg sequences:
#             model.eval()
#             with torch.no_grad():
#                 pos_embeds = model.get_embeddings(pos_seqs)
#                 neg_embeds = model.get_embeddings(neg_seqs)
#             # display maps of the embeddings in a single figure (pca, t-sne and u-map):
#             # combine embeddings and labels
#             all_embeds = torch.cat([pos_embeds, neg_embeds], dim=0).cpu().numpy()
#             labels = np.array([1] * len(pos_embeds) + [0] * len(neg_embeds))
#             # Create the figure with subplots
#             fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=450)
#             # PCA
#             pca = PCA(n_components=2)
#             pca_result = pca.fit_transform(all_embeds)
#             axes[0].scatter(pca_result[labels == 0, 0], pca_result[labels == 0, 1], c='orange', label='Healthy', alpha=0.3)
#             axes[0].scatter(pca_result[labels == 1, 0], pca_result[labels == 1, 1], c='purple', label='Disease', alpha=0.3)
#             axes[0].set_title("PCA of Embeddings")
#             axes[0].set_xlabel("Dim 1")
#             axes[0].set_ylabel("Dim 2")
#             axes[0].legend()
#             # t-SNE
#             tsne = TSNE(n_components=2, random_state=42, perplexity=30)
#             tsne_result = tsne.fit_transform(all_embeds)
#             axes[1].scatter(tsne_result[labels == 0, 0], tsne_result[labels == 0, 1], c='orange', label='Healthy', alpha=0.3)
#             axes[1].scatter(tsne_result[labels == 1, 0], tsne_result[labels == 1, 1], c='purple', label='Disease', alpha=0.3)
#             axes[1].set_title("t-SNE of Embeddings")
#             axes[1].set_xlabel("Dim 1")
#             axes[1].set_ylabel("Dim 2")
#             axes[1].legend()
#             # UMAP
#             umap_model = umap.UMAP(n_components=2, random_state=42)
#             umap_result = umap_model.fit_transform(all_embeds)
#             axes[2].scatter(umap_result[labels == 0, 0], umap_result[labels == 0, 1], c='orange', label='Healthy', alpha=0.3)
#             axes[2].scatter(umap_result[labels == 1, 0], umap_result[labels == 1, 1], c='purple', label='Disease', alpha=0.3)
#             axes[2].set_title("UMAP of Embeddings")
#             axes[2].set_xlabel("Dim 1")
#             axes[2].set_ylabel("Dim 2")
#             axes[2].legend()
#             plt.suptitle(f"{model_name} - Embeddings Visualization", fontsize=16)
#             plt.tight_layout()
#             plt.show()
#
#     # Finding uncertainty threshold and such:
#     # Process patient test subjects
#     patient_lls = []
#     healthy_lls = []
#     test_true_labels = []
#     for i, patient_probs in enumerate(disease_validation_probs):
#         patient_probs_reshaped = patient_probs.flatten().reshape(-1, 1)
#
#         # Calculate average log-likelihood for this patient across all their probability values
#         patient_ll = np.mean(final_patient_gmm.score_samples(patient_probs_reshaped))
#         healthy_ll = np.mean(final_healthy_gmm.score_samples(patient_probs_reshaped))
#         patient_lls.append(patient_ll)
#         healthy_lls.append(healthy_ll)
#
#         test_true_labels.append(1)
#     for i, healthy_test_prob in enumerate(healthy_test_probs[:len(disease_validation_probs)]):
#         healthy_test_prob_reshaped = healthy_test_prob.flatten().reshape(-1, 1)
#
#         # Calculate average log-likelihood for this healthy subject across all their probability values
#         patient_ll = np.mean(final_patient_gmm.score_samples(healthy_test_prob_reshaped))
#         healthy_ll = np.mean(final_healthy_gmm.score_samples(healthy_test_prob_reshaped))
#         patient_lls.append(patient_ll)
#         healthy_lls.append(healthy_ll)
#
#         test_true_labels.append(0)  # True label is healthy
#     find_and_display_uncertainty_threshold(patient_lls, healthy_lls, test_true_labels, chosen_components, k_fold_disease, model_config_str, to_savefig=to_savefig)
#
#     return
