import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from models.cvc_cacheing_model import CVCCachingModel


def inference_classification_model(trained_model, args, df_bld, df_hlt, test_patient_inds, valid_patient_inds, unique_patient_ids,
                                   valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs, device,
                                   vector_representation_bins=10, num_of_healthy_patients=40, num_of_healthy_test_patients=5):
    # Creating a caching model of the trained model
    trained_model.eval()
    caching_model = CVCCachingModel(trained_model, args, device)
    caching_model.to(device)
    caching_model.eval()

    # patient vectors helping data
    possible_seqs = set(np.concatenate([valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs]))
    patient_valid_test_inds = np.concatenate([test_patient_inds, valid_patient_inds])

    # get the patient vectors
    patient_vectors = []
    for patient_ind in patient_valid_test_inds:
        # Extract and process the patient sequences
        patient_seqs = df_bld.loc[df_bld["patient_id"] == unique_patient_ids[patient_ind], "AASeq"].values
        patient_seqs = np.array([x for x in np.unique(patient_seqs) if x in possible_seqs])

        # Get model outputs
        with torch.no_grad():
            disease_logits = caching_model(patient_seqs)

        # Convert to probabilities
        disease_probs = torch.softmax(disease_logits, dim=1)[:, 1].cpu().numpy()

        # bin using np into x bins (min value is 0 and max is 1)
        disease_probs = np.digitize(disease_probs, bins=np.linspace(0, 1, vector_representation_bins + 1)) - 1
        # Vectorized bin counting
        patient_vector = np.bincount(disease_probs, minlength=10).astype(np.float32)
        patient_vector /= patient_vector.sum()
        patient_vectors.append(patient_vector)
    patient_vectors = np.array(patient_vectors)

    # healthy vectors helping data
    healthy_patients = df_hlt["patient_id"].unique()
    np.random.shuffle(healthy_patients)
    healthy_patients = healthy_patients[:num_of_healthy_patients]

    # get the healthy patient vectors
    healthy_vectors = []
    for patient_ind in healthy_patients:
        # Extract and process the patient sequences
        patient_seqs = df_hlt.loc[df_hlt["patient_id"] == patient_ind, "AASeq"].values

        # Get model outputs
        with torch.no_grad():
            healthy_logits = caching_model(patient_seqs)

        # Convert to probabilities
        healthy_probs = torch.softmax(healthy_logits, dim=1)[:, 1].cpu().numpy()

        # bin using np into x bins (min value is 0 and max is 1)
        healthy_probs = np.digitize(healthy_probs, bins=np.linspace(0, 1, vector_representation_bins + 1)) - 1
        # Vectorized bin counting
        healthy_vector = np.bincount(healthy_probs, minlength=vector_representation_bins).astype(np.float32)
        healthy_vector /= healthy_vector.sum()
        healthy_vectors.append(healthy_vector)
    healthy_vectors = np.array(healthy_vectors)
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
        knn = KNeighborsClassifier(n_neighbors=3)  # You might want to tune this parameter
        knn.fit(X_train, y_train)
        y_pred_knn = knn.predict(X_test)
        cm_knn = confusion_matrix(y_test, y_pred_knn)

        # 2. Balanced Random Forest Classifier
        # Using class_weight='balanced' to handle class imbalance
        rf = RandomForestClassifier(n_estimators=10, class_weight='balanced', random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        cm_rf = confusion_matrix(y_test, y_pred_rf)

        # Append the confusion matrices to the total list
        total_cm_knn.append(cm_knn)
        total_cm_rf.append(cm_rf)

    # Display final results
    def calculate_average_cm(cm_list):
        """Calculate the average confusion matrix from a list of confusion matrices"""
        # Convert list to numpy array for easier calculation
        cm_array = np.array(cm_list)

        # Calculate the mean across the first dimension (across all matrices)
        avg_cm = np.mean(cm_array, axis=0)

        # For display purposes, you might want to round to integers or decimals
        # Use this for integer display (total counts)
        avg_cm_int = np.round(avg_cm).astype(int)

        # Or use this for decimal display (average counts, might be more appropriate)
        avg_cm_float = np.round(avg_cm, 2)

        return avg_cm_float

    def display_average_results(total_cm_knn, total_cm_rf):
        """Display the average results from multiple k-fold runs"""
        # Calculate average confusion matrices
        avg_cm_knn = calculate_average_cm(total_cm_knn)
        avg_cm_rf = calculate_average_cm(total_cm_rf)

        # Number of folds
        n_folds = len(total_cm_knn)

        # Print results
        print(f"Average results across {n_folds} folds:")

        print("\nKNN - Average Confusion Matrix:")
        print(avg_cm_knn)
        print(f"Average Disease correctly classified: {avg_cm_knn[1, 1]:.2f}")
        print(f"Average Healthy correctly classified: {avg_cm_knn[0, 0]:.2f}")

        print("\nRandom Forest - Average Confusion Matrix:")
        print(avg_cm_rf)
        print(f"Average Disease correctly classified: {avg_cm_rf[1, 1]:.2f}")
        print(f"Average Healthy correctly classified: {avg_cm_rf[0, 0]:.2f}")

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
        plt.show()

        # Calculate and display additional metrics
        print("\nAdditional average metrics:")

        # KNN metrics
        knn_sensitivity = avg_cm_knn[1, 1] / (avg_cm_knn[1, 0] + avg_cm_knn[1, 1])  # TPR
        knn_specificity = avg_cm_knn[0, 0] / (avg_cm_knn[0, 0] + avg_cm_knn[0, 1])  # TNR

        # RF metrics
        rf_sensitivity = avg_cm_rf[1, 1] / (avg_cm_rf[1, 0] + avg_cm_rf[1, 1])  # TPR
        rf_specificity = avg_cm_rf[0, 0] / (avg_cm_rf[0, 0] + avg_cm_rf[0, 1])  # TNR

        print(f"KNN - Sensitivity (TPR): {knn_sensitivity:.4f}, Specificity (TNR): {knn_specificity:.4f}")
        print(f"RF  - Sensitivity (TPR): {rf_sensitivity:.4f}, Specificity (TNR): {rf_specificity:.4f}")

    # Execute the function to display average results
    display_average_results(total_cm_knn, total_cm_rf)

    # You could also calculate standard deviations to show variability across folds
    def calculate_std_cm(cm_list):
        """Calculate standard deviation of confusion matrices across folds"""
        cm_array = np.array(cm_list)
        std_cm = np.std(cm_array, axis=0)
        return np.round(std_cm, 2)

    std_cm_knn = calculate_std_cm(total_cm_knn)
    std_cm_rf = calculate_std_cm(total_cm_rf)

    print("\nStandard deviations across folds:")
    print("\nKNN - Standard Deviation of Confusion Matrix:")
    print(std_cm_knn)

    print("\nRandom Forest - Standard Deviation of Confusion Matrix:")
    print(std_cm_rf)
