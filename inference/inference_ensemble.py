import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrices(trained_model, valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs, model_string, device):
    # Combine positive and negative sequences
    seqs_to_use = np.concatenate([valid_pos_seqs, test_pos_seqs, valid_neg_seqs, test_neg_seqs])

    # Create labels (1 for positives, 0 for negatives)
    labels_to_use = np.concatenate([
        np.ones(len(valid_pos_seqs) + len(test_pos_seqs)),  # positives
        np.zeros(len(valid_neg_seqs) + len(test_neg_seqs))  # negatives
    ])

    # Group sequences and their corresponding labels by length
    grouped_seqs = defaultdict(list)
    grouped_labels = defaultdict(list)

    for seq, label in zip(seqs_to_use, labels_to_use):
        length = len(seq)
        grouped_seqs[length].append(seq)
        grouped_labels[length].append(label)

    # Store results for plotting
    results = []
    confusion_matrices = []

    # Process each length group
    for length in sorted(grouped_seqs.keys()):
        seqs = grouped_seqs[length]
        labels = grouped_labels[length]

        if len(seqs) <= 4:
            continue

        # Count positives and negatives
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count

        # Get model outputs
        with torch.no_grad():
            disease_logits = trained_model(seqs)

        # Convert to probabilities and predictions
        disease_probs = torch.softmax(disease_logits, dim=1)[:, 1].cpu().numpy()
        y_pred = (disease_probs >= 0.5).astype(int)
        y_true = np.array(labels, dtype=int)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

        # Extract confusion matrix values
        TP = cm[0, 0]  # True Positives
        FN = cm[0, 1]  # False Negatives
        FP = cm[1, 0]  # False Positives
        TN = cm[1, 1]  # True Negatives

        # Calculate rates
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Sensitivity/Recall
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0  # False Positive Rate
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0  # False Negative Rate
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # Specificity
        ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0  # Accuracy

        # Store results
        results.append({
            'length': length,
            'total_seqs': len(seqs),
            'pos_count': int(pos_count),
            'neg_count': int(neg_count),
            'TPR': TPR,
            'FPR': FPR,
            'FNR': FNR,
            'TNR': TNR,
            'ACC': ACC,
            'TP': TP,
            'TN': TN,
            'FP': FP,
            'FN': FN
        })

        # Store confusion matrix for heatmap (normalized rates)
        cm_rates = np.array([[TPR, FNR], [FPR, TNR]])
        confusion_matrices.append((length, cm_rates, int(pos_count), int(neg_count), len(seqs)))

    if not results:
        print("No sequence groups with more than 4 sequences found.")
        return

    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))

    # 1. Performance metrics over sequence lengths
    ax1 = plt.subplot(2, 3, 1)
    lengths = [r['length'] for r in results]
    accuracies = [r['ACC'] for r in results]
    tprs = [r['TPR'] for r in results]
    tnrs = [r['TNR'] for r in results]

    plt.plot(lengths, accuracies, 'o-', label='Accuracy', linewidth=2, markersize=6)
    plt.plot(lengths, tprs, 's-', label='TPR (Sensitivity)', linewidth=2, markersize=6)
    plt.plot(lengths, tnrs, '^-', label='TNR (Specificity)', linewidth=2, markersize=6)
    plt.xlabel('Sequence Length')
    plt.ylabel('Rate')
    plt.title('Performance Metrics by Sequence Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.05)

    # 2. False rates
    ax2 = plt.subplot(2, 3, 2)
    fprs = [r['FPR'] for r in results]
    fnrs = [r['FNR'] for r in results]

    plt.plot(lengths, fprs, 'o-', label='FPR (False Positive Rate)', color='red', linewidth=2, markersize=6)
    plt.plot(lengths, fnrs, 's-', label='FNR (False Negative Rate)', color='orange', linewidth=2, markersize=6)
    plt.xlabel('Sequence Length')
    plt.ylabel('Rate')
    plt.title('Error Rates by Sequence Length')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(max(fprs), max(fnrs)) * 1.1 if max(max(fprs), max(fnrs)) > 0 else 1)

    # 3. Sample distribution
    ax3 = plt.subplot(2, 3, 3)
    pos_counts = [r['pos_count'] for r in results]
    neg_counts = [r['neg_count'] for r in results]

    x = np.arange(len(lengths))
    width = 0.35

    plt.bar(x - width / 2, pos_counts, width, label='Positive Samples', color='lightblue', alpha=0.8)
    plt.bar(x + width / 2, neg_counts, width, label='Negative Samples', color='lightcoral', alpha=0.8)
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.title('Sample Distribution by Length')
    plt.xticks(x, [str(l) for l in lengths])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # 4. Individual confusion matrix heatmaps (show first few)
    num_heatmaps = min(3, len(confusion_matrices))
    for i in range(num_heatmaps):
        length, cm_rates, pos_count, neg_count, total_seqs = confusion_matrices[i]

        ax = plt.subplot(2, 3, 4 + i)

        # Create heatmap
        sns.heatmap(cm_rates,
                    annot=True,
                    fmt='.3f',
                    cmap='Blues',
                    xticklabels=['Positive', 'Negative'],
                    yticklabels=['Positive', 'Negative'],
                    cbar_kws={'label': 'Rate'},
                    square=True)

        plt.title(f'Confusion Matrix (Rates)\nLength {length} ({total_seqs} seqs: {pos_count}+, {neg_count}-)')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')

    plt.tight_layout()
    plt.savefig(f"plots/inference_plots/inference_graph_group_sequence_lengths_{model_string}.png", dpi=600)
    plt.show()

    # Create a summary table plot
    fig2, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')

    # Prepare table data
    table_data = []
    headers = ['Length', 'Total', 'Pos', 'Neg', 'Accuracy', 'TPR', 'TNR', 'FPR', 'FNR']

    for r in results:
        row = [
            r['length'],
            r['total_seqs'],
            r['pos_count'],
            r['neg_count'],
            f"{r['ACC']:.3f}",
            f"{r['TPR']:.3f}",
            f"{r['TNR']:.3f}",
            f"{r['FPR']:.3f}",
            f"{r['FNR']:.3f}"
        ]
        table_data.append(row)

    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=headers,
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Color code the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Color code accuracy column
    acc_col = 4
    for i in range(1, len(table_data) + 1):
        acc_val = float(table_data[i - 1][acc_col])
        if acc_val >= 0.9:
            color = '#90EE90'  # Light green
        elif acc_val >= 0.8:
            color = '#FFFFE0'  # Light yellow
        else:
            color = '#FFB6C1'  # Light pink
        table[(i, acc_col)].set_facecolor(color)

    plt.title('Performance Summary Table - Grouped by Sequence Length', fontsize=16, fontweight='bold', pad=20)
    plt.savefig(f"plots/inference_plots/inference_summary_group_sequence_lengths_{model_string}.png", dpi=600)
    plt.show()

    # Print summary statistics
    print(f"\nSummary Statistics:")
    print(f"{'=' * 50}")
    print(f"Processed {len(results)} sequence length groups")
    print(f"Overall accuracy range: {min(accuracies):.3f} - {max(accuracies):.3f}")
    print(f"Mean accuracy: {np.mean(accuracies):.3f} ± {np.std(accuracies):.3f}")
    print(f"Best performing length: {lengths[np.argmax(accuracies)]} (ACC: {max(accuracies):.3f})")
    print(f"Worst performing length: {lengths[np.argmin(accuracies)]} (ACC: {min(accuracies):.3f})")
    return
    # Combine positive and negative sequences
    seqs_to_use = np.concatenate([valid_pos_seqs, test_pos_seqs, valid_neg_seqs, test_neg_seqs])

    # Create labels (1 for positives, 0 for negatives)
    labels_to_use = np.concatenate([
        np.ones(len(valid_pos_seqs) + len(test_pos_seqs)),  # positives
        np.zeros(len(valid_neg_seqs) + len(test_neg_seqs))  # negatives
    ])

    # Group sequences and their corresponding labels by length
    grouped_seqs = defaultdict(list)
    grouped_labels = defaultdict(list)

    for seq, label in zip(seqs_to_use, labels_to_use):
        length = len(seq)
        grouped_seqs[length].append(seq)
        grouped_labels[length].append(label)

    # Process each length group
    for length in sorted(grouped_seqs.keys()):
        seqs = grouped_seqs[length]
        labels = grouped_labels[length]

        if len(seqs) <= 4:
            continue

        # Count positives and negatives
        pos_count = sum(labels)
        neg_count = len(labels) - pos_count

        print(f"\n{'=' * 60}")
        print(f"Length {length} ({len(seqs)} sequences: {int(pos_count)} pos, {int(neg_count)} neg)")
        print(f"{'=' * 60}")

        # Get model outputs
        with torch.no_grad():
            disease_logits = trained_model(seqs)

        # Convert to probabilities and predictions
        disease_probs = torch.softmax(disease_logits, dim=1)[:, 1].cpu().numpy()
        y_pred = (disease_probs >= 0.5).astype(int)
        y_true = np.array(labels, dtype=int)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

        # Extract confusion matrix values
        # cm structure: [[TP, FN], [FP, TN]]
        TP = cm[0, 0]  # True Positives
        FN = cm[0, 1]  # False Negatives
        FP = cm[1, 0]  # False Positives
        TN = cm[1, 1]  # True Negatives

        # Calculate rates
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # Sensitivity/Recall
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0.0  # False Positive Rate
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0.0  # False Negative Rate
        TNR = TN / (TN + FP) if (TN + FP) > 0 else 0.0  # Specificity

        # Additional useful metrics
        ACC = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0  # Accuracy
        # PPV = TP / (TP + FP) if (TP + FP) > 0 else 0.0  # Precision/Positive Predictive Value
        # NPV = TN / (TN + FN) if (TN + FN) > 0 else 0.0  # Negative Predictive Value

        # Pretty print confusion matrix
        print(f"\nConfusion Matrix in Rates - Accuracy: {ACC:.4f}")
        print(f"┌─────────────┬──────────┬──────────┐")
        print(f"│   Actual \\  │ Positive │ Negative │")
        print(f"│ Predicted   │          │          │")
        print(f"├─────────────┼──────────┼──────────┤")
        print(f"│ Positive    │  {TPR:.3f}   │   {FPR:.3f}  │")
        print(f"│ Negative    │  {FNR:.3f}   │   {TNR:.3f}  │")
        print(f"└─────────────┴──────────┴──────────┘")
