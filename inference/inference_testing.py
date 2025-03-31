import torch
import numpy as np
from scipy import stats
from scipy.special import kl_div
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity


def calculate_probas(df, trained_model, patient_ids):
    probas = list()
    for patient_id in patient_ids:
        patient_seqs = df.loc[df["patient_id"] == patient_id, "AASeq"].values
        patient_seqs = np.unique(patient_seqs)

        # Apply the model
        print(f"Patient ID: {patient_id}, Number of Sequences: {len(patient_seqs)}")
        trained_model.eval()
        with torch.no_grad():
            pred = trained_model(patient_seqs)

        # Apply softmax
        pred = torch.softmax(pred, dim=1)

        # Get the probabilities
        proba = pred[:, 1].cpu().numpy()
        probas.append(proba)
    return probas

def dina_inference_suggestion(df_bld, df_hlt, trained_model, valid_patient_ids):
    probas_disease = calculate_probas(df_bld, trained_model, valid_patient_ids)

    healthy_patient_ids = df_hlt["patient_id"].unique()
    healthy_patient_ids = np.random.permutation(healthy_patient_ids)
    probas_healthy = list()
    for patient_id in healthy_patient_ids[:5]:
        patient_seqs = df_hlt.loc[df_hlt["patient_id"] == patient_id, "AASeq"].values
        patient_seqs = np.unique(patient_seqs)

        # Apply the model
        print(f"Patient ID: {patient_id}, Number of Sequences: {len(patient_seqs)}")
        trained_model.eval()
        with torch.no_grad():
            pred = trained_model(patient_seqs)

        # Apply softmax
        pred = torch.softmax(pred, dim=1)

        # Get the probabilities
        proba = pred[:, 1].cpu().numpy()
        probas_healthy.append(proba)

    # print num of samples with prob > 0.5
    print("Disease:")
    for proba in probas_disease:
        print(f"Number of samples with prob > 0.5: {sum(proba > 0.5)}")

    print("Healthy:")
    for proba in probas_healthy:
        print(f"Number of samples with prob > 0.5: {sum(proba > 0.5)}")


def create_average_distribution(probas_list, bins=50, range_min=0, range_max=1, bandwidth=0.05):
    """
    Create an average distribution from a list of probability arrays.

    Args:
        probas_list: List of probability arrays
        bins: Number of bins for histogram approximation
        range_min, range_max: Range for the distribution
        bandwidth: Bandwidth for kernel density estimation

    Returns:
        bin_centers: Centers of bins for the histogram
        average_density: Average distribution values
        kde_model: Fitted KDE model for the combined data
    """
    # Combine all probabilities into one array
    all_probas = np.concatenate(probas_list)

    # Create a histogram representation
    hist_values, bin_edges = np.histogram(
        all_probas,
        bins=bins,
        range=(range_min, range_max),
        density=True
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Create individual histograms for each array
    all_hists = []
    for proba in probas_list:
        hist, _ = np.histogram(
            proba,
            bins=bins,
            range=(range_min, range_max),
            density=True
        )
        all_hists.append(hist)

    # Calculate the average distribution
    all_hists = np.array(all_hists)
    average_density = np.mean(all_hists, axis=0)

    # Fit a KDE model to all data for smoother representation
    kde_model = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde_model.fit(all_probas.reshape(-1, 1))

    return bin_centers, average_density, kde_model


def compute_distribution_similarity(new_proba, reference_kde=None, reference_hist=None,
                                    bin_centers=None, bins=50, range_min=0, range_max=1):
    """
    Compare a new probability array with the reference distribution using various metrics.

    Args:
        new_proba: New probability array to compare
        reference_kde: KDE model of reference distribution
        reference_hist: Histogram representation of reference distribution
        bin_centers: Centers of bins for the histogram
        bins: Number of bins for histogram approximation
        range_min, range_max: Range for the distribution

    Returns:
        Dictionary containing various similarity metrics
    """
    results = {}

    # Create histogram for new data
    new_hist, _ = np.histogram(
        new_proba,
        bins=bins,
        range=(range_min, range_max),
        density=True
    )

    # KL divergence (need to avoid zeros)
    epsilon = 1e-10
    p = reference_hist + epsilon
    q = new_hist + epsilon
    p_normalized = p / np.sum(p)
    q_normalized = q / np.sum(q)

    # KL divergence
    kl = np.sum(kl_div(p_normalized, q_normalized))
    results['kl_divergence'] = kl

    # JS divergence (symmetric version of KL)
    m = 0.5 * (p_normalized + q_normalized)
    js = 0.5 * np.sum(kl_div(p_normalized, m)) + 0.5 * np.sum(kl_div(q_normalized, m))
    results['js_divergence'] = js

    # Earth Mover's Distance (Wasserstein distance)
    if bin_centers is not None:
        from scipy.stats import wasserstein_distance
        emd = wasserstein_distance(bin_centers, bin_centers, p_normalized, q_normalized)
        results['earth_movers_distance'] = emd

    # Log-likelihood of new data under the reference KDE
    if reference_kde is not None:
        log_likelihood = reference_kde.score_samples(new_proba.reshape(-1, 1)).mean()
        results['log_likelihood'] = log_likelihood

    return results


def plot_distributions(bin_centers, average_density, new_proba=None, bins=50, range_min=0, range_max=1):
    """
    Plot the average distribution and optionally a new probability array for comparison.

    Args:
        bin_centers: Centers of bins for the histogram
        average_density: Average distribution values
        new_proba: Optional new probability array to compare
        bins: Number of bins for histogram approximation
        range_min, range_max: Range for the distribution
    """
    plt.figure(figsize=(12, 6))

    # Plot average distribution
    plt.plot(bin_centers, average_density, label='Average Distribution', linewidth=2)

    # Plot new distribution if provided
    if new_proba is not None:
        new_hist, new_bin_edges = np.histogram(
            new_proba,
            bins=bins,
            range=(range_min, range_max),
            density=True
        )
        new_bin_centers = (new_bin_edges[:-1] + new_bin_edges[1:]) / 2
        plt.plot(new_bin_centers, new_hist, label='New Distribution', linewidth=2, alpha=0.7)

    plt.xlabel('Probability Values')
    plt.ylabel('Density')
    plt.title('Probability Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    return plt.gcf()


def summarize_distributions(probas_lists, names):
    """
    Summarize the characteristics of multiple probability lists
    """
    print("\nDistribution Summaries:")
    print("-" * 60)
    print(f"{'Distribution':<20} {'Count':<10} {'Mean Size':<15} {'Min Size':<10} {'Max Size':<10}")
    print("-" * 60)

    for probas, name in zip(probas_lists, names):
        sizes = [len(p) for p in probas]
        avg_size = np.mean(sizes)
        min_size = np.min(sizes)
        max_size = np.max(sizes)

        print(f"{name:<20} {len(probas):<10} {avg_size:<15.1f} {min_size:<10} {max_size:<10}")

    print("\nValue Statistics:")
    print("-" * 70)
    print(
        f"{'Distribution':<20} {'Mean':<10} {'Std':<10} {'Min':<10} {'25%':<10} {'Median':<10} {'75%':<10} {'Max':<10}")
    print("-" * 70)

    for probas, name in zip(probas_lists, names):
        all_values = np.concatenate(probas)
        mean = np.mean(all_values)
        std = np.std(all_values)
        min_val = np.min(all_values)
        q1 = np.percentile(all_values, 25)
        median = np.median(all_values)
        q3 = np.percentile(all_values, 75)
        max_val = np.max(all_values)

        print(
            f"{name:<20} {mean:<10.3f} {std:<10.3f} {min_val:<10.3f} {q1:<10.3f} {median:<10.3f} {q3:<10.3f} {max_val:<10.3f}")


def background_dist_inference(df_bld, df_hlt, trained_model, valid_patient_ids, test_patient_ids):
    # Get healthy patient distributions
    healthy_patient_ids = df_hlt["patient_id"].unique()
    healthy_patient_ids = np.random.permutation(healthy_patient_ids)
    probas_healthy = list()
    for patient_id in healthy_patient_ids[:40]:
        patient_seqs = df_hlt.loc[df_hlt["patient_id"] == patient_id, "AASeq"].values
        patient_seqs = np.unique(patient_seqs)

        # Apply the model
        trained_model.eval()
        with torch.no_grad():
            pred = trained_model(patient_seqs)

        # Apply softmax
        pred = torch.softmax(pred, dim=1)

        # Get the probabilities
        proba = pred[:, 1].cpu().numpy()
        probas_healthy.append(proba)

    # Split into reference and comparison groups
    probas_healthy_ref, probas_healthy_other = probas_healthy[:30], probas_healthy[30:]

    # Calculate probabilities for validation and test sets
    probas_valid = calculate_probas(df_bld, trained_model, valid_patient_ids)
    probas_test = calculate_probas(df_bld, trained_model, test_patient_ids)

    # Create the average distributions
    print("Creating reference distributions...")
    healthy_bin_centers, healthy_density, healthy_kde = create_average_distribution(
        probas_healthy_ref,
        bins=50,
        range_min=0,
        range_max=1,
        bandwidth=0.05
    )

    valid_bin_centers, valid_density, valid_kde = create_average_distribution(
        probas_valid,
        bins=50,
        range_min=0,
        range_max=1,
        bandwidth=0.05
    )

    # Setup the comparison datasets
    reference_distributions = {
        "Healthy Reference": (healthy_bin_centers, healthy_density, healthy_kde),
        "Validation Reference": (valid_bin_centers, valid_density, valid_kde)
    }

    comparison_distributions = {
        "Healthy Other": probas_healthy_other,
        "Test": probas_test
    }

    # Compare all combinations and store results
    results = {}
    print("\nComputing distribution similarities...")
    for ref_name, ref_data in reference_distributions.items():
        bin_centers, avg_density, kde_model = ref_data

        for comp_name, comp_data in comparison_distributions.items():
            key = f"{ref_name} vs {comp_name}"
            results[key] = []

            # Calculate metrics for each individual array in the comparison set
            for i, proba_array in enumerate(comp_data):
                similarity = compute_distribution_similarity(
                    proba_array,
                    reference_kde=kde_model,
                    reference_hist=avg_density,
                    bin_centers=bin_centers
                )
                results[key].append(similarity)

            # Calculate average metrics across all arrays
            avg_metrics = {}
            for metric in results[key][0].keys():
                values = [result[metric] for result in results[key]]
                avg_metrics[metric] = np.mean(values)
                avg_metrics[f"{metric}_std"] = np.std(values)

            print(f"\n{key} (Average Metrics):")
            for metric, value in avg_metrics.items():
                if not metric.endswith('_std'):
                    std = avg_metrics.get(f"{metric}_std", 0)
                    print(f"  {metric}: {value:.6f} ± {std:.6f}")

    # Visualize the distributions
    print("\nGenerating distribution plots...")
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Distribution Comparisons', fontsize=16)

    # Plot 1: All reference distributions
    ax = axs[0, 0]
    ax.plot(healthy_bin_centers, healthy_density, label='Healthy Reference', linewidth=2)
    ax.plot(valid_bin_centers, valid_density, label='Validation Reference', linewidth=2)
    ax.set_xlabel('Probability Values')
    ax.set_ylabel('Density')
    ax.set_title('Reference Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Healthy Reference vs Others
    ax = axs[0, 1]
    ax.plot(healthy_bin_centers, healthy_density, label='Healthy Reference', linewidth=2)

    # Aggregate and plot comparison distributions
    for comp_name, comp_data in comparison_distributions.items():
        all_comp_data = np.concatenate(comp_data)
        comp_hist, comp_bin_edges = np.histogram(
            all_comp_data,
            bins=50,
            range=(0, 1),
            density=True
        )
        comp_bin_centers = (comp_bin_edges[:-1] + comp_bin_edges[1:]) / 2
        ax.plot(comp_bin_centers, comp_hist, label=comp_name, linewidth=2, alpha=0.7)

    ax.set_xlabel('Probability Values')
    ax.set_ylabel('Density')
    ax.set_title('Healthy Reference vs. Other Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Validation Reference vs Others
    ax = axs[1, 0]
    ax.plot(valid_bin_centers, valid_density, label='Validation Reference', linewidth=2)

    # Plot comparison distributions
    for comp_name, comp_data in comparison_distributions.items():
        all_comp_data = np.concatenate(comp_data)
        comp_hist, comp_bin_edges = np.histogram(
            all_comp_data,
            bins=50,
            range=(0, 1),
            density=True
        )
        comp_bin_centers = (comp_bin_edges[:-1] + comp_bin_edges[1:]) / 2
        ax.plot(comp_bin_centers, comp_hist, label=comp_name, linewidth=2, alpha=0.7)

    ax.set_xlabel('Probability Values')
    ax.set_ylabel('Density')
    ax.set_title('Validation Reference vs. Other Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: KL Divergence Comparison
    ax = axs[1, 1]

    # Extract KL divergence values for plotting
    comparisons = list(results.keys())
    kl_values = [np.mean([r['kl_divergence'] for r in results[k]]) for k in comparisons]
    kl_stds = [np.std([r['kl_divergence'] for r in results[k]]) for k in comparisons]

    # Create bar chart
    bars = ax.bar(comparisons, kl_values, yerr=kl_stds, alpha=0.7)
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL Divergence Comparison')
    ax.set_xticklabels(comparisons, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('distribution_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    summarize_distributions([probas_healthy_ref, probas_valid, probas_healthy_other, probas_test],
                            ["Healthy Reference", "Validation Reference", "Healthy Other", "Test"])

    return results, (healthy_bin_centers, healthy_density, healthy_kde), (valid_bin_centers, valid_density, valid_kde)
