import os
import torch
import numpy as np
from scipy import stats
from scipy.special import kl_div
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.preprocessing import KBinsDiscretizer
from cache_handler import get_model_config_str
import pandas as pd
from tqdm import tqdm
from utils import pairwise_scores, levenshtein_dist_non_bin
import seaborn as sns


def calculate_probas(df, trained_model, patient_ids, to_print=True):
    probas = list()
    for patient_id in patient_ids:
        patient_seqs = df.loc[df["patient_id"] == patient_id, "AASeq"].values
        patient_seqs = np.unique(patient_seqs)

        # Apply the model
        if to_print:
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
    for patient_id in healthy_patient_ids[:50]:
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
    probas_healthy_ref, probas_healthy_other = probas_healthy[:40], probas_healthy[40:]

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


def flatten_probas(list_of_arrays):
    return np.concatenate(list_of_arrays, axis=0)


def discretize_probas(probas, n_bins=50):
    # Discretize for JS/KL if needed
    hist, bin_edges = np.histogram(probas, bins=n_bins, range=(0, 1), density=True)
    return hist + 1e-8  # add epsilon to avoid zero divisions


def my_plot_distributions(dists_dict, args, n_bins=50):
    plt.figure(figsize=(6, 6), dpi=600)

    last_hist = None
    for label, data in dists_dict.items():
        last_hist = plt.hist(data, bins=n_bins, range=(0, 1), alpha=0.3, density=True, label=label, histtype='stepfilled')

    plt.title("Probability Distribution Comparison")
    plt.xlabel("Probability (model output)")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(0, last_hist[0][1] + 1)
    plt.savefig(f"plots/inference_plots/inference_histogram_{get_model_config_str(args)}.png", dpi=600, bbox_inches='tight')
    plt.show()


def generate_conclusion(results):
    from collections import defaultdict

    grouped = defaultdict(list)
    for name, value in results.items():
        metric = name.split()[0]  # JSD, KS, Wasserstein, etc.
        grouped[metric].append((name, value))

    conclusion = {}
    for metric, entries in grouped.items():
        # Filter comparisons
        healthy_entries = [e for e in entries if "Healthy (Base)" in e[0]]
        disease_entries = [e for e in entries if "Disease (Base)" in e[0]]

        if healthy_entries:
            best_h = min(healthy_entries, key=lambda x: x[1])
            conclusion[f"{metric:<12} Healthy (Base)"] = f"{best_h[0]:<47} (score: {best_h[1]:.4f})"

        if disease_entries:
            best_d = min(disease_entries, key=lambda x: x[1])
            conclusion[f"{metric:<12} Disease (Base)"] = f"{best_d[0]:<47} (score: {best_d[1]:.4f})"

    return conclusion


def print_results_table(results):
    print("\nDistribution Distance Metrics")
    print("-" * 53)
    print(f"{'Metric':<43} {'Value':>8}")
    print("-" * 53)
    for k, v in results.items():
        print(f"{k:<30} {v:>8.4f}")
    print("-" * 53)

    conclusion = generate_conclusion(results)
    print("\nConclusion (Closest Match per Metric Type)")
    print("-" * 65)
    for k, v in conclusion.items():
        print(f"{k:<25} {v}")
    print("-" * 65)


def write_results_table(results, filename):
    with open(filename, "w") as f:
        f.write("\nDistribution Distance Metrics\n")
        f.write("-" * 53 + "\n")
        f.write(f"{'Metric':<43} {'Value':>8}\n")
        f.write("-" * 53 + "\n")
        for k, v in results.items():
            f.write(f"{k:<30} {v:>8.4f}\n")
        f.write("-" * 53 + "\n")

        conclusion = generate_conclusion(results)
        f.write("\nConclusion (Closest Match per Metric Type)\n")
        f.write("-" * 65 + "\n")
        for k, v in conclusion.items():
            f.write(f"{k:<25} {v}\n")
        f.write("-" * 65 + "\n")


def my_dist_inference(df_bld, df_hlt, trained_model, valid_patient_ids, test_patient_ids, args, n_bins=100):
    # Get healthy patient distributions
    healthy_patient_ids = df_hlt["patient_id"].unique()
    healthy_patient_ids = np.random.permutation(healthy_patient_ids)
    probas_healthy = list()

    print("\nCalculating healthy patient distributions...")
    for patient_id in healthy_patient_ids[:60]:
        patient_seqs = df_hlt.loc[df_hlt["patient_id"] == patient_id, "AASeq"].values
        patient_seqs = np.unique(patient_seqs)

        trained_model.eval()
        with torch.no_grad():
            pred = trained_model(patient_seqs)
            pred = torch.softmax(pred, dim=1)
            proba = pred[:, 1].cpu().numpy()
            probas_healthy.append(proba)

    probas_healthy_ref = probas_healthy[:45]
    probas_healthy_other = probas_healthy[45:]

    # Validation and test
    print("Calculating validation and test distributions...")
    probas_valid = calculate_probas(df_bld, trained_model, valid_patient_ids, to_print=False)
    probas_test = calculate_probas(df_bld, trained_model, test_patient_ids, to_print=False)

    # Flatten for comparison
    healthy_base = flatten_probas(probas_healthy_ref)
    healthy_test = flatten_probas(probas_healthy_other)
    disease_base = flatten_probas(probas_valid)
    disease_test = flatten_probas(probas_test)

    # Discretize for JS divergence
    print("Discretizing distributions...")
    healthy_base_bins = discretize_probas(healthy_base, n_bins)
    healthy_test_bins = discretize_probas(healthy_test, n_bins)
    disease_base_bins = discretize_probas(disease_base, n_bins)
    disease_test_bins = discretize_probas(disease_test, n_bins)

    # Calculate distribution distances
    print("Calculating distribution distances...")
    results = {
        "JSD         Disease (Base) vs Disease (Test)": jensenshannon(disease_base_bins, disease_test_bins),
        "JSD         Disease (Base) vs Healthy (Test)": jensenshannon(disease_base_bins, healthy_test_bins),
        "JSD         Healthy (Base) vs Healthy (Test)": jensenshannon(healthy_base_bins, healthy_test_bins),
        "JSD         Healthy (Base) vs Disease (Test)": jensenshannon(healthy_base_bins, disease_test_bins),
        "Wasserstein Disease (Base) vs Disease (Test)": wasserstein_distance(disease_base, disease_test),
        "Wasserstein Disease (Base) vs Healthy (Test)": wasserstein_distance(disease_base, healthy_test),
        "Wasserstein Healthy (Base) vs Healthy (Test)": wasserstein_distance(healthy_base, healthy_test),
        "Wasserstein Healthy (Base) vs Disease (Test)": wasserstein_distance(healthy_base, disease_test),
        "KS          Disease (Base) vs Disease (Test)": ks_2samp(disease_base, disease_test).statistic,
        "KS          Disease (Base) vs Healthy (Test)": ks_2samp(disease_base, healthy_test).statistic,
        "KS          Healthy (Base) vs Healthy (Test)": ks_2samp(healthy_base, healthy_test).statistic,
        "KS          Healthy (Base) vs Disease (Test)": ks_2samp(healthy_base, disease_test).statistic
    }

    # Pretty print and save to file
    print_results_table(results)
    os.makedirs("results/inference_results", exist_ok=True)
    write_results_table(results, f"results/inference_results/inference_results_{get_model_config_str(args)}.txt")

    # Plot the distributions and save the plot
    my_plot_distributions({
        "Healthy (Base)": healthy_base,
        "Healthy (Test)": healthy_test,
        "Disease (Base)": disease_base,
        "Disease (Test)": disease_test
    }, args, n_bins=n_bins)

    return results


def t1d_inference_other_dataset(trained_model, dataset_type):
    if dataset_type.lower() != "t1d":
        print("Dataset type is not T1D. Skipping T1D inference.")
        return

    def get_t1d_tcrs():
        import pandas as pd
        import re
        import io

        def extract_t1d_tcr_beta_chains_from_excel(file_path):
            """
            Parse an Excel file containing TCR sequences and extract T1D-related TCR beta chains.

            Parameters:
            file_path (str): Path to the Excel file containing TCR sequences

            Returns:
            pandas.DataFrame: DataFrame containing T1D-related TCR beta chain information
            """
            try:
                # Check file extension to determine format
                _, file_extension = os.path.splitext(file_path)

                # Print the file extension for debugging
                print(f"Detected file extension: {file_extension}")

                # Attempt to read the file with pandas - it will auto-detect xlsx or xls
                print(f"Attempting to read file: {file_path}")

                if file_extension.lower() in ['.xlsx', '.xls']:
                    # Read Excel file
                    print("Reading as Excel file...")
                    df = pd.read_excel(file_path)
                else:
                    # Try reading as CSV, then TSV if CSV fails
                    try:
                        print("Trying to read as CSV...")
                        df = pd.read_csv(file_path)
                    except Exception as csv_error:
                        print(f"CSV read failed: {csv_error}")
                        try:
                            print("Trying to read as TSV...")
                            df = pd.read_csv(file_path, sep='\t')
                        except Exception as tsv_error:
                            print(f"TSV read failed: {tsv_error}")
                            # Last resort: try to read with binary mode and detect encoding
                            try:
                                import chardet
                                with open(file_path, 'rb') as rawdata:
                                    result = chardet.detect(rawdata.read(100000))
                                print(f"Detected encoding: {result['encoding']} with confidence {result['confidence']}")
                                df = pd.read_csv(file_path, encoding=result['encoding'])
                            except Exception as e:
                                print(f"All reading methods failed: {e}")

                # Print column names to verify we read the file correctly
                print("Columns in the file:")
                print(df.columns.tolist())

                # Clean column names (remove any leading/trailing whitespace)
                df.columns = df.columns.str.strip()

                # Check if 'TCR ID' column exists
                if 'TCR ID' not in df.columns:
                    print("'TCR ID' column not found. Available columns:")
                    print(df.columns.tolist())

                    # Try to find a column that might contain TCR IDs
                    potential_id_columns = [col for col in df.columns if ('ID' in col or 'id' in col)]

                    if potential_id_columns:
                        print(f"Found potential ID columns: {potential_id_columns}")
                        # Check each potential ID column
                        for col in potential_id_columns:
                            # Print sample values to debug
                            print(f"Sample values in '{col}':")
                            print(df[col].head())

                            # Check if any values start with "T1D-"
                            if df[col].astype(str).str.startswith('T1D-').any():
                                print(f"Using '{col}' as TCR ID column")
                                df = df.rename(columns={col: 'TCR ID'})
                                break

                    # If still not found, look for any column with T1D- values
                    if 'TCR ID' not in df.columns:
                        print("Looking for any column with T1D- values...")
                        for col in df.columns:
                            if df[col].astype(str).str.startswith('T1D-').any():
                                print(f"Found T1D values in column '{col}'")
                                df = df.rename(columns={col: 'TCR ID'})
                                break

                # If we still don't have a TCR ID column, return empty DataFrame
                if 'TCR ID' not in df.columns:
                    print("Could not find a column containing TCR IDs starting with 'T1D-'")
                    # Display first few rows to help diagnose
                    print("First few rows of the data:")
                    print(df.head())
                    return pd.DataFrame()

                # Filter for T1D-related TCRs (TCR IDs that start with "T1D-")
                t1d_df = df[df['TCR ID'].astype(str).str.startswith('T1D-')]

                print(f"Found {len(t1d_df)} rows with TCR IDs starting with 'T1D-'")

                # Identify columns containing beta chain information
                beta_chain_cols = ['TCR ID']

                # Look for CDR3b column with various possible names
                cdr3b_col_options = ['CDR3b', 'CDR3β', 'CDR3 beta']
                for col in cdr3b_col_options:
                    if col in df.columns:
                        beta_chain_cols.append(col)
                        break
                else:
                    # If none of the specific names are found, look for any column with 'CDR3' and 'b'
                    cdr3b_cols = [col for col in df.columns if 'CDR3' in col and ('b' in col.lower() or 'β' in col)]
                    if cdr3b_cols:
                        beta_chain_cols.append(cdr3b_cols[0])

                # Look for TRBV column
                trbv_col_options = ['TRBV', 'V beta', 'V β', 'Vbeta']
                for col in trbv_col_options:
                    if col in df.columns:
                        beta_chain_cols.append(col)
                        break
                else:
                    # If none of the specific names are found, look for any column with 'V' and 'b'
                    trbv_cols = [col for col in df.columns if 'V' in col and ('b' in col.lower() or 'β' in col)]
                    if trbv_cols:
                        beta_chain_cols.append(trbv_cols[0])

                # Look for TRBJ column
                trbj_col_options = ['TRBJ', 'J beta', 'J β', 'Jbeta']
                for col in trbj_col_options:
                    if col in df.columns:
                        beta_chain_cols.append(col)
                        break
                else:
                    # If none of the specific names are found, look for any column with 'J' and 'b'
                    trbj_cols = [col for col in df.columns if 'J' in col and ('b' in col.lower() or 'β' in col)]
                    if trbj_cols:
                        beta_chain_cols.append(trbj_cols[0])

                # Add additional useful columns if they exist
                additional_cols = ['Antigen(s), HLA', 'Antigen(s)', 'Antigen', 'HLA',
                                   'CD4 or CD8', 'T cell type', 'Clone name(s)', 'Source(s)']
                for col in additional_cols:
                    if col in df.columns:
                        beta_chain_cols.append(col)

                # Check which columns actually exist in the DataFrame
                available_cols = [col for col in beta_chain_cols if col in df.columns]

                print(f"Using columns: {available_cols}")

                if len(available_cols) <= 1:  # Just the TCR ID column
                    print("No columns with beta chain information found")
                    # Print all column names to help diagnose
                    print("Available columns:")
                    print(df.columns.tolist())
                    return pd.DataFrame()

                # Select only available columns
                t1d_tcrs = t1d_df[available_cols].copy()

                # Drop duplicate entries (if any)
                t1d_tcrs = t1d_tcrs.drop_duplicates().reset_index(drop=True)

                # Fill NaN values with "Not specified" for better readability
                t1d_tcrs = t1d_tcrs.fillna('Not specified')

                return t1d_tcrs

            except Exception as e:
                print(f"Error processing file: {e}")
                import traceback
                traceback.print_exc()
                return pd.DataFrame()


        def save_t1d_tcrs(t1d_tcrs, output_file='t1d_tcr_beta_chains.csv'):
            """
            Save the extracted T1D TCR beta chains to a CSV file.

            Parameters:
            t1d_tcrs (pandas.DataFrame): DataFrame containing T1D-related TCR beta chain information
            output_file (str): Path to save the output CSV file
            """
            try:
                t1d_tcrs.to_csv(output_file, index=False)
                print(f"Successfully saved {len(t1d_tcrs)} T1D-related TCR beta chains to {output_file}")
            except Exception as e:
                print(f"Error saving to file: {e}")


        def analyze_t1d_tcrs(t1d_tcrs):
            """
            Perform basic analysis on the extracted T1D TCR beta chains.

            Parameters:
            t1d_tcrs (pandas.DataFrame): DataFrame containing T1D-related TCR beta chain information
            """
            if t1d_tcrs.empty:
                print("No T1D-related TCR beta chains found.")
                return

            # Find CDR3b column
            cdr3b_col = None
            for col in t1d_tcrs.columns:
                if 'CDR3' in col and ('b' in col.lower() or 'β' in col):
                    cdr3b_col = col
                    break

            # Find TRBV column
            trbv_col = None
            for col in t1d_tcrs.columns:
                if col == 'TRBV' or ('V' in col and ('b' in col.lower() or 'β' in col)):
                    trbv_col = col
                    break

            # Find T cell type column
            t_cell_col = None
            for col in t1d_tcrs.columns:
                if 'CD4 or CD8' in col or 'T cell' in col:
                    t_cell_col = col
                    break

            # Find antigen column
            antigen_col = None
            for col in t1d_tcrs.columns:
                if 'Antigen' in col or 'HLA' in col:
                    antigen_col = col
                    break

            # Count TCRs by type (CD4 vs CD8)
            if t_cell_col:
                t_cell_counts = t1d_tcrs[t_cell_col].value_counts()
                print("\nT-cell type distribution:")
                print(t_cell_counts)

            # Count TCRs by antigen
            if antigen_col:
                antigen_counts = t1d_tcrs[antigen_col].value_counts().head(10)
                print("\nTop 10 antigens:")
                print(antigen_counts)

            # Analyze TRBV usage
            if trbv_col:
                # Clean TRBV values for counting (remove version numbers like *01)
                t1d_tcrs['TRBV_clean'] = t1d_tcrs[trbv_col].apply(
                    lambda x: re.sub(r'\*\d+', '', str(x)) if pd.notna(x) else 'Not specified'
                )
                trbv_counts = t1d_tcrs['TRBV_clean'].value_counts().head(10)
                print("\nTop 10 TRBV gene usage:")
                print(trbv_counts)

            # Basic CDR3 length analysis
            if cdr3b_col:
                t1d_tcrs['CDR3b_length'] = t1d_tcrs[cdr3b_col].apply(
                    lambda x: len(str(x)) if pd.notna(x) and str(x) != 'Not specified' else 0
                )
                # Filter out zero lengths for average calculation
                lengths = t1d_tcrs[t1d_tcrs['CDR3b_length'] > 0]['CDR3b_length']
                if not lengths.empty:
                    avg_length = lengths.mean()
                    print(f"\nAverage CDR3b length: {avg_length:.2f} amino acids")

                    length_dist = lengths.value_counts().sort_index()
                    print("\nCDR3b length distribution:")
                    print(length_dist.head(10))


        file_path = "db/positive_t1d_data/adj6975_Data_file_S1.xlsx"  # Update with your actual file path

        print("Extracting T1D-related TCR beta chains...")
        t1d_tcrs = extract_t1d_tcr_beta_chains_from_excel(file_path)

        if not t1d_tcrs.empty:
            print(f"Found {len(t1d_tcrs)} T1D-related TCR beta chains.")

            # Display first few rows
            print("\nSample of extracted T1D TCR beta chains:")
            print(t1d_tcrs.head())

            # Save to CSV
            save_t1d_tcrs(t1d_tcrs)

            # Analyze the data
            analyze_t1d_tcrs(t1d_tcrs)
        else:
            print("No T1D-related TCR beta chains found in the file or could not process the file.")
        return t1d_tcrs

    t1d_tcrs = get_t1d_tcrs()

    # Get all TCRs in ndarray of strings:
    t1d_seqs = np.unique(t1d_tcrs['CDR3b'].values)

    # Apply model:
    trained_model.eval()
    with torch.no_grad():
        out_logits = trained_model(t1d_seqs)
        out_probs = torch.softmax(out_logits, dim=1)[:, 1].cpu().numpy()

    # Display results in histogram
    plt.hist(out_probs, bins=50)
    plt.xlabel('Probability of being positive')
    plt.ylabel('Count')
    plt.title('Distribution of probabilities for T1D TCRs')
    plt.show()

    # Print the number of T1D TCRs which got a prediction of 0.5 or higher
    print(f"Number of T1D TCRs with probability >= 0.5: {np.sum(out_probs >= 0.5)}")



def calculate_lev_distance(seqs, pred, train_pos_seqs):
    seqs_filtered = [(len(seq), seq) for seq, pred in zip(seqs, pred) if pred > 0.5]
    possible_lens = set([x[0] for x in seqs_filtered])
    all_distances = []
    corresponding_seqs = []
    for possible_len in tqdm(possible_lens):
        seqs_in_this_len = [x[1] for x in seqs_filtered if x[0] == possible_len]
        # keeping only the sequences that are of similar length
        # train_pos_seqs_important = [x for x in train_pos_seqs if abs(len(x) - possible_len) <= 2]
        # calculate the Levenshtein distance
        pwc_mat = pairwise_scores(seqs_in_this_len, train_pos_seqs, score=levenshtein_dist_non_bin)
        # get the minimum distance
        min_dist = np.min(pwc_mat, axis=1)
        all_distances.append(min_dist)
        corresponding_seqs.append(seqs_in_this_len)
    combined_distances = np.concatenate(all_distances)
    combined_seqs = np.concatenate(corresponding_seqs)
    return combined_distances, combined_seqs


def inference_ratio_distance(df_bld, df_hlt, trained_model, train_pos_seqs, valid_pos_seqs, valid_neg_seqs,
                             test_pos_seqs, test_neg_seqs, aaseq_to_ratio, dataset_loader):
    base_save_path = f'cache/ratio_distance/'
    os.makedirs(base_save_path, exist_ok=True)
    base_valid_test_df_path = os.path.join(base_save_path, 'base_valid_test_df.csv')
    distance_valid_test_df_path = os.path.join(base_save_path, 'distance_valid_test_df.csv')
    distance_healthy_df_path = os.path.join(base_save_path, 'distance_healthy_df.csv')
    print("Inference on base validation and test set")
    if os.path.exists(base_valid_test_df_path):
        df_valid_test = pd.read_csv(base_valid_test_df_path)
    else:
        # Creating a dataframe with columns: AASeq, ratio_max, ratio_min, ratio_avg, model_prediction, set_origin (train/valid/test/healthy), label
        valid_test_seqs = [valid_pos_seqs, valid_neg_seqs, test_pos_seqs, test_neg_seqs]

        ratios_max = [aaseq_to_ratio(x) for x in valid_test_seqs]
        dataset_loader.build_clone_fraction_df(df_bld, method='min')
        ratios_min = [aaseq_to_ratio(x) for x in valid_test_seqs]
        dataset_loader.build_clone_fraction_df(df_bld, method='avg')
        ratios_avg = [aaseq_to_ratio(x) for x in valid_test_seqs]

        trained_model.eval()
        with torch.no_grad():
            predictions = [torch.softmax(trained_model(x), dim=1)[:, 1].cpu().numpy() for x in valid_test_seqs]

        # Create a dataframe with the results
        valid_test_set_origins = [['valid'] * len(valid_pos_seqs), ['valid'] * len(valid_neg_seqs),
                                  ['test'] * len(test_pos_seqs), ['test'] * len(test_neg_seqs)]
        valid_test_labels = [np.ones(len(valid_pos_seqs)), np.zeros(len(valid_neg_seqs)),
                             np.ones(len(test_pos_seqs)), np.zeros(len(test_neg_seqs))]

        all_dfs = []
        for i in range(len(valid_test_seqs)):
            df = pd.DataFrame({
                'AASeq': valid_test_seqs[i],
                'ratio_max': ratios_max[i],
                'ratio_min': ratios_min[i],
                'ratio_avg': ratios_avg[i],
                'model_prediction': predictions[i],
                'set_origin': valid_test_set_origins[i],
                'label': valid_test_labels[i]
            })
            all_dfs.append(df)
        df_valid_test = pd.concat(all_dfs, ignore_index=True)

        # save to csv into base folder
        df_valid_test.to_csv(base_valid_test_df_path, index=False)

    print("Inference on distances of the validation and test set")
    if os.path.exists(distance_valid_test_df_path):
        df_valid_test = pd.read_csv(distance_valid_test_df_path)
    else:
        # calculate the minimum Levenshtein distance between the sequences with model_prediction > 0.5 in df_valid_test
        # from the set of all positives in the train set (train_pos_seqs).
        combined_distances, combined_seqs = calculate_lev_distance(df_valid_test['AASeq'], df_valid_test['model_prediction'], train_pos_seqs)

        distances = []
        for seq in df_valid_test['AASeq']:
            if seq in combined_seqs:
                # get the index of the sequence in the combined_seqs
                index = np.where(combined_seqs == seq)[0][0]
                distances.append(combined_distances[index])
            else:
                distances.append(0)

        # create a new column in df_valid_test with the distances and save as distance_valid_test_df
        df_valid_test['distances'] = distances
        df_valid_test.to_csv(distance_valid_test_df_path, index=False)

    print("Inference on healthy set")
    if os.path.exists(distance_healthy_df_path):
        df_healthy = pd.read_csv(distance_healthy_df_path)
    else:
        healthy_patients = df_hlt["patient_id"].unique()
        np.random.shuffle(healthy_patients)
        chosen_patients = healthy_patients[:5]
        chosen_df_hlt = df_hlt[df_hlt["patient_id"].isin(chosen_patients)]
        chosen_seqs = np.unique(df_hlt[df_hlt["patient_id"].isin(chosen_patients)]["AASeq"])

        # predict on healthy patients
        trained_model.eval()
        with torch.no_grad():
            patient_prediction = torch.softmax(trained_model(chosen_seqs), dim=1)[:, 1].cpu().numpy()

        # calculate levenshtein distance for the healthy patients
        combined_distances, combined_seqs = calculate_lev_distance(chosen_seqs, patient_prediction, train_pos_seqs)

        distances = []
        for seq in chosen_seqs:
            if seq in combined_seqs:
                # get the index of the sequence in the combined_seqs
                index = np.where(combined_seqs == seq)[0][0]
                distances.append(combined_distances[index])
            else:
                distances.append(0)

        # calculate the ratios for the healthy patients
        dataset_loader.build_clone_fraction_df(chosen_df_hlt, method='min')
        ratios_max = aaseq_to_ratio(chosen_seqs)
        dataset_loader.build_clone_fraction_df(chosen_df_hlt, method='min')
        ratios_min = aaseq_to_ratio(chosen_seqs)
        dataset_loader.build_clone_fraction_df(chosen_df_hlt, method='avg')
        ratios_avg = aaseq_to_ratio(chosen_seqs)

        # create a new dataframe with the distances and save as distance_healthy_df
        df_healthy = pd.DataFrame({
            'AASeq': chosen_seqs,
            'ratio_max': ratios_max,
            'ratio_min': ratios_min,
            'ratio_avg': ratios_avg,
            'model_prediction': patient_prediction,
            'set_origin': ['healthy'] * len(chosen_seqs),
            'distances': distances
        })

        # save the dataframe to csv
        df_healthy.to_csv(distance_healthy_df_path, index=False)

    print("Done with ratio-distance csv files!")

    # display statistics about the distances in relation to the ratio and to the model prediction
    # Perform comprehensive analysis
    analysis_results = perform_comprehensive_analysis(df_valid_test, df_healthy)

    # Display analysis results
    display_analysis_results(analysis_results)

    print('Done with ratio-distance inference!')


def perform_comprehensive_analysis(df_valid_test, df_healthy, ratio_threshold=0.5):
    """
    Perform comprehensive analysis of model predictions, ratios, and distances

    Parameters:
    - df_valid_test: DataFrame containing validation and test sequences
    - df_healthy: DataFrame containing healthy sequences
    - ratio_threshold: Threshold for defining high/low ratio (default: 0.5)

    Returns:
    - Dictionary of analysis results
    """
    # Combine validation and test datasets
    analysis_results = {}

    # 1. Negative Sequences Analysis
    neg_seqs = df_valid_test[df_valid_test['label'] == 0]

    # High vs Low Ratio for Negative Sequences
    neg_high_ratio = neg_seqs[neg_seqs['ratio_avg'] >= ratio_threshold]
    neg_low_ratio = neg_seqs[neg_seqs['ratio_avg'] < ratio_threshold]

    analysis_results['negative_sequences'] = {
        'total_negative_sequences': len(neg_seqs),
        'high_ratio_sequences': len(neg_high_ratio),
        'low_ratio_sequences': len(neg_low_ratio),
        'high_ratio_pred_above_threshold': len(neg_high_ratio[neg_high_ratio['model_prediction'] > 0.5]),
        'low_ratio_pred_above_threshold': len(neg_low_ratio[neg_low_ratio['model_prediction'] > 0.5])
    }

    # 2. Correlation Analysis
    correlation_results = {
        'prediction_ratio_corr': {
            'max_ratio': stats.pearsonr(df_valid_test['model_prediction'], df_valid_test['ratio_max']),
            'min_ratio': stats.pearsonr(df_valid_test['model_prediction'], df_valid_test['ratio_min']),
            'avg_ratio': stats.pearsonr(df_valid_test['model_prediction'], df_valid_test['ratio_avg'])
        },
        'prediction_distance_corr': stats.pearsonr(df_valid_test['model_prediction'], df_valid_test['distances'])
    }
    analysis_results['correlations'] = correlation_results

    # 3. Visualization Functions
    def create_distribution_plots(df_valid_test, df_healthy):
        """Create distribution plots for various metrics"""
        plt.figure(figsize=(20, 12))

        # Prediction Distribution
        plt.subplot(2, 3, 1)
        sns.histplot(data=df_valid_test, x='model_prediction', hue='label', multiple='stack', bins=20)
        plt.title('Validation & Test Set: Model Prediction Distribution')
        plt.xlabel('Model Prediction Probability')
        plt.ylabel('Count')

        # Ratio Distributions (Violin Plot)
        plt.subplot(2, 3, 2)
        sns.violinplot(data=df_valid_test, x='label', y='ratio_max')
        plt.title('Validation & Test Set: Max Ratio by Label')
        plt.xlabel('Label (0: Negative, 1: Positive)')
        plt.ylabel('Max Ratio')

        # Distance Distribution (Violin Plot)
        plt.subplot(2, 3, 3)
        sns.violinplot(data=df_valid_test, x='label', y='distances')
        plt.title('Validation & Test Set: Levenshtein Distance by Label')
        plt.xlabel('Label (0: Negative, 1: Positive)')
        plt.ylabel('Levenshtein Distance')

        # Scatter Plot: Prediction vs Ratio
        plt.subplot(2, 3, 4)
        scatter = plt.scatter(df_valid_test['model_prediction'], df_valid_test['ratio_max'],
                              c=df_valid_test['label'], cmap='viridis', alpha=0.6)
        plt.title('Validation & Test Set: Model Prediction vs Max Ratio')
        plt.xlabel('Model Prediction Probability')
        plt.ylabel('Max Ratio')
        plt.colorbar(scatter, label='Label')

        # Scatter Plot: Prediction vs Distance
        plt.subplot(2, 3, 5)
        plt.scatter(df_valid_test['model_prediction'], df_valid_test['distances'],
                    c=df_valid_test['label'], cmap='viridis', alpha=0.6)
        plt.title('Validation & Test Set: Model Prediction vs Levenshtein Distance')
        plt.xlabel('Model Prediction Probability')
        plt.ylabel('Levenshtein Distance')
        plt.colorbar(label='Label')

        plt.tight_layout()
        plt.savefig('model_analysis_plots_valid_test.png')
        plt.close()

        # Healthy Dataset Analysis
        plt.figure(figsize=(15, 5))

        # Healthy Set: Prediction Distribution
        plt.subplot(1, 3, 1)
        sns.histplot(data=df_healthy, x='model_prediction')
        plt.title('Healthy Set: Model Prediction Distribution')
        plt.xlabel('Model Prediction Probability')
        plt.ylabel('Count')

        # Healthy Set: Ratio Distribution
        plt.subplot(1, 3, 2)
        sns.violinplot(data=df_healthy, x='model_prediction', y='ratio_max')
        plt.title('Healthy Set: Max Ratio vs Prediction')
        plt.xlabel('Model Prediction Probability')
        plt.ylabel('Max Ratio')

        # Healthy Set: Prediction vs Distance
        plt.subplot(1, 3, 3)
        plt.scatter(df_healthy['model_prediction'], df_healthy['distances'],
                    alpha=0.6, c='blue')
        plt.title('Healthy Set: Prediction vs Levenshtein Distance')
        plt.xlabel('Model Prediction Probability')
        plt.ylabel('Levenshtein Distance')

        plt.tight_layout()
        plt.savefig('model_analysis_plots_healthy.png')
        plt.close()

    # 4. Threshold Sensitivity Analysis
    def threshold_sensitivity_analysis(df_valid_test):
        """Analyze model performance across different prediction thresholds"""
        thresholds = np.linspace(0, 1, 21)
        sensitivity_results = []

        for threshold in thresholds:
            # Positive predictions at this threshold
            pos_preds = df_valid_test[df_valid_test['model_prediction'] >= threshold]

            # High ratio analysis
            high_ratio_pos = pos_preds[pos_preds['ratio_avg'] >= ratio_threshold]

            sensitivity_results.append({
                'threshold': threshold,
                'total_predictions': len(pos_preds),
                'high_ratio_percentage': len(high_ratio_pos) / len(pos_preds) * 100 if len(pos_preds) > 0 else 0
            })

        return pd.DataFrame(sensitivity_results)

    # Execute visualizations and additional analyses
    create_distribution_plots(df_valid_test, df_healthy)
    threshold_sensitivity_df = threshold_sensitivity_analysis(df_valid_test)

    # Save threshold sensitivity results
    threshold_sensitivity_df.to_csv('threshold_sensitivity_analysis.csv', index=False)

    # Update analysis results with threshold sensitivity
    analysis_results['threshold_sensitivity'] = threshold_sensitivity_df.to_dict('records')

    return analysis_results


def display_analysis_results(analysis_results):
    """
    Display the comprehensive analysis results in a readable format
    """
    print("\n--- Negative Sequences Analysis ---")
    neg_analysis = analysis_results['negative_sequences']
    print(f"Total Negative Sequences: {neg_analysis['total_negative_sequences']}")
    print(f"Negative Sequences with High Ratio: {neg_analysis['high_ratio_sequences']}")
    print(f"Negative Sequences with Low Ratio: {neg_analysis['low_ratio_sequences']}")
    print(f"High Ratio Sequences with Prediction > 0.5: {neg_analysis['high_ratio_pred_above_threshold']}")
    print(f"Low Ratio Sequences with Prediction > 0.5: {neg_analysis['low_ratio_pred_above_threshold']}")

    print("\n--- Correlation Analysis ---")
    corr_analysis = analysis_results['correlations']
    print("Correlation between Model Prediction and:")
    print(
        f"Max Ratio: {corr_analysis['prediction_ratio_corr']['max_ratio'][0]:.4f} (p-value: {corr_analysis['prediction_ratio_corr']['max_ratio'][1]:.4f})")
    print(
        f"Min Ratio: {corr_analysis['prediction_ratio_corr']['min_ratio'][0]:.4f} (p-value: {corr_analysis['prediction_ratio_corr']['min_ratio'][1]:.4f})")
    print(
        f"Avg Ratio: {corr_analysis['prediction_ratio_corr']['avg_ratio'][0]:.4f} (p-value: {corr_analysis['prediction_ratio_corr']['avg_ratio'][1]:.4f})")
    print(
        f"Levenshtein Distance: {corr_analysis['prediction_distance_corr'][0]:.4f} (p-value: {corr_analysis['prediction_distance_corr'][1]:.4f})")

    print("\n--- Threshold Sensitivity Analysis ---")
    print("Saved detailed results in 'threshold_sensitivity_analysis.csv'")

    # Plotting threshold sensitivity
    plt.figure(figsize=(10, 5))
    thresholds = [entry['threshold'] for entry in analysis_results['threshold_sensitivity']]
    high_ratio_percentages = [entry['high_ratio_percentage'] for entry in analysis_results['threshold_sensitivity']]
    total_predictions = [entry['total_predictions'] for entry in analysis_results['threshold_sensitivity']]

    plt.subplot(1, 2, 1)
    plt.plot(thresholds, high_ratio_percentages, marker='o')
    plt.title('High Ratio Percentage vs Prediction Threshold')
    plt.xlabel('Prediction Threshold')
    plt.ylabel('High Ratio Percentage')

    plt.subplot(1, 2, 2)
    plt.plot(thresholds, total_predictions, marker='o')
    plt.title('Total Predictions vs Prediction Threshold')
    plt.xlabel('Prediction Threshold')
    plt.ylabel('Number of Predictions')

    plt.tight_layout()
    plt.savefig('threshold_sensitivity_plot.png')
    plt.close()
