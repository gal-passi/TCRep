import os
import torch
import numpy as np
from itertools import chain
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_handlers.processing_and_filtering import filter_df
import pickle


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


def display_common_sequences_figure(dataset_loader, df, df_h, dataset_type, l=8, log_space=True, to_recalculate=False, num_rand_patients=60):
    base_plot_save_path = f"plots/common_seqs/{dataset_type}"
    os.makedirs(base_plot_save_path, exist_ok=True)

    if not to_recalculate and os.path.exists(os.path.join(base_plot_save_path, 'plot_common_sequences.png')):
        print(f"Common Sequences plot already exists for dataset {dataset_type}, skipping...")
        return
    else:
        print(f"Calculating common sequences for dataset {dataset_type}...")

    # find max len of uniques patient_id
    if l == None:
        l = min(1 + len(df_h['patient_id'].unique()), len(df['patient_id'].unique())) + 1

    num_rand_patients = min([num_rand_patients, len(df['patient_id'].unique()), len(df_h['patient_id'].unique())])

    value_to_take = "percent_of_total"  # "percent_of_total" or "num_common"

    # calculate common sequences in disease and healthy samples
    rand_patients_disease = np.random.choice(df['patient_id'].unique(), size=num_rand_patients, replace=False)
    df_df = df[df['patient_id'].isin(rand_patients_disease)]
    x_disease_list = [dataset_loader.common_aaseq_analysis(df_df, num_of_patients=i, mode=1, std_val=value_to_take) for i in range(2, l)]
    x_disease = np.array([x[0][value_to_take] for x in x_disease_list])
    x_disease_std = np.array([x[1] for x in x_disease_list])

    random_patients_healthy = np.random.choice(df_h['patient_id'].unique(), size=num_rand_patients, replace=False)
    df_hf = df_h[df_h['patient_id'].isin(random_patients_healthy)]
    x_healthy_list = [dataset_loader.common_aaseq_analysis(df_hf, num_of_patients=i, mode=1, std_val=value_to_take) for i in range(2, l)]
    x_healthy = np.array([x[0][value_to_take] for x in x_healthy_list])
    x_healthy_std = np.array([x[1] for x in x_healthy_list])

    if log_space and (value_to_take != "num_common"):
        x_disease = np.log(x_disease)
        x_healthy = np.log(x_healthy)

    # Figure without legend
    plt.figure(figsize=(6, 6), dpi=600)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.75)
    plt.plot(range(2, len(x_disease) + 2), x_disease, label="Patients", color="#FFA500")
    plt.plot(range(2, len(x_healthy) + 2), x_healthy, label="Healthy", color="#7BC8F6")
    plt.fill_between(range(2, len(x_disease) + 2), x_disease - x_disease_std, x_disease + x_disease_std,
                     color="#FFA500", alpha=0.2)
    plt.fill_between(range(2, len(x_healthy) + 2), x_healthy - x_healthy_std, x_healthy + x_healthy_std,
                     color="#7BC8F6", alpha=0.2)
    # add the percentage of common sequences in the plot with rounded values
    for i, txt in enumerate(x_disease):
        plt.annotate(f"{txt:.2f}", (i + 2, x_disease[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    for i, txt in enumerate(x_healthy):
        plt.annotate(f"{txt:.2f}", (i + 2, x_healthy[i]), textcoords="offset points", xytext=(0, 10), ha='center')
    plt.xlabel("Number of Patients")
    plt.ylabel("Percentage of Common Sequences")
    plt.title("Percentage of Common Sequences in Patients" + (" (Log Scale)" if log_space else ""))
    plt.ylim(min(min(x_disease), min(x_healthy)),
             max(max(x_disease + x_disease_std), max(x_healthy + x_healthy_std)) * 1.)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(base_plot_save_path, 'plot_common_sequences_nolegend.png'))
    plt.legend(framealpha=1.0)
    plt.savefig(os.path.join(base_plot_save_path, 'plot_common_sequences.png'))
    plt.show()
    pass


def display_multiple_common_sequences_figure(dataset_loader, df_dict, dataset_type, l=8, log_space=True, to_recalculate=True, num_rand_patients=30):
    """
    Display common sequences figure for multiple dataframes.

    Args:
        dataset_loader: The dataset loader object with common_aaseq_analysis method
        df_dict: Dictionary where keys are labels and values are dataframes
                 e.g., {'Patients': df_disease, 'Healthy': df_healthy, 'Control': df_control}
        dataset_type: String identifier for the dataset
        l: Maximum number of patients to analyze
        log_space: Whether to apply log transformation
        to_recalculate: Whether to recalculate if plot exists
        num_rand_patients: Number of random patients to sample
    """
    base_plot_save_path = f"plots/common_seqs/{dataset_type}"
    # os.makedirs(base_plot_save_path, exist_ok=True)

    if not to_recalculate and os.path.exists(os.path.join(base_plot_save_path, 'plot_common_sequences.png')):
        return

    def sample_patients(df, num_patients, dataset_type):
        if 'cmv' in dataset_type or 'article_sle' in dataset_type or 't1d' in dataset_type:
            if 'study_id' in df.columns:
                study_groups = df.groupby('study_id')['patient_id'].unique().apply(list)
                patients_per_study = min(num_patients // len(study_groups) + 1, num_patients)
                rand_patients = [np.random.choice(x, size=min(patients_per_study, len(x)), replace=False) for x in
                                 study_groups]
                rand_patients = list(chain(*rand_patients))
            else:
                rand_patients = np.random.choice(df['patient_id'].unique(),
                                                 size=min(len(df['patient_id'].unique()), num_patients),
                                                 replace=False)
        elif 'study_id' in df.columns:
            study_groups = df.groupby('study_id')['patient_id'].unique().apply(list)
            rand_patients = [np.random.choice(x, size=min(5, len(x)), replace=False) for x in study_groups]
            rand_patients = list(chain(*rand_patients))
        else:
            rand_patients = np.random.choice(df['patient_id'].unique(),
                                             size=min(len(df['patient_id'].unique()), num_patients),
                                             replace=False)

        if len(rand_patients) < num_patients:
            remaining_patients = set(df['patient_id'].unique()) - set(rand_patients)
            additional_needed = min(num_patients - len(rand_patients), len(remaining_patients))
            if additional_needed > 0:
                additional_patients = np.random.choice(list(remaining_patients), size=additional_needed, replace=False)
                rand_patients = np.concatenate([rand_patients, additional_patients])

        return rand_patients

    def calculate_common_sequences(df, label):
        sampled_patients = sample_patients(df, num_rand_patients, dataset_type)
        df_sampled = df[df['patient_id'].isin(sampled_patients)]

        if l is None:
            max_l = len(df_sampled['patient_id'].unique()) + 1
        else:
            max_l = min(l, len(df_sampled['patient_id'].unique()) + 1)

        value_to_take = "percent_of_total"
        x_list = [dataset_loader.common_aaseq_analysis(df_sampled, num_of_patients=i, mode=1) for i in range(2, max_l)]
        x_values = np.array([x[0][value_to_take] for x in x_list])
        x_std = np.array([x[1] for x in x_list])

        return x_values, x_std, max_l

    results = {}
    colors = [
        "#F58231",  # Orange
        "#46F0F0",  # Cyan
        "#0082C8",  # Blue
        "#3CB44B",  # Green
        "#911EB4",  # Purple
        "#FFE119",  # Yellow
        "#E6194B",  # Red
    ]

    for i, (label, df) in enumerate(df_dict.items()):
        x_values, x_std, max_l = calculate_common_sequences(df, label)
        results[label] = {
            'values': x_values,
            'std': x_std,
            'max_l': max_l,
            'color': colors[i % len(colors)]
        }

    if log_space:
        for label in results:
            results[label]['values'] = np.log(results[label]['values'])

    plt.figure(figsize=(6, 6), dpi=600)
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(0.75)

    all_mins, all_maxs = [], []

    for label, data in results.items():
        x_range = range(2, len(data['values']) + 2)
        plt.plot(x_range, data['values'], label=label, color=data['color'])
        plt.fill_between(x_range, data['values'] - data['std'], data['values'] + data['std'],
                         color=data['color'], alpha=0.2)

        # for i, txt in enumerate(data['values']):
        #     plt.annotate(f"{txt:.2f}", (i + 2, data['values'][i]),
        #                  textcoords="offset points", xytext=(0, 10), ha='center')

        all_mins.append(np.min(data['values']))
        all_maxs.append(np.max(data['values'] + data['std']))

    plt.xlabel("Number of Patients")
    plt.ylabel("Percentage of Common Sequences")
    plt.title("Percentage of Common Sequences" + (" (Log Scale)" if log_space else ""))
    plt.ylim(min(all_mins), max(all_maxs) * 1.1)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()

    # plt.savefig(os.path.join(base_plot_save_path, 'plot_common_sequences_nolegend.png'))
    plt.legend(framealpha=1.0)
    # plt.savefig(os.path.join(base_plot_save_path, 'plot_common_sequences.png'))
    plt.show()
    pass


def display_background_top_n_sequences_model_figure(wrapper_load_dataset, wrapper_common_aaseq_analysis, df_bld, df_hlt,
                                                    dataset_type, num_of_patients=3, num_of_healthy=3,
                                                    max_combs=15000, value_to_take='percent_of_total', use_pos_per_topn=False, use_num_common_markersize=False):
    possible_patients = sorted(df_bld['patient_id'].unique())
    possible_healthy = sorted(df_hlt['patient_id'].unique())
    # pick 60 random patients from disease and healthy groups (using local np rng variable)
    rng = np.random.default_rng(42)
    selected_patients = rng.choice(possible_patients, size=min(60, len(possible_patients)), replace=False)
    selected_healthy = rng.choice(possible_healthy, size=min(60, len(possible_healthy)), replace=False)
    positives_per_topn = [(20, 29507), (17, 20311), (15, 17022), (12, 16747), (10, 12273), (7, 11182), (5, 8440)][::-1]
    use_pos_per_topn = num_of_patients == 3 and num_of_healthy == 3 and use_pos_per_topn and not use_num_common_markersize

    x_diseases = []
    x_diseases_std = []
    x_diseases_num_common = []
    x_healthies = []
    x_healthies_std = []
    x_healthies_num_common = []
    top_n_seqs_lst = [5, 7, 10, 12, 15, 17, 20]
    for top_n in top_n_seqs_lst:
        print(f"Top {top_n}k sequences:")
        dataset_type_tmp = dataset_type[:-3] + f"{top_n}k"
        dataset_tmp = wrapper_load_dataset(dataset_type_tmp, top_n_seqs_wrap=top_n)
        df_disease, df_healthy = dataset_tmp

        # filter df_disease and df_healthy to only include selected patients
        df_disease = df_disease[df_disease['patient_id'].isin(selected_patients)]
        df_healthy = df_healthy[df_healthy['patient_id'].isin(selected_healthy)]

        x_disease_pair = wrapper_common_aaseq_analysis(df_disease, num_of_patients, mode=1, max_combinations=max_combs, std_val=value_to_take)
        x_disease_result = x_disease_pair[0]
        x_disease = x_disease_result[value_to_take]
        x_disease_std = x_disease_pair[1]

        x_healthy_pair = wrapper_common_aaseq_analysis(df_healthy, num_of_healthy, mode=1, max_combinations=max_combs, std_val=value_to_take)
        x_healthy_result = x_healthy_pair[0]
        x_healthy = x_healthy_result[value_to_take]
        x_healthy_std = x_healthy_pair[1]

        x_diseases.append(x_disease)
        x_diseases_std.append(x_disease_std)
        x_diseases_num_common.append(x_disease_result['num_common'])
        x_healthies.append(x_healthy)
        x_healthies_std.append(x_healthy_std)
        x_healthies_num_common.append(x_healthy_result['num_common'])

    # pickle save of x_diseases, x_diseases_std, x_diseases_num_common, x_healthies, x_healthies_std, x_healthies_num_common, top_n_seqs_lst
    # with open(f"cache/datasets_information/{dataset_type[:-8]}/background_data_d{num_of_patients}.h{num_of_healthy}.pkl", "wb") as f:
    #     pickle.dump((x_diseases, x_diseases_std, x_diseases_num_common, x_healthies, x_healthies_std, x_healthies_num_common, top_n_seqs_lst), f)

    # plotting
    plt.figure(figsize=(6, 6), dpi=600)
    fontsize = 14
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.0)
    if use_pos_per_topn:
        # like: plt.plot(top_n_seqs_lst, x_diseases, label="Patients", color="#FFA500", marker='o'), but add markersize according to positives_per_topn list
        marker_sizes = np.array([x[1] for x in positives_per_topn])
        marker_sizes = marker_sizes - min(marker_sizes)
        marker_sizes = marker_sizes / max(marker_sizes) * 150 + 50
        plt.plot(top_n_seqs_lst, x_diseases, label="Patients", color="#FFA500", linewidth=1)  # line only
        plt.scatter(top_n_seqs_lst, x_diseases, color="#FFA500", s=marker_sizes, zorder=5)  # variable dots
        plt.plot(top_n_seqs_lst, x_healthies, label="Healthy", color="#7BC8F6", marker='o')
    elif use_num_common_markersize:
        marker_sizes_disease = np.array(x_diseases_num_common) - min(x_diseases_num_common)
        marker_sizes_disease /= max(marker_sizes_disease)
        marker_sizes_healthy = np.array(x_healthies_num_common) - min(x_healthies_num_common)
        marker_sizes_healthy /= max(marker_sizes_healthy)
        marker_sizes_disease = marker_sizes_disease * 150 + 50
        marker_sizes_healthy = marker_sizes_healthy * 150 + 50
        plt.plot(top_n_seqs_lst, x_diseases, label="Patients", color="#FFA500", linewidth=1)
        plt.scatter(top_n_seqs_lst, x_diseases, color="#FFA500", s=marker_sizes_disease, zorder=5)
        plt.plot(top_n_seqs_lst, x_healthies, label="Healthy", color="#7BC8F6", linewidth=1)
        plt.scatter(top_n_seqs_lst, x_healthies, color="#7BC8F6", s=marker_sizes_healthy, zorder=5)
    else:
        plt.plot(top_n_seqs_lst, x_diseases, label="Patients", color="#FFA500", marker='o', markersize=8)
        plt.plot(top_n_seqs_lst, x_healthies, label="Healthy", color="#7BC8F6", marker='o', markersize=8)
    def calc_error(x, q=75):
        q_upper, q_lower = np.percentile(x, [q, 100 - q])
        iqr = q_upper - q_lower
        robust_std = iqr / 1.349  # convert IQR to "std-like" measure
        return robust_std
    x_disease_err = np.array([calc_error(x) for x in x_diseases_std])
    x_healthies_err = np.array([calc_error(x) for x in x_healthies_std])
    # add error bars as filled area
    plt.fill_between(top_n_seqs_lst, np.array(x_diseases) - x_disease_err,
                     np.array(x_diseases) + x_disease_err, color="#FFA500", alpha=0.14)
    plt.fill_between(top_n_seqs_lst, np.array(x_healthies) - x_healthies_err,
                     np.array(x_healthies) + x_healthies_err, color="#7BC8F6", alpha=0.15)
    x_disease_err = np.array([calc_error(x, q=60) for x in x_diseases_std])
    x_healthies_err = np.array([calc_error(x, q=60) for x in x_healthies_std])
    plt.fill_between(top_n_seqs_lst, np.array(x_diseases) - x_disease_err,
                     np.array(x_diseases) + x_disease_err, color="#FFA500", alpha=0.16)
    plt.fill_between(top_n_seqs_lst, np.array(x_healthies) - x_healthies_err,
                     np.array(x_healthies) + x_healthies_err, color="#7BC8F6", alpha=0.26)
    for i, txt in enumerate(x_diseases):
        if i == 5:
            offset = (0, -20)
        else:
            offset = (0, 10)
        plt.annotate(f"{txt:.2f}",
                     (top_n_seqs_lst[i], x_diseases[i]),
                     textcoords="offset points",
                     xytext=offset,
                     ha='center',
                     fontsize=fontsize)
    # for i, txt in enumerate(x_diseases):
    #     plt.annotate(f"{txt:.2f}", (top_n_seqs_lst[i], x_diseases[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=fontsize)
    for i, txt in enumerate(x_healthies):
        plt.annotate(f"{txt:.2f}", (top_n_seqs_lst[i], x_healthies[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=fontsize)
    plt.xlabel("Top N Sequences (Thousands)", fontsize=fontsize + 2)
    plt.ylabel("Percentage of Common Sequences", fontsize=fontsize + 2)
    # plt.title("Percentage of Common Sequences vs Top N Sequences", fontsize=fontsize)
    plt.ylim(-0.025, 0.49)
    # plt.ylim(min(min(x_diseases), min(x_healthies)) - 0.025, max(max(x_diseases), max(x_healthies)) * 1.1)
    plt.xticks(top_n_seqs_lst, fontsize=fontsize)
    plt.yticks(fontsize=fontsize, ticks=np.arange(0, 0.5, 0.1))
    plt.tight_layout()
    plt.legend(framealpha=1.0, fontsize=fontsize)
    os.makedirs(f"plots/common_seqs/{dataset_type[:-8]}", exist_ok=True)
    if use_num_common_markersize:
        metadata_name = "_num_common_markersize"
    elif use_pos_per_topn:
        metadata_name = "_positives_per_topn_markersize"
    else:
        metadata_name = ""
    plt.savefig(f"plots/common_seqs/{dataset_type[:-8]}/background{metadata_name}_d{num_of_patients}.h{num_of_healthy}.png")
    plt.show()
    pass
    exit(0)


# VALUES OF USING MS: (about 5000 combinations max), (d=3,h=3)
# x_diseases, x_diseases_std, x_diseases_num_common, x_healthies, x_healthies_std, x_healthies_num_common = ([0.2795126057722363, 0.28591238845895617, 0.2959588851055297, 0.2729448900857168, 0.2974419908532758, 0.2869916832830128, 0.27429219762815105], [0.4106126432522795, 0.4060239284024549, 0.37447192838338167, 0.32270263951779926, 0.3097617179307628, 0.2927543883215102, 0.22900267693656562], [58.5596, 86.9002, 135.1578, 176.803, 224.2084, 248.6418, 297.993], [0.05352490380092235, 0.08298810799457529, 0.1189856398313361, 0.15308367546928808, 0.19839663105729516, 0.2287788306440118, 0.3205993793724343], [0.09050303105014607, 0.12292624919562507, 0.17418149410344325, 0.18384742713065522, 0.21929576028669573, 0.20960862482580742, 0.24004790278231955], [15.4466, 34.8018, 87.4318, 185.2924, 303.9572, 422.2288, 670.281])
# positives_per_topn = [(20, 29507), (17, 20311), (15, 17022), (12, 16747), (10, 12273), (7, 11182), (5, 8440)]
#
# Old:
# x_diseases, x_healthies = ([0.21397144835741966, 0.22693402844148589, 0.23535420194141768, 0.24256527777898937, 0.24961090419707493, 0.25389163380619323, 0.25522848644601787], [0.29821382238721594, 0.29821382238721594, 0.29821382238721594, 0.29821382238721594, 0.29821382238721594, 0.29821382238721594, 0.29821382238721594])
# x_diseases_std, x_healthies_std = ([0.17001466811852992, 0.16028585981531132, 0.15310538280199465, 0.14844809529730568, 0.14359903000517413, 0.14080844216902985, 0.13951895164505934], [0.19272245882652694, 0.19272245882652694, 0.19272245882652694, 0.19272245882652694, 0.19272245882652694, 0.19272245882652694, 0.19272245882652694])

# VALUES OF USING SLE: (about 5000 combinations max), (d=3,h=3), (No positives_per_topn available)
# x_diseases, x_diseases_std, x_diseases_num_common, x_healthies, x_healthies_std, x_healthies_num_common = ([0.12215009569840901, 0.1308212874823904, 0.13650884532946492, 0.13949168206511967, 0.14642858503434142, 0.14738740384391869, 0.15938390314288495], [0.07116410360414938, 0.07301962680273916, 0.06907161158086068, 0.06801288698862934, 0.07077520322532871, 0.06881667192026708, 0.0748420265865219], [25.3002, 34.828, 59.4446, 67.1616, 93.1182, 107.5898, 137.1978], [0.1044895062686026, 0.10816588987253895, 0.10223301927474347, 0.10889114804892941, 0.11731117907070536, 0.12238296264571691, 0.1621762253519902], [0.08298392043534074, 0.08068601550842336, 0.06786892516061076, 0.06982497132245372, 0.07158112529311941, 0.07323622408050343, 0.08307317075859771], [23.3208, 30.8388, 62.7952, 75.593, 93.3006, 106.517, 151.1002])


