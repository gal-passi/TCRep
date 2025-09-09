import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import logomaker
from collections import Counter

from inference.plot_dataset import display_common_sequences_figure, display_ratio_figures
from inference.plot_training import (
    plot_average_individual_distributions,
    plot_output_distributions_per_patient,
    # plot_output_distributions_claude,
    # plot_output_distributions_unseen_ms,
)


def plot_sequence_logo(positive_seqs, other_seqs, version=2):
    """
    Generate sequence logo plots for top sequence lengths in positives vs others.
    version=1: frequency matrix logos
    version=2: information content logos
    """
    if version == 1:
        positive_lengths = [len(seq) for seq in positive_seqs]
        top3_lengths = [length for length, _ in Counter(positive_lengths).most_common(3)]

        def get_pfm(seqs, seq_len):
            filtered = [seq for seq in seqs if len(seq) == seq_len]
            if not filtered:
                return None
            return pd.DataFrame([Counter(col) for col in zip(*filtered)]).fillna(0)

        fig, axs = plt.subplots(len(top3_lengths), 2, figsize=(14, 4 * len(top3_lengths)))
        for i, seq_len in enumerate(top3_lengths):
            pos_pfm = get_pfm(positive_seqs, seq_len)
            oth_pfm = get_pfm(other_seqs, seq_len)

            for ax, pfm, title in [(axs[i, 0], pos_pfm, "Positive"), (axs[i, 1], oth_pfm, "Other")]:
                if pfm is not None:
                    logomaker.Logo(pfm, ax=ax)
                    ax.set_title(f"{title} Sequences (Length {seq_len})")
                    ax.set_ylabel("Freq")
                    ax.set_xlabel("Position")
                else:
                    ax.text(0.5, 0.5, "No sequences", ha="center", va="center")
                    ax.axis("off")

        plt.tight_layout()
        plt.show()

    else:  # version 2
        tmp_pos_seqs = np.array([seq[1:-1] for seq in positive_seqs])  # strip codons
        tmp_other_seqs = np.array([seq[1:-1] for seq in other_seqs])

        positive_lengths = [len(seq) for seq in tmp_pos_seqs]
        top_len = Counter(positive_lengths).most_common(1)[0][0]

        def get_info_matrix(seqs, seq_len):
            filtered = [seq for seq in seqs if len(seq) == seq_len]
            if not filtered:
                return None
            columns = list(zip(*filtered))
            counts_df = pd.DataFrame([Counter(col) for col in columns]).fillna(0)
            counts_df.index = range(1, seq_len + 1)
            return logomaker.transform_matrix(counts_df, from_type="counts", to_type="information")

        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        for ax in axs:
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
                spine.set_linewidth(1.0)

        pos_matrix = get_info_matrix(tmp_pos_seqs, top_len)
        oth_matrix = get_info_matrix(tmp_other_seqs, top_len)

        for ax, matrix, title in [(axs[0], pos_matrix, "Positive"), (axs[1], oth_matrix, "Other")]:
            if matrix is not None:
                logomaker.Logo(matrix, ax=ax)
                ax.set_title(f"{title} Sequences (Length {top_len})")
                ax.set_ylabel("Bits")
                ax.set_xlabel("Position")
                ax.set_xticks(range(1, top_len + 1))
                ax.set_ylim(0, 3.55)
            else:
                ax.text(0.5, 0.5, "No sequences", ha="center", va="center")
                ax.axis("off")

        plt.tight_layout()
        plt.show()


def plot_all_training(trained_model, args,
                      positive_seqs, neg_seqs, valid_neg_seqs, test_neg_seqs,
                      test_patient_inds, valid_patient_inds, unique_patient_ids,
                      test_masks, valid_masks, df_bld, df_hlt,
                      sample_plots, device):
    """
    Wrapper to handle all post-training plots depending on flags.
    """
    if args.dont_plot:
        return

    # Optional sequence logos
    if getattr(args, "plot_extra_sequence_logo", False):
        other_seqs = np.concatenate([neg_seqs, valid_neg_seqs, test_neg_seqs])
        plot_sequence_logo(positive_seqs, other_seqs, version=2)

    # Output distributions
    np.random.seed(42)
    print("Plotting the output distributions per patient")
    plot_output_distributions_per_patient(
        trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
        test_masks, valid_masks, positive_seqs, df_bld, df_hlt,
        args.model_type, not args.no_wandb_log, sample_plots, args, device
    )

    np.random.seed(42)
    print("Plotting the average individual distributions per patient")
    plot_average_individual_distributions(
        trained_model, test_patient_inds, valid_patient_inds, unique_patient_ids,
        test_masks, valid_masks, positive_seqs, df_bld, df_hlt,
        args.model_type, not args.no_wandb_log, args, device
    )


def plot_all_dataset(dataset_loader, df_bld, df_hlt, dataset_type, train_patient_ids, positive_seqs,
                     neg_seqs, aaseq_to_ratio, to_display_dict):
    # Display common sequences in disease and healthy samples
    if to_display_dict.get("TO_DISPLAY_COMMON_SEQUENCES", False):
        l = 8
        try:
            display_common_sequences_figure(dataset_loader, df_bld, df_hlt, dataset_type, l=min(l, len(train_patient_ids)))
        except Exception as e:
            print(f"Error displaying common sequences figure: {e}")
            print("Skipping the display of common sequences figure.")

    if to_display_dict.get("TO_DISPLAY_RATIO_FIGURES", False):
        display_ratio_figures(df_bld, positive_seqs, neg_seqs, aaseq_to_ratio, dataset_type)

