import os
import pickle
import torch
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
from tqdm import tqdm
from dataset_handlers.Curation import Study, filter_df
from utils.utils import pairwise_scores, levenshtein_dist_non_bin
import hashlib
import random
import matplotlib.pyplot as plt
from dataset_handlers.constants import *
from dataset_handlers.helpers import helper_function_common_aaseq_analysis, process_combination, run_multiprocess_analysis
from dataset_handlers.processing_and_filtering import (apply_extra_filter, validate_required_columns, split_patients,
                                      calculate_positive_sequences, build_patient_masks, finalize_pos_neg_seqs)
from dataset_handlers.losses_utils import LossPreprocessor
from dataset_handlers.dataset_builder import build_ms_dataset, build_sle_dataset, build_t1d_dataset, build_other_dataset


# TODO: Formatting changes:
#  1. remove redundant unused functions
#  2. move more code from the init function of the DatasetLoader class (to other files)
class DatasetLoader:
    def __init__(self, dataset_type: str, unique_patient_ids=None, get_only_unique_patient_ids=False, k_fold=0,
                 dist_loss_type='none', neg_partition=0, use_similar_negatives=False, neg_pos_ratio=10,
                 filter_num_of_patients=3, filter_num_of_healthy=3, filter_to_inflate=False, ratio=None,
                 remove_seqs_by_len=False, top_percent=None, top_n_seqs=None, extra_filter=False,
                 use_nneighbors_loss=False, display_extra_plots=False, loss_version=0, run_on_full_data=False, num_test_patients=8, verbose=True):
        self.dataset_type = dataset_type
        self.top_percent = top_percent
        self.top_n_seqs = top_n_seqs
        self.losses_preprocessor = None

        # cache folder members
        self.cache_path = 'cache'
        self.cache_dataframes_path = os.path.join(self.cache_path, "dataloader_cache/dataframes")
        self.cache_dataframes_healthy_path = os.path.join(self.cache_path, "dataloader_cache/dataframes_healthy")
        self.saved_dataframe_path = os.path.join(self.cache_dataframes_path, f"{dataset_type}_df.pkl")
        self.cache_sequences_path = os.path.join(self.cache_path, "dataloader_cache/processed_sequences")
        self.synapse_db_path = "data/db/synapse_Mal_ID"

        disease = 'Multiple sclerosis'
        print('Loading Disease:')
        df = self.get_all_usable_disease_data(disease=disease, dataset_type=dataset_type)
        print('Loading Healthy:')
        df_h = self.get_all_usable_healthy_data(dataset_type=dataset_type)

        # load from saved dataset if exists
        saved_df = None
        if os.path.exists(self.saved_dataframe_path):
            with open(self.saved_dataframe_path, 'rb') as f:
                saved_df = pickle.load(f)

        dataset = build_ms_dataset(self, df, df_h, dataset_type, top_percent=top_percent, top_n_seqs=top_n_seqs, extra_filter=extra_filter)
        if dataset is None:
            dataset = build_sle_dataset(self, saved_df, dataset_type, top_percent=top_percent, top_n_seqs=top_n_seqs, extra_filter=extra_filter)
        if dataset is None:
            dataset = build_t1d_dataset(self, saved_df, dataset_type, top_percent=top_percent, top_n_seqs=top_n_seqs, extra_filter=extra_filter)
        if dataset is None:
            dataset = build_other_dataset(self, saved_df, dataset_type, top_percent=top_percent, top_n_seqs=top_n_seqs, extra_filter=extra_filter)
        if dataset is None:
            raise ValueError("Invalid dataset type")
        elif 'hlt_as_ms' in dataset_type:
            df_bld, df_excess_hlt, df_hlt = dataset
        else:
            df_bld, df_hlt = dataset

        if verbose:
            print('Done loading datasets.')

        # Check that the required columns are in the dataframes
        df_bld, df_hlt = validate_required_columns(df_bld, df_hlt)

        # Display extra plots if needed
        if display_extra_plots:
            # For disease group
            for study_id in df_bld['study_id'].unique():
                df_study = df_bld[df_bld['study_id'] == study_id]
                self.plot_per_patient_histogram(
                    dfs=[df_study],
                    labels=[f"Disease ({study_id})"],
                    colors=["salmon"],
                    disease=f"Disease - study: {study_id}",  # or replace with actual disease name if known
                    to_add_top_percent=False,
                    to_add_bar_vals=True
                )
            # For healthy group
            for study_id in df_hlt['study_id'].unique():
                df_study = df_hlt[df_hlt['study_id'] == study_id]
                self.plot_per_patient_histogram(
                    dfs=[df_study],
                    labels=[f"Healthy ({study_id})"],
                    colors=["lightblue"],
                    disease=f"Healthy - study: {study_id}",
                    to_add_top_percent=False,
                    to_add_bar_vals=True
                )

        # returning only df_bld and df_hlt
        if get_only_unique_patient_ids:
            self.df_bld, self.df_hlt = df_bld, df_hlt
            return

        # pick unique patient ids if not provided
        split_patients_out = split_patients(df_bld, num_test_patients, unique_patient_ids, run_on_full_data)
        train_patient_ids, valid_patient_ids, test_patient_ids, unique_patient_ids, df_bld = split_patients_out

        # calculate positive sequences of train, validation and test sets
        pos_seqs_res = calculate_positive_sequences(df_bld, df_hlt, dataset_type, self.cache_sequences_path,
                                                    train_patient_ids, valid_patient_ids, test_patient_ids,
                                                    filter_num_of_patients, filter_num_of_healthy, filter_to_inflate,
                                                    extra_filter, k_fold, top_percent, top_n_seqs, verbose)
        train_pos_seqs, valid_pos_seqs, test_pos_seqs = pos_seqs_res

        # TODO: This code is for the case where we tested adding healthy to the MS dataset and only then running the code. Consider removing now!
        #  It is located in this position of the code because I wanted to calculate positives first
        #  without those injected healthy patients and only then add them to the data.
        if 'tcrdb2' in dataset_type and '_hlt_as_ms' in dataset_type:
            # append df_excess_hlt to the df_bld set and add positives to train accordingly:
            df_bld = pd.concat([df_bld, df_excess_hlt], axis=0, ignore_index=True)

        # make sure that there is no intersection between train, valid and test sequences
        train_pos_seqs = np.array(list(set(train_pos_seqs) - set(np.concatenate((valid_pos_seqs, test_pos_seqs)))))
        if len(set(valid_pos_seqs) & set(test_pos_seqs)) > 0:
            if len(valid_pos_seqs) > len(test_pos_seqs):
                valid_pos_seqs = np.array(list(set(valid_pos_seqs) - set(test_pos_seqs)))
            else:
                test_pos_seqs = np.array(list(set(test_pos_seqs) - set(valid_pos_seqs)))

        if display_extra_plots:
            # Train-positive AASeqs (assumed to be a set for speed)
            train_pos_seqs = set(train_pos_seqs)

            hlt_patient_ids = df_hlt['patient_id'].unique()
            res = []
            for p in hlt_patient_ids:
                patient_seqs = set(df_hlt[df_hlt['patient_id'] == p].AASeq)
                shared_seqs = patient_seqs.intersection(train_pos_seqs)
                if len(patient_seqs) > 0:
                    percent = 100 * len(shared_seqs) / len(patient_seqs)
                else:
                    percent = 0
                res.append(percent)

            all_bld_ids = df_bld['patient_id'].unique()
            filtered_bld_ids = all_bld_ids[:-8]  # exclude last 8

            res_d = []
            for p in filtered_bld_ids:
                patient_seqs = set(df_bld[df_bld['patient_id'] == p].AASeq)
                shared_seqs = patient_seqs.intersection(train_pos_seqs)
                if len(patient_seqs) > 0:
                    percent = 100 * len(shared_seqs) / len(patient_seqs)
                else:
                    percent = 0
                res_d.append(percent)

            # Sort by descending percentage
            res = sorted(res, reverse=True)
            res_d = sorted(res_d, reverse=True)

            x_hlt = np.arange(len(res))
            x_bld = np.arange(len(res_d))

            plt.figure(figsize=(12, 6))
            plt.bar(x_bld, res_d, color='salmon', label='Disease', alpha=0.5)
            plt.bar(x_hlt, res, color='skyblue', label='Healthy', alpha=0.5)

            plt.xlabel("Patient index (sorted by % overlap)")
            plt.ylabel("Percentage of patient's unique AASeqs in train_pos_seqs (%)")
            plt.title("Patient Overlap with Positive Training Sequences")
            plt.legend()
            plt.tight_layout()
            plt.show()

        # define positive sequences
        positive_seqs = np.concatenate((train_pos_seqs, valid_pos_seqs, test_pos_seqs))

        # Calculate the masks for each patient
        res = build_patient_masks(df_bld, unique_patient_ids, positive_seqs, train_patient_ids, valid_patient_ids, test_patient_ids)
        train_inds, valid_inds, test_inds, train_masks, valid_masks, test_masks, patient_id_masks, test_patient_inds, valid_patient_inds, train_patient_inds = res

        if verbose:
            print(f"Number of Positive Sequences in General: {len(positive_seqs)}")
            print(f"Number of Positive Sequences in Test: {sum(test_inds)}, Percentage: {sum(test_inds) / len(positive_seqs) * 100:.2f}%")
            print(f"Number of Positive Sequences in Valid: {sum(valid_inds)}, Percentage: {sum(valid_inds) / len(positive_seqs) * 100:.2f}%")
            print(f"Number of Positive Sequences in Train: {sum(train_inds)}, Percentage: {sum(train_inds) / len(positive_seqs) * 100:.2f}%\n")
            print(f"Number of patients in General: {df_bld['patient_id'].nunique() + df_hlt['patient_id'].nunique()}")
            print(f"Number of Disease patients: {df_bld['patient_id'].nunique()}")
            print(f"Number of Healthy patients: {df_hlt['patient_id'].nunique()}")

        # finalize positive and negative sequences
        res = finalize_pos_neg_seqs(df_bld, train_patient_ids, valid_patient_ids, test_patient_ids,
                                    positive_seqs, train_inds, valid_inds, test_inds)
        train_pos_seqs, valid_pos_seqs, test_pos_seqs, neg_seqs, valid_neg_seqs, test_neg_seqs = res

        if verbose:
            print('Done.')

        if use_similar_negatives:
            neg_seqs = self.use_similar_negatives_handler(neg_seqs, train_pos_seqs, neg_pos_ratio, neg_partition)

        if remove_seqs_by_len:
            df_bld = df_bld[df_bld['AASeq'].str.len() > 11]  # remove sequences that are too short
            df_hlt = df_hlt[df_hlt['AASeq'].str.len() > 11]  # remove sequences that are too short
            df_bld = df_bld[df_bld['AASeq'].str.len() < remove_seqs_by_len]  # remove sequences that are too long
            df_hlt = df_hlt[df_hlt['AASeq'].str.len() < remove_seqs_by_len]  # remove sequences that are too long
            test_pos_seqs = np.array([seq for seq in test_pos_seqs if 11 < len(seq) < remove_seqs_by_len])
            test_neg_seqs = np.array([seq for seq in test_neg_seqs if 11 < len(seq) < remove_seqs_by_len])
            valid_pos_seqs = np.array([seq for seq in valid_pos_seqs if 11 < len(seq) < remove_seqs_by_len])
            valid_neg_seqs = np.array([seq for seq in valid_neg_seqs if 11 < len(seq) < remove_seqs_by_len])
            train_pos_seqs = np.array([seq for seq in train_pos_seqs if 11 < len(seq) < remove_seqs_by_len])
            neg_seqs = np.array([seq for seq in neg_seqs if 11 < len(seq) < remove_seqs_by_len])
            # Calculate the masks for each patient
            masks = []
            for patient in unique_patient_ids:
                # Get sequences that belong to the current patient
                patient_seqs = set(df_bld.loc[df_bld["patient_id"] == patient, "AASeq"])
                # Create a mask for sequences
                mask = np.array([1 if seq in patient_seqs else 0 for seq in positive_seqs])
                if 1 in mask:
                    masks.append(mask)
                else:
                    print(f"Patient {patient} has NO positive sequences! Look into this case!")
                    masks.append(mask)
            # Convert to ndarray
            patient_id_masks = np.array(masks)  # Shape: (num_unique_patients, len(positive_seqs))
            # Get the masks for the test and train patients
            test_masks = patient_id_masks[test_patient_inds]
            valid_masks = patient_id_masks[valid_patient_inds]
            train_masks = patient_id_masks[train_patient_inds]
            test_inds = test_masks.any(axis=0)
            valid_inds = valid_masks.any(axis=0)
            valid_test_inds = test_inds & valid_inds
            if sum(valid_inds) > sum(test_inds):
                test_inds = test_inds | valid_test_inds
                valid_inds = valid_inds & ~valid_test_inds
            else:
                valid_inds = valid_inds | valid_test_inds
                test_inds = test_inds & ~valid_test_inds
            train_inds = (~test_inds) & (~valid_inds)

        # use only negatives according to the neg_partition parameter, that is: divide the negatives into 5 parts and use only chunk no. neg_partition of it
        if neg_partition > 0 and not use_similar_negatives:
            neg_partition = neg_partition - 1
            num_of_negatives = len(neg_seqs)
            chunk_size = num_of_negatives // 5
            start_index = chunk_size * neg_partition
            end_index = start_index + chunk_size
            new_neg_seqs = neg_seqs[start_index:end_index]
            if len(positive_seqs) * neg_pos_ratio > len(new_neg_seqs):
                neg_seqs_to_add = int(len(positive_seqs) * neg_pos_ratio - len(new_neg_seqs))
                remaining_neg_seqs = np.array(list(set(neg_seqs) - set(new_neg_seqs)))
                # add randomly sequences that do not appear in new_neg_seqs but do appear in neg_seqs
                np.random.seed(neg_partition)
                replace = True if neg_seqs_to_add > len(remaining_neg_seqs) else False
                neg_seqs = np.concatenate(
                    (new_neg_seqs, np.random.choice(remaining_neg_seqs, size=neg_seqs_to_add, replace=replace)))
                np.random.seed(42)
            else:
                neg_seqs = new_neg_seqs

        # loss preprocessor
        self.losses_preprocessor = LossPreprocessor(df_bld, df_hlt, train_pos_seqs, train_patient_ids, test_patient_ids,
                                                    valid_patient_ids, filter_num_of_patients, ratio, dataset_type,
                                                    dist_loss_type, use_nneighbors_loss, self.cache_path)

        # set sequences as class attributes
        self.test_pos_seqs = test_pos_seqs
        self.test_neg_seqs = test_neg_seqs
        self.valid_pos_seqs = valid_pos_seqs
        self.valid_neg_seqs = valid_neg_seqs
        self.train_pos_seqs = train_pos_seqs
        self.train_neg_seqs = neg_seqs
        # set all sequences as class attributes
        self.positive_seqs = positive_seqs
        # set patient ids as class attributes
        self.test_patient_ids = test_patient_ids
        self.valid_patient_ids = valid_patient_ids
        self.train_patient_ids = train_patient_ids
        # set patient inds as class attributes
        self.test_patient_inds = test_patient_inds
        self.valid_patient_inds = valid_patient_inds
        self.train_patient_inds = train_patient_inds
        self.train_inds = train_inds
        # set patient masks as class attributes
        self.test_masks = test_masks
        self.valid_masks = valid_masks
        self.train_masks = train_masks
        self.patient_id_masks = patient_id_masks
        self.unique_patient_ids = unique_patient_ids
        # set dataframes as class attributes
        self.df_bld = df_bld
        self.df_hlt = df_hlt
        self.aaseq_to_ratio = self.losses_preprocessor.aaseq_to_ratio
        self.aaseq_to_distance = self.losses_preprocessor.aaseq_to_distance
        self.aaseq_to_nneighbors = self.losses_preprocessor.aaseq_to_nneighbors

    def use_similar_negatives_handler(self, neg_seqs, train_pos_seqs, neg_pos_ratio, neg_partition):
        similar_neg_cache_path = os.path.join(self.cache_path, '/similar_negatives')
        os.makedirs(similar_neg_cache_path, exist_ok=True)
        # Check if the file already exists
        similar_neg_seqs_path = os.path.join(similar_neg_cache_path, "similar_negatives.npy")
        similar_neg_min_dists_path = os.path.join(similar_neg_cache_path, "similar_neg_min_dists.npy")
        if os.path.exists(similar_neg_seqs_path) and os.path.exists(similar_neg_min_dists_path):
            # Load the file
            neg_seqs = np.load(similar_neg_seqs_path, allow_pickle=True)
            min_dist = np.load(similar_neg_min_dists_path, allow_pickle=True)
        else:
            # Calculate the similar negatives
            def batched_min_dist(neg_seqs, train_pos_seqs, score_func, num_batches=4):
                batch_size = len(neg_seqs) // num_batches
                min_dists = []

                for i in range(num_batches):
                    start = i * batch_size
                    end = (i + 1) * batch_size if i < num_batches - 1 else len(neg_seqs)

                    batch = neg_seqs[start:end]
                    pwc_mat_batch = pairwise_scores(batch, train_pos_seqs, score=score_func)
                    min_dists_batch = np.min(pwc_mat_batch, axis=1)
                    min_dists.append(min_dists_batch)

                return np.concatenate(min_dists)

            # get the minimum distance
            min_dist = batched_min_dist(neg_seqs, train_pos_seqs, levenshtein_dist_non_bin, num_batches=16)

            # save min_dist to file and corresponding neg_seqs to file (under base_path = 'cache/ms'):
            np.save(similar_neg_seqs_path, neg_seqs)
            np.save(similar_neg_min_dists_path, min_dist)

            # Plot the histogram of min_dist
            num_bins = len(np.unique(min_dist))
            # Create the histogram
            counts, bins, patches = plt.hist(min_dist, bins=np.arange(1, num_bins + 2) - 0.5, edgecolor='black')
            # Add counts above bars
            for count, patch in zip(counts, patches):
                plt.text(patch.get_x() + patch.get_width() / 2, count + 0.5, int(count), ha='center', va='bottom',
                         fontsize=10)
            plt.xticks(range(1, num_bins + 1))  # Set x-ticks from 1 to 8
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.title('Histogram of min_dist')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.show()

            # Save the histogram
            plt.savefig(os.path.join(similar_neg_cache_path, 'histogram_min_dist.png'))

        if neg_partition > 0:
            np.random.seed(neg_partition)

        # In total take len(train_pos_seqs) * neg_pos_ratio sequences from neg_seqs in the following manner:
        # 1. Take 30% sequences of dist 1
        # 2. Take sequences from dist 2 at random until we reach 65% of our capacity
        # 3. Take sequences from dist 3 at random until we reach 80% of our capacity
        # 4. Take sequences from dist 4 or more at random until we reach 100% of our capacity

        # Calculate how many negatives we want
        total_needed = int(len(train_pos_seqs) * neg_pos_ratio)

        # Step 1: Take 30% sequences with distance 1
        dist_1_mask = (min_dist == 1)
        dist_1_seqs = neg_seqs[dist_1_mask]
        dist_1_seqs = np.random.choice(dist_1_seqs, size=min(int(total_needed * 0.3), len(dist_1_seqs)), replace=False)

        sampled = list(dist_1_seqs)
        remaining_capacity = total_needed - len(sampled)
        if remaining_capacity <= 0:
            return np.array(sampled[:total_needed])

        # Helper function to sample from a distance category
        def sample_from_dist(dist_val, target_capacity):
            nonlocal sampled, remaining_capacity
            mask = (min_dist == dist_val)
            candidates = neg_seqs[mask]
            to_sample = min(target_capacity, len(candidates))
            if to_sample > 0:
                selected = np.random.choice(candidates, size=to_sample, replace=False)
                sampled.extend(selected)
                remaining_capacity -= to_sample

        # Step 2: Fill up to 65% with dist == 2
        sample_from_dist(2, int(total_needed * 0.65) - len(sampled))

        # Step 3: Fill up to 80% with dist == 3
        sample_from_dist(3, int(total_needed * 0.8) - len(sampled))

        # Step 4: Fill the rest with dist >= 4
        mask_4_or_more = (min_dist >= 4)
        candidates = neg_seqs[mask_4_or_more]
        to_sample = min(remaining_capacity, len(candidates))
        if to_sample > 0:
            selected = np.random.choice(candidates, size=to_sample, replace=False)
            sampled.extend(selected)

        neg_seqs = np.array(sampled[:total_needed])

        # shuffle the negative sequences
        np.random.shuffle(neg_seqs)

        return neg_seqs

    def get_aaseq_to_ratio_func(self):
        return self.aaseq_to_ratio

    def get_aaseq_to_distance_func(self):
        return self.aaseq_to_distance

    def get_aaseq_to_nneighbors_func(self):
        return self.aaseq_to_nneighbors

    def get_seqs(self):
        return self.train_pos_seqs, self.train_neg_seqs, self.valid_pos_seqs, self.valid_neg_seqs, self.test_pos_seqs, self.test_neg_seqs

    def get_dfs(self):
        return self.df_bld, self.df_hlt

    def get_patient_ids(self):
        return self.train_patient_ids, self.valid_patient_ids, self.test_patient_ids

    def get_patient_inds(self):
        return self.train_patient_inds, self.valid_patient_inds, self.test_patient_inds

    def get_masks(self):
        return self.train_masks, self.valid_masks, self.test_masks

    def get_all_usable_disease_data(self, disease='Multiple sclerosis', dataset_type=None, get_all=False):
        cache_dir = self.cache_dataframes_path
        os.makedirs(cache_dir, exist_ok=True)

        cache_path = None
        if dataset_type is not None:
            cache_path = os.path.join(cache_dir, f"{dataset_type}.pkl")
            if os.path.exists(cache_path):
                print(f"Loading cached data from {cache_path}")
                return pd.read_pickle(cache_path)

        studies = []
        if disease == 'Ankylosing spondylitis':
            study_ids = [STUDY_ID]
        elif disease == 'Hepatitis B virus':
            study_ids = [STUDY_ID2]
        elif disease == 'Rheumatoid arthritis':
            study_ids = [STUDY_ID3]
        elif disease == 'HIV':
            study_ids = [STUDY_ID4]
        elif disease == 'Multiple sclerosis':
            study_ids = [STUDY_ID8, STUDY_ID6, STUDY_ID7]
            # study_ids = [STUDY_ID6, STUDY_ID7, STUDY_ID8]  # TODO: Change back to this order!
        elif disease == 'CMV':  # Cytomegalovirus
            study_ids = [STUDY_ID9, STUDY_ID10, STUDY_ID11]
        else:
            raise ValueError(f"Invalid disease: {disease}")

        for study_id in study_ids:
            # df = self.load_study_df(study_id, disease)
            # studies.append(df)
            # continue

            study = Study(study_id)
            usable_samples = study._samples['usable']
            data_source = 'tcrdb2' if 'tcrdb2' in dataset_type else 'tcrdb'
            df = study.read_sample(usable_samples, condition=disease if not get_all else None,
                                   top_percent=self.top_percent, top_n_seqs=self.top_n_seqs, data_source=data_source)
            # if not get_all:
            #     df = df[df['condition'] == disease]
            # df['study_id'] = study_id  # TODO: There is a SettingWithCopyWarning here!
            studies.append(df)

        all_df = pd.concat(studies, ignore_index=True)
        # Save to cache if applicable
        if cache_path is not None:
            print(f"Saving data to cache at {cache_path}")
            all_df.to_pickle(cache_path)
        return all_df

    def get_all_usable_healthy_data(self, dataset_type='ms'):
        # Set up cache path
        cache_dir = self.cache_dataframes_healthy_path
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = None
        if dataset_type is not None:
            cache_path = os.path.join(cache_dir, f"{dataset_type}.pkl")
            if os.path.exists(cache_path):
                print(f"Loading cached healthy data from {cache_path}")
                return pd.read_pickle(cache_path)

        healthy_study_ids = [HEALTHY_STUDY_ID, HEALTHY_STUDY_ID2, HEALTHY_STUDY_ID3]
        if not 'no_healthy_ms' in dataset_type:
            healthy_study_ids += [HEALTHY_STUDY_ID4, HEALTHY_STUDY_ID5]
        if 'tcrdb2' in dataset_type:
            healthy_study_ids += [HEALTHY_STUDY_ID6, HEALTHY_STUDY_ID7, HEALTHY_STUDY_ID8, HEALTHY_STUDY_ID9]

        healthy_studies = []
        for study_id in healthy_study_ids:
            # df = self.load_study_df(study_id, disease)
            # healthy_studies.append(df)
            # continue

            study = Study(study_id)
            usable_samples = study._samples['usable']
            data_source = 'tcrdb2' if 'tcrdb2' in dataset_type else 'tcrdb'
            df = study.read_sample(usable_samples, condition='Healthy',
                                   top_percent=self.top_percent, top_n_seqs=self.top_n_seqs, data_source=data_source)
            # df = df[df['condition'] == 'Healthy']
            # df['study_id'] = study_id
            healthy_studies.append(df)
        df_concat = pd.concat(healthy_studies, ignore_index=True)
        df_concat = df_concat.dropna(subset=['AASeq'])

        # Save to cache
        if cache_path is not None:
            print(f"Saving healthy data to cache at {cache_path}")
            df_concat.to_pickle(cache_path)
        return df_concat

    def load_study_df(self, study_id, disease):
        # Note: cols that should always be in the returned df (in this order):
        # ['AASeq', 'cloneFraction', 'Vregion', 'Dregion', 'Jregion', 'RunId', 'patient_id', 'tissue', 'cell_type', 'condition', 'study_id']

        # check if there is a folder under TCRDB2_PATH named study_id:
        study_folder = os.path.join(TCRDB2_PATH, study_id)
        if not os.path.isdir(study_folder):
            assert False, f"Study folder not found: {study_folder}"

        csv_files = [f for f in os.listdir(study_folder) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {study_folder}")

        # # Get study metadata per patient:
        # metadata_path = os.path.join(study_folder, f"{study_id}.txt")
        # if not os.path.isfile(metadata_path):
        #     raise FileNotFoundError(f"Did not find {study_id}.txt in {study_folder}")

        # Read all .csv files from the dir and append to study_dfs
        study_dfs = []
        for filename in csv_files:
            file_path = os.path.join(study_folder, filename)
            df = pd.read_csv(file_path)
            df = self.tcrdb2_filtering(df, study_id, disease)
            study_dfs.append(df)

        return pd.concat(study_dfs, ignore_index=True)

    def tcrdb2_filtering(self, df, study_id, disease):
        df = df.drop(columns=['Unnamed: 0', 'cloneCount', 'Length', 'NNSeq'])  # will also drop 'Chain' later

        df['condition'] = disease
        df['study_id'] = study_id
        # TODO!!!!!
        #  1. ADD: 'patient_id', 'tissue', 'cell_type' (and remove the temporary solution...)

        df['patient_id'] = df['RunId'].unique()[0]
        df['tissue'] = 'Unknown'
        df['cell_type'] = 'Unknown'

        # Keep only beta chains
        df = df[df['Chain'] == 'TRB'].copy()
        df = df.drop(columns=['Chain'])

        # Keep only complete sequences with valid V, J, and in-frame CDR3
        df = df.dropna(subset=['AASeq', 'Vregion', 'Jregion'])

        # Keep only CDR3 sequences that start with C and end with F and don't contain stop codons (*)
        df = df[df['AASeq'].str.match(r'^C[^*]*F$')]

        # Normalize V and J region by removing alleles (e.g., TRBV7-9*01 → TRBV7-9)
        df['Vregion'] = df['Vregion'].str.extract(r'^(TRBV[\d\-]+)')
        df['Jregion'] = df['Jregion'].str.extract(r'^(TRBJ[\d\-]+)')

        # Group by AASeq and take the most frequent V and J gene
        df_grouped = (
            df.groupby('AASeq')
            .apply(lambda g: g.loc[g['cloneFraction'].idxmax()])
            .reset_index(drop=True)
        )

        # Filter by cloneFraction ratio threshold (> 0.00001%)
        df_grouped = df_grouped[df_grouped['cloneFraction'] > 0.00001]
        # df_grouped = df_grouped[df_grouped['cloneFraction'] > 0.0000001]  # 0.00001% = 0.0000001

        # Ensure column order and presence
        required_cols = ['AASeq', 'cloneFraction', 'Vregion', 'Dregion', 'Jregion',
                         'RunId', 'patient_id', 'tissue', 'cell_type', 'condition', 'study_id']
        for col in required_cols:
            if col not in df_grouped.columns:
                df_grouped[col] = np.nan

        df_grouped = df_grouped[required_cols]

        return df_grouped

    # This function loads the synapse dataframe of Mal-ID of only TCR and healthy samples for sure.
    def get_full_healthy_synapse_mal_id_dataframe(self, to_recalculate=False, get_all=False):
        # Defining constants
        synapse_metadata_file = os.path.join(self.synapse_db_path, "metadata.tsv")

        # Define the filename based on the get_all flag
        file_suffix = "_all" if get_all else "_healthy_only"
        df_filename = os.path.join(self.synapse_db_path, f"synapse_mal_id_dataframe{file_suffix}.pkl")
        # Check if the DataFrame is already saved
        if not to_recalculate:
            if os.path.exists(df_filename):
                # Load the DataFrame from file
                df = pd.read_pickle(df_filename)
                return df

        # Reading metadata
        synapse_metadata = pd.read_csv(synapse_metadata_file, sep='\t')

        # Reading and interpreting data files
        synapse_datafiles = [x for x in os.listdir(self.synapse_db_path) if x.endswith(".bz2")]
        # lupus_ids
        healthy_samples = []
        for datafile in tqdm(synapse_datafiles, desc="Processing data files", total=len(synapse_datafiles)):
            datafile_id = datafile.split("_")[-1][:-4]
            datafile_path = os.path.join(self.synapse_db_path, datafile)

            # Reading metadata and data
            metadata = synapse_metadata[synapse_metadata['participant_label'] == datafile_id]
            condition = metadata['disease'].values[0]
            if get_all or 'Healthy' in condition:
                data = pd.read_csv(datafile_path, sep='\t', compression='bz2')
                data = data.loc[:, ['v_segment', 'd_segment', 'j_segment', 'run_id', 'cdr3_seq_aa_q', 'participant_label', 'specimen_tissue']]
                data['cdr3_seq_aa_q'] = data['cdr3_seq_aa_q'].str.replace(' ', '')
                # add the condition to the metadata
                if 'Healthy' in condition:
                    condition = 'Healthy'
                data['condition'] = condition
                healthy_samples.append(data)

        # need to make a df with: AASeq, patient_id, tissue, cell_type
        df = pd.concat(healthy_samples, ignore_index=True)
        df = df.dropna(subset=['cdr3_seq_aa_q'])
        # rename columns
        df = df.rename(
            columns={'v_segment' : 'Vregion', 'd_segment' : 'Dregion', 'j_segment' : 'Jregion', 'run_id' : 'RunId',
                     'cdr3_seq_aa_q': 'AASeq', 'participant_label': 'patient_id', 'specimen_tissue': 'tissue'})
        df = df[~df['AASeq'].str.contains('[^ACDEFGHIKLMNPQRSTVWY]', regex=True)]
        df['AASeq'] = 'C' + df['AASeq'] + 'F'  # This is needed: because with this line we get a large set of seqs that also appear in the ms blood df

        # Step 1: Count how many times each AASeq appears per patient → cloneCount
        df['cloneCount'] = df.groupby(['patient_id', 'AASeq'])['AASeq'].transform('count')
        # Step 2: Drop duplicates so you have one row per unique (patient_id, AASeq)
        df_unique = df.drop_duplicates(subset=['patient_id', 'AASeq']).copy()
        # Step 3: Compute cloneFraction per patient
        df_unique['cloneFraction'] = df_unique.groupby('patient_id')['cloneCount'].transform(lambda x: x / x.sum())
        # Step 4: Drop the cloneCount column
        df_unique = df_unique.drop(columns=['cloneCount'])

        # Normalize V and J regions
        df_unique['Vregion'] = df_unique['Vregion'].str.extract(r'^(TRBV[\d\-]+)')
        df_unique.loc[df_unique['Dregion'].apply(lambda x: isinstance(x, str)), 'Dregion'] = \
            df_unique.loc[df_unique['Dregion'].apply(lambda x: isinstance(x, str)), 'Dregion'].str.extract(r'^(TRBD[\d\-]+)')
        df_unique['Jregion'] = df_unique['Jregion'].str.extract(r'^(TRBJ[\d\-]+)')

        df_unique['study_id'] = 'synapse_mal_id'
        required_cols = ['AASeq', 'cloneFraction', 'Vregion', 'Dregion', 'Jregion',
                         'RunId', 'patient_id', 'tissue', 'cell_type', 'condition', 'study_id']
        for col in required_cols:
            if col not in df_unique.columns:
                df_unique[col] = np.nan
        df_unique = df_unique[required_cols]

        # Save the DataFrame for future use
        df_unique.to_pickle(df_filename)

        return df_unique

    def process_in_batches(self, df, all_common_seqs, batch_size, lev_dist_accept):
        all_common_seqs = list(all_common_seqs)
        # Split all_common_seqs into batches
        all_common_seqs_batches = [all_common_seqs[i:i + batch_size] for i in range(0, len(all_common_seqs), batch_size)]

        valid_seqs_set = set()  # Use a set to store valid sequences across batches

        for batch in tqdm(all_common_seqs_batches):
            df_new = df.copy()

            # Flag 'valid' for sequences in the current batch
            df_new['patient_id'] = df_new['AASeq'].apply(lambda x: 'valid' if x in batch else 'all')

            # Separate valid and all sequences
            valid_df = df_new[df_new['patient_id'] == 'valid'].drop_duplicates(subset='AASeq')
            all_df = df_new[df_new['patient_id'] == 'all'].drop_duplicates(subset='AASeq')

            # Concatenate and reset index
            df_combined = pd.concat([valid_df, all_df]).reset_index(drop=True)

            # Apply the analysis function to the valid sequences
            valid_seqs_batch = helper_function_common_aaseq_analysis(df_combined, lev_dist_accept, only_valid=True, verbose=False)

            # Accumulate valid sequences from this batch (as a set)
            valid_seqs_set.update(valid_seqs_batch)

        # The result is a set of valid sequences
        return valid_seqs_set

    def common_aaseq_analysis(self, df, num_of_patients, lev_dist_accept=0, mode=1, max_combinations=50000, std_val='percent_of_total'):

        # Select the unique patients
        unique_patients = df['patient_id'].unique()

        if len(unique_patients) < num_of_patients:
            raise ValueError("Number of patients in the dataframe is less than num_of_patients")

        # Create a dictionary mapping each patient_id to their set of AASeq
        patient_sequences = {pid: set(df[df['patient_id'] == pid]['AASeq']) for pid in unique_patients}

        results = []
        if mode == 1:
            # Mode 1: All combinations
            patient_combinations = list(combinations(unique_patients, num_of_patients))
        elif mode == 2:
            # Mode 2: Always include the first patient
            patient_combinations = [tuple([unique_patients[0]] + list(comb)) for comb in
                                    combinations(unique_patients[1:], num_of_patients - 1)]
        else:
            raise ValueError("Mode must be 1 (All combinations) or 2 (Always include the first patient).")

        # Pick max_combinations at random from patient_combinations, if it is too large
        if len(patient_combinations) > max_combinations:
            patient_combinations = random.sample(patient_combinations, max_combinations)
        print(f"Number of patient combinations: {len(patient_combinations)}")

        # TODO: Check if multiprocessing is better than just running normally!
        # results, percent_of_total_values = run_multiprocess_analysis(
        #     patient_combinations, patient_sequences, df, lev_dist_accept,
        #     helper_function_common_aaseq_analysis, num_processes=min(mp.cpu_count(), 8)  # Limit to 8 processes
        # )
        values_to_std = []
        for combination in patient_combinations:
            selected_sequences = [patient_sequences[pid] for pid in combination]

            if lev_dist_accept >= 1:
                temp_df = df[df['patient_id'].isin(combination)]
                masks = helper_function_common_aaseq_analysis(temp_df, lev_dist_accept)
                # TODO: This will always increase when we look at more patients... this isnt the calculation that we want here
                num_common = np.sum(np.any(masks == 1, axis=0))
                print("Using lev_dist_accept >= 1, there might be a problem in the implementation here!!")
            else:
                common_sequences = set.intersection(*selected_sequences)
                num_common = len(common_sequences)
            total_sequences = sum(len(seqs) for seqs in selected_sequences)
            min_sequences = min(len(seqs) for seqs in selected_sequences)
            max_sequences = max(len(seqs) for seqs in selected_sequences)
            percent_of_total = (num_common / total_sequences) * 100 if total_sequences > 0 else 0
            percent_of_mean = [num_common / len(seqs) for seqs in selected_sequences]
            percent_of_mean = (sum(percent_of_mean) / len(percent_of_mean)) * 100 if percent_of_mean else 0
            percent_of_min = (num_common / min_sequences) * 100 if min_sequences > 0 else 0
            percent_of_max = (num_common / max_sequences) * 100 if max_sequences > 0 else 0
            results.append({
                'num_total_seqs': total_sequences,
                'num_common': num_common,
                'percent_of_total': percent_of_total,
                'percent_of_mean': percent_of_mean,
                'percent_of_min': percent_of_min,
                'percent_of_max': percent_of_max
            })
            values_to_std.append(results[-1][std_val])  # Collect values for std calculation

        # Calculate means
        mean_results = {
            'num_total_seqs': sum(r['num_total_seqs'] for r in results) / len(results),
            'num_common': sum(r['num_common'] for r in results) / len(results),
            'percent_of_total': sum(r['percent_of_total'] for r in results) / len(results),
            'percent_of_mean': sum(r['percent_of_mean'] for r in results) / len(results),
            'percent_of_min': sum(r['percent_of_min'] for r in results) / len(results),
            'percent_of_max': sum(r['percent_of_max'] for r in results) / len(results)
        }

        # Calculate std for the given value
        std_values = np.std(values_to_std)

        return mean_results, std_values

    def get_ms_extra_bld_dataframe(self, top_percent=None, top_n_seqs=None, df_bld=None):
        extra_ms_path = '../data/db/tcrdb/special2'
        extra_ms_files = [x for x in os.listdir(extra_ms_path) if x.endswith('Pre.csv')]
        extra_ms_dfs = []

        def print_names_and_tags():
            extra_ms_path = '../data/db/tcrdb/special'
            file_path = os.path.join(extra_ms_path, 'information/names_and_tags.txt')
            output_path = os.path.join(extra_ms_path, 'information/parsed_names_and_tags/pairs.pickle')

            pairs = []

            with open(file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]  # remove empty lines

            def parse_tags(tag_line):
                tag_dict = {}
                for item in tag_line.split(','):
                    if ':' in item:
                        key, value = item.split(':', 1)  # only split on first ':'
                        tag_dict[key.strip()] = value.strip()
                return tag_dict

            # group lines into pairs and parse tags as dicts
            for i in range(0, len(lines), 2):
                tsv = lines[i]
                tag_line = lines[i + 1] if i + 1 < len(lines) else ''
                tag_dict = parse_tags(tag_line)
                pairs.append((tsv, tag_dict))

            # Save to pickle
            with open(output_path, 'wb') as f:
                pickle.dump(pairs, f)

            # print the pairs
            for tsv, tag_dict in pairs:
                print(f"Filename: {tsv}")
                print("Tags:")
                for k, v in tag_dict.items():
                    print(f"  {k}: {v}")
                print()

        # print_names_and_tags()

        # Ensure every sequence starts with 'C' and ends with 'F'
        def enforce_start_end(seq):
            if not seq.startswith('C'):
                seq = 'C' + seq
            if not seq.endswith('F'):
                seq = seq + 'F'
            return seq

        for file_name in tqdm(extra_ms_files):
            file_path = os.path.join(extra_ms_path, file_name)
            df = pd.read_csv(file_path, sep=',', low_memory=False)
            # df = pd.read_csv(file_path, sep='\t', low_memory=False)  # FOR .tsv (and not for csv!)
            # Possible cols: ['sample_name' 'species' 'locus' 'product_subtype' 'kit_pool' 'sku', 'test_name' 'sample_catalog_tags' 'sample_rich_tags', 'sample_rich_tags_json' 'hla_class_i' 'hla_class_ii' 'kit_control', 'total_templates' 'productive_templates' 'outofframe_templates', 'stop_templates' 'dj_templates' 'total_rearrangements', 'productive_rearrangements' 'outofframe_rearrangements', 'stop_rearrangements' 'dj_rearrangements' 'total_reads', 'total_productive_reads' 'total_outofframe_reads' 'total_stop_reads', 'total_dj_reads' 'productive_simpson_clonality' 'productive_clonality', 'productive_entropy' 'sample_simpson_clonality' 'sample_clonality', 'sample_entropy' 'sample_amount_ng' 'sample_cells_mass_estimate', 'fraction_productive_of_cells_mass_estimate' 'sample_cells', 'fraction_productive_of_cells' 'max_productive_frequency' 'max_frequency', 'counting_method' 'primer_set' 'sequence_result_status' 'release_date', 'upload_date' 'sample_tags' 'fraction_productive' 'order_name' 'kit_id', 'total_t_cells' 'total_templates_agg' 'rearrangement' 'amino_acid', 'frame_type' 'rearrangement_type' 'templates' 'seq_reads' 'frequency', 'productive_frequency' 'cdr3_length' 'v_family' 'v_gene' 'v_allele', 'd_family' 'd_gene' 'd_allele' 'j_family' 'j_gene' 'j_allele', 'v_deletions' 'd5_deletions' 'd3_deletions' 'j_deletions' 'n2_insertions', 'n1_insertions' 'v_index' 'n1_index' 'n2_index' 'd_index' 'j_index', 'v_family_ties' 'v_gene_ties' 'v_allele_ties' 'd_family_ties', 'd_gene_ties' 'd_allele_ties' 'j_family_ties' 'j_gene_ties', 'j_allele_ties' 'sequence_tags' 'v_shm_count' 'v_shm_indexes' 'antibody', 'bio_identity' 'rearrangement_trunc' 'v_resolved' 'd_resolved', 'j_resolved' 'extended_rearrangement' 'cdr1_rearrangement', 'cdr1_amino_acid' 'cdr1_start_index' 'cdr1_rearrangement_length', 'cdr2_rearrangement' 'cdr2_amino_acid' 'cdr2_start_index', 'cdr2_rearrangement_length' 'cdr3_rearrangement' 'cdr3_amino_acid', 'cdr3_start_index' 'cdr3_rearrangement_length' 'chosen_v_family', 'chosen_v_gene' 'chosen_v_allele' 'chosen_j_family' 'chosen_j_gene', 'chosen_j_allele']
            # what we need: ['AASeq', 'cloneFraction', 'Vregion', 'Dregion', 'Jregion', 'RunId', 'patient_id', 'tissue', 'cell_type', 'condition', 'study_id']
            # Important cols:
            # 1. 'AASeq' is 'cdr3_amino_acid' (remove rows with sequences that have non-standard amino acids)
            # 2. 'cloneFraction' is 'productive_frequency' (fill nans with 0)
            # 3. 'Vregion' is 'v_gene' (fill nans with 'None')
            # 4. 'Dregion' is 'd_gene' (fill nans with 'None')
            # 5. 'Jregion' is 'j_gene' (fill nans with 'None')
            # 6. 'RunId' is 'sample_name' (without the '_Pre' suffix)
            # 7. 'patient_id' is 'sample_name' (split by '_' and take first element)
            # 8. 'tissue' is 'locus'
            # 9. 'cell_type' is 'sample_name' (split by '_' and take second element)
            # 10. 'condition' is 'Multiple Sclerosis'
            # 11. 'study_id' is 'immunoSEQ33'

            df_filtered = df[['AASeq', 'cloneFraction', 'Vregion', 'Dregion', 'Jregion']]
            df_filtered = df_filtered[df_filtered['AASeq'].str.contains('^C[ACDEFGHIKLMNPQRSTVWY]*F$', na=False)]
            df_filtered.loc[:, 'RunId'] = file_name.split('_')[2]
            df_filtered.loc[:, 'patient_id'] = file_name.split('_')[0]
            df_filtered.loc[:, 'tissue'] = 'Blood'
            df_filtered.loc[:, 'cell_type'] = file_name.split('_')[1] if file_name.split('_')[1] in ['CD4', 'CD8'] else 'Unknown'
            df_filtered.loc[:, 'condition'] = 'Multiple Sclerosis'
            df_filtered.loc[:, 'study_id'] = 'immunoSEQ33'

            # df_filtered = df.rename(columns={
            #     'cdr3_amino_acid': 'AASeq',
            #     'productive_frequency': 'cloneFraction',
            #     'v_gene': 'Vregion',
            #     'd_gene': 'Dregion',
            #     'j_gene': 'Jregion',
            #     'sample_name': 'RunId',
            #     'locus': 'tissue',
            # })
            # # drop rows with NaN in 'AASeq' or 'cloneFraction'
            # df_filtered = df_filtered.dropna(subset=['AASeq'])
            # df_filtered = df_filtered[df_filtered['AASeq'].str.contains('^C[ACDEFGHIKLMNPQRSTVWY]*F$', na=False)]
            # # make sure that every sequence in AASeq starts with 'C' and ends with 'F'
            # # df_filtered['AASeq'] = df_filtered['AASeq'].apply(enforce_start_end)  # Note: This will add 'C' and 'F' to the sequence, but we want to just remove it instead.
            # df_filtered['cloneFraction'] = df_filtered['cloneFraction'].fillna(0.0)
            # df_filtered['Vregion'] = df_filtered['Vregion'].fillna('unresolved')
            # df_filtered['Dregion'] = df_filtered['Dregion'].fillna('unresolved')
            # df_filtered['Jregion'] = df_filtered['Jregion'].fillna('unresolved')
            # df_filtered['RunId'] = df_filtered['RunId'].str.replace('_Pre', '')
            # df_filtered['RunId'] = df_filtered['RunId'].str.replace('_', '-')
            # df_filtered['patient_id'] = df_filtered['RunId'].apply(lambda x: x.split('-')[0])
            # df_filtered['tissue'] = df_filtered['tissue'].str.replace(' ', '-')
            # df_filtered['tissue'] = df_filtered['tissue'].replace({'TCRB': 'PBMC'})
            # df_filtered['cell_type'] = df_filtered['RunId'].apply(lambda x: x.split('-')[1] if len(x.split('-')) > 1 else 'Unknown')
            # df_filtered['condition'] = 'Multiple Sclerosis'
            # df_filtered['study_id'] = 'immunoSEQ33'
            # df_filtered = df_filtered[['AASeq', 'cloneFraction', 'Vregion', 'Dregion', 'Jregion', 'RunId', 'patient_id',
            #                            'tissue', 'cell_type', 'condition', 'study_id']]
            extra_ms_dfs.append(df_filtered)

        # concatenate all dataframes
        extra_ms_df = pd.concat(extra_ms_dfs, ignore_index=True)

        def print_groups_info():
            overlap_sizes = []
            grouped = extra_ms_df.groupby('patient_id')
            for patient_id, group in grouped:
                cd4_seqs = set(group.loc[group['cell_type'].str.contains('CD4', na=False), 'AASeq'])
                cd8_seqs = set(group.loc[group['cell_type'].str.contains('CD8', na=False), 'AASeq'])
                pbmc_seqs = set(group.loc[group['cell_type'].str.contains('PBMC', na=False), 'AASeq'])

                combined_cd4_cd8 = cd4_seqs.union(cd8_seqs)
                overlap_with_pbmc = combined_cd4_cd8.intersection(pbmc_seqs)

                print(f"Patient ID: {patient_id}")
                print(f"  CD4 unique AASeqs: {len(cd4_seqs)}")
                print(f"  CD8 unique AASeqs: {len(cd8_seqs)}")
                print(f"  PBMC unique AASeqs: {len(pbmc_seqs)}")
                print(f"  CD4+CD8 combined unique AASeqs: {len(combined_cd4_cd8)}")
                print(f"  Overlap with PBMC: {len(overlap_with_pbmc)}\n")
                overlap_sizes.append(len(overlap_with_pbmc))
            print(f"Overlap Mean: {np.mean(overlap_sizes):.3f}")

        # print number of AASeqs per patient_id (before filtering/sampling)
        # aa_counts = extra_ms_df.groupby('patient_id')['AASeq'].count()
        # print("AASeq counts per patient_id:\n", aa_counts.sort_values(ascending=False))

        # limit to at most 20,000 unique AASeqs per patient
        # def sample_unique_seqs(group):
        #     group_unique = group.drop_duplicates(subset='AASeq')
        #     if len(group_unique) > 20000:
        #         group_unique = group_unique.sample(n=20000, random_state=42)
        #     return group_unique
        # extra_ms_df = extra_ms_df.groupby('patient_id', group_keys=False).apply(sample_unique_seqs)

        # remove all samples with 'cell_type' of PBMC
        extra_ms_df = extra_ms_df.sort_values('cloneFraction', ascending=False)
        extra_ms_df = extra_ms_df.drop_duplicates(subset=['patient_id', 'AASeq'], keep='first')

        # extra_ms_df = extra_ms_df[extra_ms_df['cell_type'] != 'Unknown']
        # extra_ms_df = extra_ms_df.drop_duplicates(subset=['patient_id', 'AASeq'])

        extra_ms_df = Study.do_tcrdb2_threshold_filtering(extra_ms_df, top_percent, top_n_seqs)

        return extra_ms_df

    def plot_per_patient_histogram(self, dfs, labels, colors, disease, to_add_top_percent=False, to_add_bar_vals=False):
        bar_vals = []
        # Get the number of unique AASeqs per patient
        for df in dfs:
            counts_per_patient = df.groupby("patient_id")["AASeq"].nunique()
            bar_vals.append(sorted(counts_per_patient))
        # Plotting the bar graph
        plt.figure(figsize=(12, 6))
        for bar_val, label, color in zip(bar_vals, labels, colors):
            plt.bar(np.arange(len(bar_val)), bar_val, label=label, alpha=0.7, color=color)
        if to_add_bar_vals:
            for i, val in enumerate(bar_vals[0]):
                plt.text(i, val + 50, str(val), rotation=-45, ha='center', va='bottom', fontsize=8)
        plt.xlabel('Patient ID')
        plt.ylabel('Number of Unique AASeqs')
        title = f'Unique AASeqs per Patient for {disease} and Healthy'
        if to_add_top_percent:
            title += f" - Top {int(self.top_percent)}%"
        plt.title(title)
        plt.xticks(rotation=90)
        plt.legend()
        plt.show()
