import os
import pickle
from cProfile import label

import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
from tqdm import tqdm
from Curation import Study
from utils import pairwise_scores, levenshtein_dist, levenshtein_dist_non_bin
import hashlib


STUDY_ID = 'PRJNA393498'  # Ankylosing Spondylitis study
STUDY_ID2 = 'immunoSEQ47'  # Hepatitis B virus study
STUDY_ID3 = 'immunoSEQ77'  # Rheumatoid arthritis study (plus healthy)
STUDY_ID4 = 'PRJNA258001'  # HIV study (plus healthy)
STUDY_ID5 = 'PRJNA390125'  # Only healthy study
STUDY_ID6 = 'PRJNA495603'  # Multiple sclerosis study (plus healthy)
STUDY_ID7 = 'PRJNA579190'  #  Multiple sclerosis study (plus healthy)
STUDY_ID8 = 'PRJNA280417'  #  Multiple sclerosis study
STUDY_ID9 = 'PRJNA427746'  #  Cytomegalovirus (plus healthy)
STUDY_ID10 = 'PRJNA318421'  #  Cytomegalovirus
STUDY_ID11 = 'PRJNA473147'  #  Cytomegalovirus
HEALTHY_STUDY_ID = STUDY_ID3  # ONLY CD8
HEALTHY_STUDY_ID2 = STUDY_ID4  # Both CD8 and CD4
HEALTHY_STUDY_ID3 = STUDY_ID5  # Larger both CD8 and CD4 (But fewer patients!)
HEALTHY_STUDY_ID4 = STUDY_ID6  # Other healthy study
HEALTHY_STUDY_ID5 = STUDY_ID7  # Other healthy study
STUDIES = [STUDY_ID, STUDY_ID2, STUDY_ID3, STUDY_ID4, STUDY_ID5, STUDY_ID6, STUDY_ID7]


class DatasetLoader:
    def __init__(self, dataset_type: str, unique_patient_ids=None, get_only_unique_patient_ids=False, k_fold=0,
                 dist_loss_type='none', neg_partition=0, use_similar_negatives=False, neg_pos_ratio=10,
                 filter_num_of_patients=3, filter_to_inflate=False, remove_seqs_by_len=False, verbose=True):
        self.dataset_type = dataset_type

        disease = 'Multiple sclerosis'
        df = self.get_all_usable_disease_data(disease=disease)
        df_h = self.get_all_usable_healthy_data()

        # Choosing cell type
        cell_type = ['DC8', 'CD4', 'ALL'][2]
        if cell_type != 'ALL' and dataset_type == 'ms':
            # reading blood samples
            df_bld = df[df['cell_type'] == cell_type]
            # reading healthy study:
            df_hlt = df_h[df_h['cell_type'] == cell_type]
        else:
            if dataset_type == 'ms':
                df_bld, df_hlt = df, df_h
            elif dataset_type == 'article':
                df_article = self.get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=True)
                df_bld = df_article[df_article["condition"] == "T1D"]  # condition options: ['HIV' 'Healthy' 'T1D' 'Lupus' 'Covid19']
                df_hlt = df_article[df_article["condition"] == "Healthy"]
                # TODO: This might be problematic: adding the other healthy dataset to the article healthy dataset!
                #  This can cause problems because the healthy people from a different study might be too different from healthy people from this dataset.
                #  So it might be too easy for the model to separate between them.
                df_hlt = pd.concat([df_hlt, df_h[['AASeq', 'patient_id', 'tissue', 'condition']]], axis=0, ignore_index=True)
            elif dataset_type == 'cmv':
                df_cmv = self.get_all_usable_disease_data(disease='CMV', get_all=True)
                df_bld = df_cmv[df_cmv["condition"] == "CMV"]
                filtered_patient_ids = [x[0] for x in df_bld.groupby("patient_id")["AASeq"] if len(x[1]) >= 2000]  # this leaves 25 patients
                df_bld = df_bld[df_bld["patient_id"].isin(filtered_patient_ids)]
                df_hlt = df_cmv[df_cmv["condition"] == "Healthy"]
                # df_hlt = pd.concat([df_hlt, df_h], axis=0, ignore_index=True)  # According to Dina, it might be problematic to add healthy from different studies
            elif dataset_type == 'article2':
                df_bld = self.get_full_article2_dataframe()
                df_hlt = df_h
            elif dataset_type == 'article_sle':
                df_article = self.get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=True)
                df_bld = df_article[df_article["condition"] == "Lupus"]  # condition options: ['HIV' 'Healthy' 'T1D' 'Lupus' 'Covid19']
                df_hlt = df_article[df_article["condition"] == "Healthy"]
            elif dataset_type == 'ms_plus_article2_ms':
                df_bld_article2 = self.get_full_article2_dataframe()
                df_bld = pd.concat([df, df_bld_article2], axis=0, ignore_index=True)
                df_hlt = df_h
            else:
                raise ValueError("Invalid dataset type")

        if get_only_unique_patient_ids:
            self.df_bld, self.df_hlt = df_bld, df_hlt
            return
        # TODO: This code checks the intersection of healthy and disease samples with the article
        # healthy_unique = set(df_hlt['AASeq'].unique())
        # df_article = self.get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=True)
        # article_unique = set(df_article['AASeq'].unique())
        # print(f"Number of sequences in the article: \t{len(df_article['AASeq'])}")
        # print(f"Number of unique sequences in the article: {len(article_unique)}")
        # print(f"Number of sequences in the intersection of healthy and article: {len(healthy_unique & article_unique)}")
        # print(f"Number of sequences in the intersection of disease and article: {len(set(df_bld['AASeq']) & article_unique)}")
        # exit(0)

        # pick index of 8 unique patients from unique_patient_ids as test patients and the rest as train patients
        if dataset_type == 'cmv' or dataset_type == 'article_sle':
            num_test_patients = 4
        else:
            num_test_patients = 8
        if unique_patient_ids is None:
            unique_patient_ids = df_bld["patient_id"].unique()
            unique_patient_ids = np.random.permutation(unique_patient_ids)
        test_patient_ids = unique_patient_ids[:num_test_patients // 2]
        valid_patient_ids = unique_patient_ids[num_test_patients // 2:num_test_patients]
        train_patient_ids = unique_patient_ids[num_test_patients:]
        # translate back to the inds according to unique_patient_ids
        test_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in test_patient_ids])
        valid_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in valid_patient_ids])
        train_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in train_patient_ids])

        # Check that each sequences in the dataset starts with 'C' and ends with 'F'! Otherwise, raise an error
        for seq in df_bld['AASeq'].unique().tolist() + df_hlt['AASeq'].unique().tolist():
            if not (seq.startswith('C') and seq.endswith('F')):
                raise ValueError(
                    f"Sequence {seq} does not start with 'C' and end with 'F'! (Working with dataset {dataset_type})")

        if k_fold > 0:
            name_metadata = f"_fold_{k_fold}"
        else:
            name_metadata = ""
        if dataset_type == 'article_sle':
            num_of_patients = 2
        else:
            num_of_patients = filter_num_of_patients

        # # TODO: DELETE:
        # temp_df_h = df_hlt
        # grouped_h = temp_df_h.groupby('patient_id')['AASeq'].unique()
        # aa_seq_counter_h = Counter()
        # for aa_seqs in grouped_h:
        #     aa_seq_counter_h.update(aa_seqs)
        # valid_aa_seqs_h = set({aa_seq for aa_seq, count in aa_seq_counter_h.items() if count >= 2})
        # common_seqs_healthy_and_disease = []
        # total_num_of_positives = []
        # for i in range(2, 17):
        #     num_of_patients = i
        #     train_pos_seqs, _ = self.calculate_pos_neg_sequences(df_bld, df_hlt, "train" + name_metadata,
        #                                                          train_patient_ids,
        #                                                          dataset_type, cell_type,
        #                                                          num_of_patients=num_of_patients,
        #                                                          filter_to_inflate=filter_to_inflate)
        #     # Calculate positive valid sequences
        #     train_and_valid_ids = np.concatenate((train_patient_ids, valid_patient_ids))
        #     valid_pos_seqs, _ = self.calculate_pos_neg_sequences(df_bld, df_hlt, "valid" + name_metadata,
        #                                                          train_and_valid_ids,
        #                                                          dataset_type, cell_type,
        #                                                          num_of_patients=num_of_patients,
        #                                                          filter_to_inflate=filter_to_inflate)
        #     valid_bld_seqs = df_bld[df_bld['patient_id'].isin(valid_patient_ids)]["AASeq"].unique()
        #     valid_pos_seqs = np.array(list(set(valid_pos_seqs) & set(valid_bld_seqs)))
        #     # Calculate positive test sequences
        #     train_and_test_ids = np.concatenate((train_patient_ids, test_patient_ids))
        #     test_pos_seqs, _ = self.calculate_pos_neg_sequences(df_bld, df_hlt, "test" + name_metadata,
        #                                                         train_and_test_ids,
        #                                                         dataset_type, cell_type,
        #                                                         num_of_patients=num_of_patients,
        #                                                         filter_to_inflate=filter_to_inflate)
        #     test_bld_seqs = df_bld[df_bld['patient_id'].isin(test_patient_ids)]["AASeq"].unique()
        #     test_pos_seqs = np.array(list(set(test_pos_seqs) & set(test_bld_seqs)))
        #
        #     # make sure that there is no intersection between train, valid and test sequences
        #     train_pos_seqs = np.array(list(set(train_pos_seqs) - set(np.concatenate((valid_pos_seqs, test_pos_seqs)))))
        #     if len(set(valid_pos_seqs) & set(test_pos_seqs)) > 0:
        #         if len(valid_pos_seqs) > len(test_pos_seqs):
        #             valid_pos_seqs = np.array(list(set(valid_pos_seqs) - set(test_pos_seqs)))
        #         else:
        #             test_pos_seqs = np.array(list(set(test_pos_seqs) - set(valid_pos_seqs)))
        #
        #     positive_seqs = np.concatenate((train_pos_seqs, valid_pos_seqs, test_pos_seqs))
        #
        #     common_seqs_healthy_and_disease.append(len(set(positive_seqs).intersection(valid_aa_seqs_h)))
        #     total_num_of_positives.append(len(positive_seqs))
        # # Plot common_seqs_healthy_and_disease in a simple plot
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(10, 6))
        # plt.plot(range(3, 17), common_seqs_healthy_and_disease[1:], label='Common Sequences of Healthy and Positive Disease')
        # plt.plot(range(3, 17), total_num_of_positives[1:], label='Total Positives')
        # plt.xticks(range(3, 17))
        # plt.xlabel("Number of Patients")
        # plt.ylabel("Number of Common Sequences with Healthy")
        # plt.title("Common Sequences with Healthy vs Number of Patients")
        # plt.legend()
        # plt.grid()
        # plt.show()

        train_pos_seqs, _ = self.calculate_pos_neg_sequences(df_bld, df_hlt, "train" + name_metadata, train_patient_ids, dataset_type, cell_type, num_of_patients=num_of_patients, filter_to_inflate=filter_to_inflate, verbose=verbose)
        # Calculate positive valid sequences
        train_and_valid_ids = np.concatenate((train_patient_ids, valid_patient_ids))
        valid_pos_seqs, _ = self.calculate_pos_neg_sequences(df_bld, df_hlt, "valid" + name_metadata, train_and_valid_ids, dataset_type, cell_type, num_of_patients=num_of_patients, filter_to_inflate=filter_to_inflate, verbose=verbose)
        valid_bld_seqs = df_bld[df_bld['patient_id'].isin(valid_patient_ids)]["AASeq"].unique()
        valid_pos_seqs = np.array(list(set(valid_pos_seqs) & set(valid_bld_seqs)))
        # Calculate positive test sequences
        train_and_test_ids = np.concatenate((train_patient_ids, test_patient_ids))
        test_pos_seqs, _ = self.calculate_pos_neg_sequences(df_bld, df_hlt, "test" + name_metadata, train_and_test_ids, dataset_type, cell_type, num_of_patients=num_of_patients, filter_to_inflate=filter_to_inflate, verbose=verbose)
        test_bld_seqs = df_bld[df_bld['patient_id'].isin(test_patient_ids)]["AASeq"].unique()
        test_pos_seqs = np.array(list(set(test_pos_seqs) & set(test_bld_seqs)))

        # make sure that there is no intersection between train, valid and test sequences
        train_pos_seqs = np.array(list(set(train_pos_seqs) - set(np.concatenate((valid_pos_seqs, test_pos_seqs)))))
        if len(set(valid_pos_seqs) & set(test_pos_seqs)) > 0:
            if len(valid_pos_seqs) > len(test_pos_seqs):
                valid_pos_seqs = np.array(list(set(valid_pos_seqs) - set(test_pos_seqs)))
            else:
                test_pos_seqs = np.array(list(set(test_pos_seqs) - set(valid_pos_seqs)))

        positive_seqs = np.concatenate((train_pos_seqs, valid_pos_seqs, test_pos_seqs))

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
        if verbose:
            print(f"Number of Positive Sequences in General: {len(positive_seqs)}")
            print(f"Number of Positive Sequences in Test: {sum(test_inds)}, Percentage: {sum(test_inds) / len(positive_seqs) * 100:.2f}%")
            print(f"Number of Positive Sequences in Valid: {sum(valid_inds)}, Percentage: {sum(valid_inds) / len(positive_seqs) * 100:.2f}%")
            print(f"Number of Positive Sequences in Train: {sum(train_inds)}, Percentage: {sum(train_inds) / len(positive_seqs) * 100:.2f}%\n")
            print(f"Number of patients in General: {df_bld['patient_id'].nunique() + df_hlt['patient_id'].nunique()}")
            print(f"Number of Disease patients: {df_bld['patient_id'].nunique()}")
            print(f"Number of Healthy patients: {df_hlt['patient_id'].nunique()}")

        # Get the positive sequences for the test and train sets
        test_pos_seqs = np.array(positive_seqs)[test_inds]
        valid_pos_seqs = np.array(positive_seqs)[valid_inds]
        train_pos_seqs = np.array(positive_seqs)[train_inds]
        # Get the negative sequences
        neg_seqs = df_bld[df_bld['patient_id'].isin(train_patient_ids)]['AASeq'].unique()
        test_neg_seqs = df_bld[df_bld['patient_id'].isin(test_patient_ids)]['AASeq'].unique()
        valid_neg_seqs = df_bld[df_bld['patient_id'].isin(valid_patient_ids)]['AASeq'].unique()
        # Remove positive sequences from the negative sequences
        neg_seqs = np.array(list(set(neg_seqs) - set(positive_seqs)))
        test_neg_seqs = np.array(list(set(test_neg_seqs) - set(positive_seqs)))
        valid_neg_seqs = np.array(list(set(valid_neg_seqs) - set(positive_seqs)))
        # Get all sequences that are in valid_neg_seqs and valid_neg_seqs
        valid_test_neg_seqs = np.array(list(set(test_neg_seqs) & set(valid_neg_seqs)))
        if len(test_neg_seqs) < len(valid_neg_seqs):
            # remove valid_test_neg_seqs from valid_neg_seqs
            valid_neg_seqs = valid_neg_seqs[~np.isin(valid_neg_seqs, valid_test_neg_seqs)]
        else:
            # remove valid_test_neg_seqs from test_neg_seqs
            test_neg_seqs = test_neg_seqs[~np.isin(test_neg_seqs, valid_test_neg_seqs)]
        # Remove all valid_neg_seqs and test_neg_seqs sequences from the negative sequences
        neg_seqs = np.array(list(set(neg_seqs) - set(np.concatenate((valid_neg_seqs, test_neg_seqs)))))

        # get the dataframes for the test and train sets to convert AASeqs to ratios
        if dataset_type not in ['article', 'article_sle']:
            # concat df_bld and df_hlt to get the dataframes
            df_concat = pd.concat([df_bld, df_hlt], axis=0, ignore_index=True)
            self.build_clone_fraction_df(df_concat, method='max')

        # V1: f(x, a=1, b=0.5, c=0.5)  # b=1.5 might be better if we want most to be 1.0
        #     return a + c * (x ** b)
        # V2: f(x, a=1, b=0.3, c=1.5)  # Best?
        #     return a + c * (x ** b)
        # V3: f(x, a=0.2, b=1.5):  # Bad
        #     return a + b * x
        # V4:
        def f(x, a=1.01, b=160, c=0.03):
            h = lambda y: a / (1 + torch.e ** (-b * (y - c)))
            base = 1.0 - h(0)
            return base + h(x)

        def aaseq_to_ratio(aaseq_array, default_value=0.0, dont_use_function=False):
            lookup_series = self.df_aaseq_to_ratio.set_index('AASeq')['cloneFraction']
            result = pd.Series(aaseq_array).map(lookup_series).fillna(default_value)
            return result if dont_use_function else f(torch.tensor(result))

        self.build_distance_df(df_bld, train_pos_seqs, dataset_type)

        def distance_func(x, a=2.0, k=3.5, x_0=2.0, b=1.0):
            sig = 1 - 1 / (1 + torch.exp(-k * (x - x_0)))
            return a * sig + b

        self._dist_a = {"none" : 0, "v1" : 1, "v2" : 2, "v3" : 4, "v4" : 6}[dist_loss_type.lower()]
        def aaseq_to_distance(aaseq_array, default_value=1.0, dont_use_function=False):
            lookup_series = self.df_aaseq_to_distance.set_index('AASeq')['distance']
            result = pd.Series(aaseq_array).map(lookup_series).fillna(default_value)
            return result if dont_use_function else distance_func(torch.tensor(result), a=self._dist_a)

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
                neg_seqs = np.concatenate((new_neg_seqs, np.random.choice(remaining_neg_seqs, size=neg_seqs_to_add, replace=replace)))
                np.random.seed(42)
            else:
                neg_seqs = new_neg_seqs

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
        self.aaseq_to_ratio = aaseq_to_ratio
        self.aaseq_to_distance = aaseq_to_distance
        # [set(positive_seqs).intersection(set(df_hlt[df_hlt['patient_id'] == p]['AASeq'].values)) for p in df_hlt['patient_id'].unique()]

        # Note: looks like when we have less sequences in the set and when we take only seqs that appear in i patients,
        # and not in i healthy people, then when i rises the variance lowers.
        if False:
            import matplotlib.pyplot as plt
            def plot_lev(seqs, title_info=''):
                pwc_mat = pairwise_scores(seqs, seqs, score=levenshtein_dist_non_bin)
                similarities = np.mean(pwc_mat, axis=0)
                # similarities = np.min(pwc_mat + np.eye(len(pwc_mat)) * 100, axis=0)
                plt.hist(similarities, bins=10, edgecolor='black')
                plt.xlabel('Minimum Levenshtein Distance')
                plt.ylabel('Frequency')
                plt.title(f'Histogram of Minimum Levenshtein Distances - {title_info}\nMean: {similarities.mean():.3f}')
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.show()
            plot_lev(positive_seqs, 'Normal')
            for i in range(3, 12, 2):
                pos_seqs, _ =self.calculate_pos_neg_sequences(df_bld, df_hlt, "train" + name_metadata, train_patient_ids,
                                                              dataset_type, cell_type, num_of_patients=i, special_test=True)
                plot_lev(pos_seqs, f'num_of_patients f{i}')


    def use_similar_negatives_handler(self, neg_seqs, train_pos_seqs, neg_pos_ratio, neg_partition):
        similar_neg_cache_path = 'cache/ms'
        os.makedirs(similar_neg_cache_path, exist_ok=True)
        # Check if the file already exists
        similar_neg_seqs_path = os.path.join(similar_neg_cache_path, "similar_negatives.npy")
        similar_neg_min_dists_path = os.path.join(similar_neg_cache_path, "similar_neg_min_dists.npy")
        if os.path.exists(similar_neg_seqs_path) and os.path.exists(similar_neg_min_dists_path):
            # Load the file
            neg_seqs = np.load(similar_neg_seqs_path, allow_pickle=True)
            min_dist = np.load(similar_neg_min_dists_path, allow_pickle=True)
        else:
            import matplotlib.pyplot as plt
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

    def build_distance_df(self, df_bld, train_pos_seqs, dataset_type='ms', batch_size=1000):
        sorted_seqs = sorted(list(df_bld["AASeq"].unique()))
        # create hash of the df_dist in order to save or load it

        hash_str = "_".join(sorted_seqs)
        hash_of_df = hashlib.sha256(hash_str.encode()).hexdigest()

        # check if the df_dist already exists
        df_dist_filename = f"cache/{dataset_type}/df_dist_{hash_of_df}.pkl"
        if os.path.exists(df_dist_filename):
            # load the df_dist from the file
            df_dist = pd.read_pickle(df_dist_filename)
            self.df_aaseq_to_distance = df_dist
            return

        seqs = [(len(seq), seq) for seq in df_bld["AASeq"].unique()]
        possible_lens = set([x[0] for x in seqs])
        all_distances = []
        corresponding_seqs = []
        for possible_len in tqdm(possible_lens):
            seqs_in_this_len = [x[1] for x in seqs if x[0] == possible_len]
            num_seqs = len(seqs_in_this_len)
            # Process in batches if the number of sequences is large
            if num_seqs > batch_size:
                batched_min_dists = []
                for i in range(0, num_seqs, batch_size):
                    batch_end = min(i + batch_size, num_seqs)
                    batch_seqs = seqs_in_this_len[i:batch_end]

                    # Calculate pairwise distance for this batch
                    batch_pwc_mat = pairwise_scores(batch_seqs, train_pos_seqs, score=levenshtein_dist_non_bin)

                    # Get minimum distance for each sequence in the batch
                    batch_min_dist = np.min(batch_pwc_mat, axis=1)
                    batched_min_dists.append(batch_min_dist)

                min_dist = np.concatenate(batched_min_dists)
            else:
                # Original calculation for small sequence sets
                pwc_mat = pairwise_scores(seqs_in_this_len, train_pos_seqs, score=levenshtein_dist_non_bin)
                min_dist = np.min(pwc_mat, axis=1)
            all_distances.append(min_dist)
            corresponding_seqs.append(seqs_in_this_len)
        combined_distances = np.concatenate(all_distances)
        combined_seqs = np.concatenate(corresponding_seqs)
        # create a df that will be used to map the sequences to the distances
        df_dist = pd.DataFrame({'AASeq': combined_seqs, 'distance': combined_distances})
        self.df_aaseq_to_distance = df_dist

        # save the df_dist to the file
        os.makedirs(os.path.dirname(df_dist_filename), exist_ok=True)
        df_dist.to_pickle(df_dist_filename)

    def build_clone_fraction_df(self, df_bld, method='max'):
        """
        Create a DataFrame with unique AASeqs and their aggregated cloneFraction.

        Parameters:
        - df_bld (pd.DataFrame): Original DataFrame with 'AASeq' and 'cloneFraction'.
        - method (str): 'max', 'min', or 'avg' for aggregation.

        Returns:
        - pd.DataFrame: With columns 'AASeq' and 'cloneFraction'.
        """
        method_map = {
            'max': 'max',
            'min': 'min',
            'avg': 'mean',
            'med': 'median',
        }
        if method not in method_map:
            raise ValueError("method must be one of 'max', 'min', or 'avg'")

        df_norm = df_bld.copy()
        df_norm['cloneFraction'] = df_norm.groupby('patient_id')['cloneFraction'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.0
        )
        df_unique = df_norm.groupby('AASeq', as_index=False)['cloneFraction'].agg(method_map[method])
        self.df_aaseq_to_ratio = df_unique

    def calculate_pos_neg_sequences(self, df_bld, df_hlt, df_name, patient_ids, dataset_type, cell_type, num_of_patients=3, filter_to_inflate=True, verbose=True):
        # TODO: CHANGE THIS BACK TO THE NORMAL IMPLEMENTATION AFTER TESTING!!!
        if filter_to_inflate:
            df_bld = df_bld[df_bld['patient_id'].isin(patient_ids)]
            positive_seqs, negative_seqs = self.get_positive_negative(df_bld, df_hlt, df_name, dataset_type, cell_type, num_of_patients=num_of_patients, verbose=verbose)
            all_common_seqs = self.find_all_common_sequences(df_bld, num_of_patients=3)
            valid_seqs_healthy = self.find_all_common_sequences(df_hlt, num_of_patients=3)
            all_common_seqs = all_common_seqs - valid_seqs_healthy
            positive_seqs.update(all_common_seqs)
            return positive_seqs, negative_seqs
        else:
            temp_df = df_bld[df_bld['patient_id'].isin(patient_ids)]
            grouped = temp_df.groupby('patient_id')['AASeq'].unique()
            aa_seq_counter = Counter()
            for aa_seqs in grouped:
                aa_seq_counter.update(aa_seqs)
            valid_aa_seqs = {aa_seq for aa_seq, count in aa_seq_counter.items() if count >= num_of_patients}

            temp_df_h = df_hlt
            grouped_h = temp_df_h.groupby('patient_id')['AASeq'].unique()
            aa_seq_counter_h = Counter()
            for aa_seqs in grouped_h:
                aa_seq_counter_h.update(aa_seqs)
            valid_aa_seqs_h = {aa_seq for aa_seq, count in aa_seq_counter_h.items() if count >= num_of_patients}

            positive_seqs = np.array(list(set(valid_aa_seqs) - set(valid_aa_seqs_h)))
            negative_seqs = np.array(list(set(valid_aa_seqs_h) - set(valid_aa_seqs)))
            return positive_seqs, negative_seqs

    def get_aaseq_to_ratio_func(self):
        return self.aaseq_to_ratio

    def get_aaseq_to_distance_func(self):
        return self.aaseq_to_distance

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

    # Loading all valid sequences for disease and healthy samples
    def get_positive_negative(self, df_bld, df_hlt, df_name, dataset_type, cell_type, num_of_patients=3, verbose=True):
        all_common_seqs = self.find_all_common_sequences(df_bld, num_of_patients=num_of_patients)
        valid_seqs_healthy = self.find_all_common_sequences(df_hlt, num_of_patients=num_of_patients)
        all_common_seqs = all_common_seqs - valid_seqs_healthy
        # choosing valid samples according to their re-occurrence in different patients and a given distance
        valid_seqs_disease = self.calculate_valid_near_sequences(df_bld,
                                                                 save_name=f'{df_name}_disease_{dataset_type}_{cell_type}_neighbours{num_of_patients}',
                                                                 lev_dist_accept=1,
                                                                 num_of_patients=num_of_patients,
                                                                 all_common_seqs=all_common_seqs)

        positive_seqs = set(valid_seqs_disease)
        if verbose:
            print(f"Valid Disease Sequence (num of common = {num_of_patients}): {len(positive_seqs)}")

        # Extract valid letters
        valid_letters = set(''.join(valid_seqs_healthy))
        # Group healthy sequences by length
        length_groups = {}
        for seq in valid_seqs_healthy:
            length_groups.setdefault(len(seq), set()).add(seq)
        # Process each length group separately
        for seq_len, seq_group in length_groups.items():
            # Generate neighbors for this group
            neighbors = self.generate_neighbors(seq_group, valid_letters)
            # Remove neighbors from positive_seqs immediately
            positive_seqs -= neighbors  # This prevents storing all neighbors
        negative_seqs = set(valid_seqs_healthy)  # Negative sequences remain unchanged
        return positive_seqs, negative_seqs

    def get_all_usable_disease_data(self, disease='Multiple sclerosis', get_all=False):
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
            study_ids = [STUDY_ID6, STUDY_ID7, STUDY_ID8]
        elif disease == 'CMV':  # Cytomegalovirus
            study_ids = [STUDY_ID9, STUDY_ID10, STUDY_ID11]
        else:
            raise ValueError(f"Invalid disease: {disease}")

        for study_id in study_ids:
            study = Study(study_id)
            usable_samples = study._samples['usable']
            df = study.read_sample(usable_samples)
            if not get_all:
                df = df[df['condition'] == disease]
            df['study_id'] = study_id  # TODO: There is a SettingWithCopyWarning here!
            studies.append(df)

        return pd.concat(studies, ignore_index=True)

    def get_all_usable_healthy_data(self):
        healthy_study_ids = [HEALTHY_STUDY_ID, HEALTHY_STUDY_ID2, HEALTHY_STUDY_ID3, HEALTHY_STUDY_ID4,
                             HEALTHY_STUDY_ID5]
        healthy_studies = []
        for study_id in healthy_study_ids:
            study = Study(study_id)
            usable_samples = study._samples['usable']
            df = study.read_sample(usable_samples)
            df = df[df['condition'] == 'Healthy']
            df['study_id'] = study_id
            healthy_studies.append(df)
        df_concat = pd.concat(healthy_studies, ignore_index=True)
        df_concat = df_concat.dropna(subset=['AASeq'])
        return df_concat

    # This function loads the synapse dataframe of Mal-ID of only TCR and healthy samples for sure.
    def get_full_healthy_synapse_mal_id_dataframe(self, to_recalculate=False, get_all=False):
        # Defining constants
        synapse_db_folder = "db/synapse_Mal_ID"
        synapse_metadata_file = os.path.join(synapse_db_folder, "metadata.tsv")

        # Define the filename based on the get_all flag
        file_suffix = "_all" if get_all else "_healthy_only"
        df_filename = os.path.join(synapse_db_folder, f"synapse_mal_id_dataframe{file_suffix}.pkl")
        # Check if the DataFrame is already saved
        if not to_recalculate:
            if os.path.exists(df_filename):
                # Load the DataFrame from file
                df = pd.read_pickle(df_filename)
                return df

        # Reading metadata
        synapse_metadata = pd.read_csv(synapse_metadata_file, sep='\t')

        # Reading and interpreting data files
        synapse_datafiles = [x for x in os.listdir(synapse_db_folder) if x.endswith(".bz2")]

        healthy_samples = []
        for datafile in tqdm(synapse_datafiles, desc="Processing data files", total=len(synapse_datafiles)):
            datafile_id = datafile.split("_")[-1][:-4]
            datafile_path = os.path.join(synapse_db_folder, datafile)

            # Reading metadata and data
            metadata = synapse_metadata[synapse_metadata['participant_label'] == datafile_id]
            condition = metadata['disease'].values[0]
            if get_all or 'Healthy' in condition:
                data = pd.read_csv(datafile_path, sep='\t', compression='bz2')
                data = data.loc[:, ['cdr3_seq_aa_q', 'participant_label', 'specimen_tissue']]
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
            columns={'cdr3_seq_aa_q': 'AASeq', 'participant_label': 'patient_id', 'specimen_tissue': 'tissue'})

        df = df[~df['AASeq'].str.contains('[^ACDEFGHIKLMNPQRSTVWY]', regex=True)]
        df['AASeq'] = 'C' + df['AASeq'] + 'F'

        # Save the DataFrame for future use
        df.to_pickle(df_filename)

        return df

    # TODO: There is a major problem with loading this dataset. The current way doesn't use information from "file_key" so its unusable now..
    def get_full_article2_dataframe(self):
        article2_data_folder = 'db/test_db/data_tcrb'
        article2_data_files = os.listdir(article2_data_folder)
        article2_data_files = [x for x in article2_data_files if 'CDR3_list' in x and x.endswith('2.csv')]
        # Open all files and read the contents
        all_article2_dfs = []
        for file_name in article2_data_files:
            # read the file as .csv (include header as well)
            file_path = os.path.join(article2_data_folder, file_name)
            df = pd.read_csv(file_path, names=['AASeq', 'col 1', 'col 2', 'cloneFraction'])

            # normalize the cloneFraction column
            df['cloneFraction'] -= df['cloneFraction'].min()
            df['cloneFraction'] /= df['cloneFraction'].max()

            # add patient_id as 5th and 6th columns
            patient_id = file_name.split('_')[0]
            df['patient_id'] = patient_id
            df['study_id'] = 'article2'

            # modify AASeq to start with 'C' and end with 'F'
            df['AASeq'] = 'C' + df['AASeq'] + 'F'

            # add the sequences to the set
            all_article2_dfs.append(df)
        article2_df = pd.concat(all_article2_dfs, ignore_index=True)
        patients_with_samples = [x[0] for x in article2_df.groupby('patient_id')['AASeq'] if len(x[1]) >= 2000]
        return article2_df[article2_df['patient_id'].isin(patients_with_samples)]

    def find_all_common_sequences(self, df, num_of_patients=3):
        # Step 1: Group by 'patient_id' and get unique AASeqs
        grouped = df.groupby('patient_id')['AASeq'].unique()

        # Step 2: Count occurrences of each AASeq across different patient groups
        aa_seq_counter = Counter()
        for aa_seqs in grouped:
            aa_seq_counter.update(aa_seqs)

        # Step 3: Filter AASeqs that appear in at least num_of_patients different patients
        valid_aa_seqs = {aa_seq for aa_seq, count in aa_seq_counter.items() if count >= num_of_patients}

        return valid_aa_seqs

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
            valid_seqs_batch = self.helper_function_common_aaseq_analysis(df_combined, lev_dist_accept, only_valid=True, verbose=False)

            # Accumulate valid sequences from this batch (as a set)
            valid_seqs_set.update(valid_seqs_batch)

        # The result is a set of valid sequences
        return valid_seqs_set

    def calculate_valid_near_sequences(self, df, save_name, lev_dist_accept=1, num_of_patients=3, all_common_seqs=None):
        save_folder = "cache/valid_sequences/multiple_sclerosis"
        save_file = os.path.join(save_folder, f"{save_name}_valid_seqs_dist_{lev_dist_accept}.pkl")
        if not os.path.exists(save_file):
            if all_common_seqs is None:
                all_common_seqs = self.find_all_common_sequences(df, num_of_patients=num_of_patients)
            valid_seqs = self.process_in_batches(df, all_common_seqs, 512, lev_dist_accept)
            os.makedirs(save_folder, exist_ok=True)
            with open(save_file, "wb") as f:
                pickle.dump(valid_seqs, f)
        else:
            with open(save_file, "rb") as f:
                valid_seqs = pickle.load(f)
        return valid_seqs

    def generate_neighbors(self, sequences, valid_letters):
        valid_letters = set(valid_letters)  # Ensure valid letters are a set for quick lookup
        neighbor_set = set(sequences)  # Start with the original sequences

        for seq in sequences:
            seq_len = len(seq)

            # Generate substitutions
            for i in range(seq_len):
                for letter in valid_letters:
                    if seq[i] != letter:  # Avoid replacing with the same letter
                        neighbor_set.add(seq[:i] + letter + seq[i + 1:])

            # Generate insertions
            for i in range(seq_len + 1):
                for letter in valid_letters:
                    neighbor_set.add(seq[:i] + letter + seq[i:])

            # Generate deletions
            if seq_len > 1:  # Ensure we don't delete the only character
                for i in range(seq_len):
                    neighbor_set.add(seq[:i] + seq[i + 1:])

        return neighbor_set

    def helper_function_common_aaseq_analysis(self, df, lev_dist_accept, only_valid=False, verbose=True):
        # Step 1: Create patient-wise groups
        seqs_by_patient = df.groupby('patient_id')['AASeq'].unique().to_dict()

        # Step 2: Compare sequences across different patients
        valid_sequences = set()
        iterator = list(combinations(seqs_by_patient.items(), 2))
        if verbose and len(iterator) > 1:
            iterator = tqdm(iterator, total=len(iterator))
        for (pid1, seqs1), (pid2, seqs2) in iterator:
            # Sort seqs1 by length
            seqs1_by_len = {}
            for seq in seqs1:
                seq_len = len(seq)
                if seq_len not in seqs1_by_len:
                    seqs1_by_len[seq_len] = []
                seqs1_by_len[seq_len].append(seq)

            # Convert lists to numpy arrays for efficiency
            seqs1_by_len = {k: np.array(v, dtype=object) for k, v in seqs1_by_len.items()}

            # Sort seqs2 by length for efficient filtering
            seqs2_by_len = np.array(sorted(seqs2, key=len), dtype=object)
            seqs2_lens = np.array([len(seq) for seq in seqs2_by_len])

            # Iterate over length groups in seqs1
            inner_itter = list(seqs1_by_len.items())
            if verbose:
                inner_itter = tqdm(inner_itter, total=len(inner_itter))
            for length, group1 in inner_itter:
                # Select seqs2 that are in the range [length-1, length+1] (or +- lev_dist_accept)
                min_len, max_len = length - lev_dist_accept, length + lev_dist_accept
                mask = (seqs2_lens >= min_len) & (seqs2_lens <= max_len)
                group2 = seqs2_by_len[mask]

                if len(group2) > 0:
                    # Compute pairwise identity matrix
                    pwc_mat = pairwise_scores(group1, group2, score=levenshtein_dist_non_bin)
                    sim_inxs = np.where(pwc_mat <= lev_dist_accept)

                    # Add matching sequences with patient_id to valid set
                    for x, y in zip(sim_inxs[0], sim_inxs[1]):
                        if only_valid:
                            valid_sequences.add(group1[x])
                            valid_sequences.add(group2[y])
                        else:
                            valid_sequences.add((group1[x], pid1, pid2))  # Tuple (seq, originating pid x2)
                            valid_sequences.add((group2[y], pid1, pid2))  # Tuple (seq, originating pid x2)

        if only_valid:
            return valid_sequences

        # Step 3: Calculate the mean number of common sequences and percentages
        unique_patient_ids = df["patient_id"].unique()
        masks = []
        for patient in unique_patient_ids:
            # Get sequences that belong to the current patient
            patient_seqs = set(df.loc[df["patient_id"] == patient, "AASeq"])

            # Create a mask for sequences
            mask = np.array([1 if seq[0] in patient_seqs else 0 for seq in valid_sequences])
            masks.append(mask)

        # Convert to ndarray
        masks = np.array(masks)  # Shape: (num_unique_patients, len(sequences))
        return masks

    def common_aaseq_analysis(self, df, num_of_patients, lev_dist_accept=0, mode=1):
        # Select the unique patients
        unique_patients = df['patient_id'].unique()

        if len(unique_patients) < num_of_patients:
            raise ValueError("Number of patients in the dataframe is less than num_of_patients")

        # Create a dictionary mapping each patient_id to their set of AASeq
        patient_sequences = {pid: set(df[df['patient_id'] == pid]['AASeq']) for pid in unique_patients}

        results = []
        if mode == 1:
            # Mode 1: All combinations
            patient_combinations = combinations(unique_patients, num_of_patients)
        elif mode == 2:
            # Mode 2: Always include the first patient
            patient_combinations = [tuple([unique_patients[0]] + list(comb)) for comb in
                                    combinations(unique_patients[1:], num_of_patients - 1)]
        else:
            raise ValueError("Mode must be 1 (All combinations) or 2 (Always include the first patient).")

        percent_of_total_values = []
        for combination in patient_combinations:
            selected_sequences = [patient_sequences[pid] for pid in combination]

            if lev_dist_accept >= 1:
                temp_df = df[df['patient_id'].isin(combination)]
                masks = self.helper_function_common_aaseq_analysis(temp_df, lev_dist_accept)
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
            percent_of_min = (num_common / min_sequences) * 100 if min_sequences > 0 else 0
            percent_of_max = (num_common / max_sequences) * 100 if max_sequences > 0 else 0
            percent_of_total_values.append(percent_of_total)
            results.append({
                'num_total_seqs': total_sequences,
                'num_common': num_common,
                'percent_of_total': percent_of_total,
                'percent_of_min': percent_of_min,
                'percent_of_max': percent_of_max
            })

        # Calculate means
        mean_results = {
            'num_total_seqs': sum(r['num_total_seqs'] for r in results) / len(results),
            'num_common': sum(r['num_common'] for r in results) / len(results),
            'percent_of_total': sum(r['percent_of_total'] for r in results) / len(results),
            'percent_of_min': sum(r['percent_of_min'] for r in results) / len(results),
            'percent_of_max': sum(r['percent_of_max'] for r in results) / len(results)
        }

        # Calculate std for percent_of_total
        std_percent_of_total = np.std(percent_of_total_values)  # Calculate std for percent_of_total

        return mean_results, std_percent_of_total
