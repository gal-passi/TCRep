import os
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from itertools import combinations
from tqdm import tqdm
from Curation import Study
from utils import pairwise_scores, levenshtein_dist, levenshtein_dist_non_bin


STUDY_ID = 'PRJNA393498'  # Ankylosing Spondylitis study
STUDY_ID2 = 'immunoSEQ47'  # Hepatitis B virus study
STUDY_ID3 = 'immunoSEQ77'  # Rheumatoid arthritis study (plus healthy)
STUDY_ID4 = 'PRJNA258001'  # HIV study (plus healthy)
STUDY_ID5 = 'PRJNA390125'  # Only healthy study
STUDY_ID6 = 'PRJNA495603'  # Multiple sclerosis study (plus healthy)
STUDY_ID7 = 'PRJNA579190'  #  Multiple sclerosis study (plus healthy)
STUDY_ID8 = 'PRJNA280417'  #  Multiple sclerosis study
HEALTHY_STUDY_ID = STUDY_ID3  # ONLY CD8
HEALTHY_STUDY_ID2 = STUDY_ID4  # Both CD8 and CD4
HEALTHY_STUDY_ID3 = STUDY_ID5  # Larger both CD8 and CD4 (But fewer patients!)
HEALTHY_STUDY_ID4 = STUDY_ID6  # Other healthy study
HEALTHY_STUDY_ID5 = STUDY_ID7  # Other healthy study
STUDIES = [STUDY_ID, STUDY_ID2, STUDY_ID3, STUDY_ID4, STUDY_ID5, STUDY_ID6, STUDY_ID7]


class DatasetLoader:
    def __init__(self, dataset_type: str):
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
                # TODO: Consider adding the other healthy dataset to the article healthy dataset!
                df_bld = df_article[df_article["condition"] == "T1D"]
                df_hlt = df_article[df_article["condition"] == "Healthy"]
            else:
                raise ValueError("Invalid dataset type")

        # TODO: This code checks the intersection of healthy and disease samples with the article
        # healthy_unique = set(df_hlt['AASeq'].unique())
        # df_article = self.get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=True)
        # article_unique = set(df_article['AASeq'].unique())
        # print(f"Number of sequences in the article: \t{len(df_article['AASeq'])}")
        # print(f"Number of unique sequences in the article: {len(article_unique)}")
        # print(f"Number of sequences in the intersection of healthy and article: {len(healthy_unique & article_unique)}")
        # print(f"Number of sequences in the intersection of disease and article: {len(set(df_bld['AASeq']) & article_unique)}")
        # exit(0)

        positive_seqs, negative_seqs = self.get_positive_negative(df_bld, df_hlt, dataset_type, cell_type, num_of_patients=3)
        all_common_seqs = self.find_all_common_sequences(df_bld, num_of_patients=3)
        valid_seqs_healthy = self.find_all_common_sequences(df_hlt, num_of_patients=3)
        all_common_seqs = all_common_seqs - valid_seqs_healthy
        positive_seqs.update(all_common_seqs)

        # make list and sort
        positive_seqs = list(positive_seqs)
        positive_seqs.sort()
        np.random.seed(42)
        np.random.shuffle(positive_seqs)

        # getting patient id masks in order to do k-fold by patient (according to synovial samples)
        unique_patient_ids = df_bld["patient_id"].unique()
        unique_patient_ids = np.random.permutation(unique_patient_ids)
        masks = []
        for patient in unique_patient_ids:
            # Get sequences that belong to the current patient
            patient_seqs = set(df_bld.loc[df_bld["patient_id"] == patient, "AASeq"])
            # Create a mask for sequences
            mask = np.array([1 if seq in patient_seqs else 0 for seq in positive_seqs])
            if 1 in mask:
                masks.append(mask)
        # Convert to ndarray
        patient_id_masks = np.array(masks)  # Shape: (num_unique_patients, len(positive_seqs))

        # pick index of 8 unique patients from unique_patient_ids as test patients and the rest as train patients
        num_test_patients = 8
        test_patient_ids = unique_patient_ids[:num_test_patients]
        test_patient_ids, valid_patient_ids = test_patient_ids[:num_test_patients // 2], test_patient_ids[
                                                                                         num_test_patients // 2:]
        train_patient_ids = unique_patient_ids[num_test_patients:]
        # translate back to the inds according to unique_patient_ids
        test_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in test_patient_ids])
        valid_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in valid_patient_ids])
        train_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in train_patient_ids])

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
        # neg_seqs = neg_seqs[~np.isin(neg_seqs, np.concatenate((valid_neg_seqs, test_neg_seqs)))]

        # TODO: Figure out what to do about cases where negative sequences appear in other sets
        #  (such as positives and maybe in across different valid\test\train sets)

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
    def get_positive_negative(self, df_bld, df_hlt, dataset_type, cell_type, num_of_patients=3):
        all_common_seqs = self.find_all_common_sequences(df_bld, num_of_patients=num_of_patients)
        valid_seqs_healthy = self.find_all_common_sequences(df_hlt, num_of_patients=num_of_patients)
        all_common_seqs = all_common_seqs - valid_seqs_healthy
        # choosing valid samples according to their re-occurrence in different patients and a given distance
        valid_seqs_disease = self.calculate_valid_near_sequences(df_bld,
                                                                 save_name=f'disease_{dataset_type}_{cell_type}_neighbours{num_of_patients}',
                                                                 lev_dist_accept=1,
                                                                 num_of_patients=num_of_patients,
                                                                 all_common_seqs=all_common_seqs)

        positive_seqs = set(valid_seqs_disease)
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

    def get_all_usable_disease_data(self, disease='Multiple sclerosis'):
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
        else:
            raise ValueError(f"Invalid disease: {disease}")

        for study_id in study_ids:
            study = Study(study_id)
            usable_samples = study._samples['usable']
            df = study.read_sample(usable_samples)
            df = df[df['condition'] == disease]
            df['study_id'] = study_id
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

    # TODO: Remove the variable healthy_unique from the function signature
    # TODO: This function loads the synapse dataframe of Mal-ID of only TCR and healthy samples for sure.
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
                num_common = np.sum(np.any(masks == 1,
                                           axis=0))  # TODO: This will always increase when we look at more patients... this isnt the calculation that we want here
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


