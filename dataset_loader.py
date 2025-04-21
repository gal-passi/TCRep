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
    def __init__(self, dataset_type: str, unique_patient_ids=None, get_only_unique_patient_ids=False, k_fold=0):
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
                df_bld = df_article[df_article["condition"] == "T1D"]
                df_hlt = df_article[df_article["condition"] == "Healthy"]
                # TODO: This might be problematic: adding the other healthy dataset to the article healthy dataset!
                #  This can cause problems because the healthy people from a different study might be too different from healthy people from this dataset.
                #  So it might be too easy for the model to separate between them.
                df_hlt = pd.concat([df_hlt, df_h[['AASeq', 'patient_id', 'tissue', 'condition']]], axis=0, ignore_index=True)
            elif dataset_type == 'cmv':
                df_cmv = self.get_all_usable_disease_data(disease='CMV', get_all=True)
                df_bld = df_cmv[df_cmv["condition"] == "CMV"]
                df_hlt = df_cmv[df_cmv["condition"] == "Healthy"]
                df_hlt = pd.concat([df_hlt, df_h], axis=0, ignore_index=True)
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

        if k_fold > 0:
            name_metadata = f"_fold_{k_fold}"
        else:
            name_metadata = ""

        # TODO: The save cache files when doing k-fold should have different names!
        train_pos_seqs, _ = self.calculate_pos_neg_sequences(df_bld, df_hlt, "train" + name_metadata, train_patient_ids, dataset_type, cell_type, num_of_patients=3)
        # Calculate positive valid sequences
        train_and_valid_ids = np.concatenate((train_patient_ids, valid_patient_ids))
        valid_pos_seqs, _ = self.calculate_pos_neg_sequences(df_bld, df_hlt, "valid" + name_metadata, train_and_valid_ids, dataset_type, cell_type, num_of_patients=3)
        valid_bld_seqs = df_bld[df_bld['patient_id'].isin(valid_patient_ids)]["AASeq"].unique()
        valid_pos_seqs = np.array(list(set(valid_pos_seqs) & set(valid_bld_seqs)))
        # Calculate positive test sequences
        train_and_test_ids = np.concatenate((train_patient_ids, test_patient_ids))
        test_pos_seqs, _ = self.calculate_pos_neg_sequences(df_bld, df_hlt, "test" + name_metadata, train_and_test_ids, dataset_type, cell_type, num_of_patients=3)
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
        self.build_clone_fraction_df(df_bld, method='max')

        def aaseq_to_ratio(aaseq_array, default_value=0.0):
            lookup_series = self.df_aaseq_to_ratio.set_index('AASeq')['cloneFraction']
            result = pd.Series(aaseq_array).map(lookup_series).fillna(default_value)
            return result.to_numpy()

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
            'avg': 'mean'
        }
        if method not in method_map:
            raise ValueError("method must be one of 'max', 'min', or 'avg'")

        df_unique = df_bld.groupby('AASeq', as_index=False)['cloneFraction'].agg(method_map[method])
        self.df_aaseq_to_ratio = df_unique

    def calculate_pos_neg_sequences(self, df_bld, df_hlt, df_name, patient_ids, dataset_type, cell_type, num_of_patients=3):
        df_bld = df_bld[df_bld['patient_id'].isin(patient_ids)]
        positive_seqs, negative_seqs = self.get_positive_negative(df_bld, df_hlt, df_name, dataset_type, cell_type, num_of_patients=num_of_patients)
        all_common_seqs = self.find_all_common_sequences(df_bld, num_of_patients=3)
        valid_seqs_healthy = self.find_all_common_sequences(df_hlt, num_of_patients=3)
        all_common_seqs = all_common_seqs - valid_seqs_healthy
        positive_seqs.update(all_common_seqs)
        return positive_seqs, negative_seqs

    def get_aaseq_to_ratio_func(self):
        return self.aaseq_to_ratio

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
    def get_positive_negative(self, df_bld, df_hlt, df_name, dataset_type, cell_type, num_of_patients=3):
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

    # # Claude implementation!
    # def __init__(self, dataset_type='ms', random_seed=42, cell_type='ALL', num_test_patients=8):
    #     """
    #     Initialize the DatasetLoader with dataset type and options.
    #
    #     Args:
    #         dataset_type (str): Type of dataset ('ms' or 'article')
    #         random_seed (int): Random seed for reproducibility
    #         cell_type (str): Cell type to filter by ('CD8', 'CD4', or 'ALL')
    #         num_test_patients (int): Number of patients to use for test and validation
    #     """
    #     self.dataset_type = dataset_type
    #     self.random_seed = random_seed
    #     self.cell_type = cell_type
    #     self.num_test_patients = num_test_patients
    #
    #     # Initialize datasets and split data
    #     self.setup_datasets()
    #     self.prepare_data(cell_type, num_test_patients)
    #
    # def setup_datasets(self):
    #     """Set up the disease and healthy datasets based on dataset_type."""
    #     disease = 'Multiple sclerosis'
    #
    #     if self.dataset_type == 'ms':
    #         self.df_bld = self.get_all_usable_disease_data(disease=disease)
    #         self.df_hlt = self.get_all_usable_healthy_data()
    #     elif self.dataset_type == 'article':
    #         df_article = self.get_full_healthy_synapse_mal_id_dataframe(to_recalculate=False, get_all=True)
    #         self.df_bld = df_article[df_article["condition"] == "T1D"]
    #         self.df_hlt = df_article[df_article["condition"] == "Healthy"]
    #     else:
    #         raise ValueError(f"Invalid dataset type: {self.dataset_type}")
    #
    # def get_dfs(self):
    #     """Return the disease and healthy dataframes."""
    #     return self.df_bld, self.df_hlt
    #
    # def get_seqs(self):
    #     """Return sequence sets (train positive, negative, validation positive, validation negative,
    #     test positive, test negative)."""
    #     return (
    #         self.train_pos_seqs,
    #         self.train_neg_seqs,
    #         self.valid_pos_seqs,
    #         self.valid_neg_seqs,
    #         self.test_pos_seqs,
    #         self.test_neg_seqs
    #     )
    #
    # def get_patient_ids(self):
    #     """Return patient IDs for each split."""
    #     return self.train_patient_ids, self.valid_patient_ids, self.test_patient_ids
    #
    # def get_patient_inds(self):
    #     """Return patient indices for each split."""
    #     return self.train_patient_inds, self.valid_patient_inds, self.test_patient_inds
    #
    # def get_masks(self):
    #     """Return masks for each split."""
    #     return self.train_masks, self.valid_masks, self.test_masks
    #
    # def filter_by_cell_type(self, cell_type='ALL'):
    #     """
    #     Filter datasets by cell type if applicable.
    #
    #     Args:
    #         cell_type (str): Cell type to filter by ('CD8', 'CD4', or 'ALL')
    #
    #     Returns:
    #         tuple: Filtered disease and healthy dataframes
    #     """
    #     if cell_type != 'ALL' and self.dataset_type == 'ms':
    #         df_bld = self.df_bld[self.df_bld['cell_type'] == cell_type]
    #         df_hlt = self.df_hlt[self.df_hlt['cell_type'] == cell_type]
    #     else:
    #         df_bld, df_hlt = self.df_bld, self.df_hlt
    #
    #     return df_bld, df_hlt
    #
    # def split_datasets(self, cell_type='ALL', num_test_patients=8):
    #     """
    #     Split datasets into train, validation, and test sets.
    #
    #     Args:
    #         cell_type (str): Cell type to filter by ('CD8', 'CD4', or 'ALL')
    #         num_test_patients (int): Number of patients to use for test+validation
    #
    #     Returns:
    #         dict: Dictionary containing all split datasets and metadata
    #     """
    #     # Reset random seed for reproducibility
    #     np.random.seed(self.random_seed)
    #
    #     # Get filtered dataframes
    #     df_bld, df_hlt = self.filter_by_cell_type(cell_type)
    #
    #     # Step 1: Split patients into train, validation, and test sets
    #     patient_splits = self._split_patients(df_bld, num_test_patients)
    #
    #     # Step 2: Process positive sequences for each split
    #     positive_data = self._process_positive_sequences(df_bld, df_hlt, patient_splits, cell_type)
    #
    #     # Step 3: Process negative sequences for each split without intersections
    #     negative_data = self._process_negative_sequences(df_bld, patient_splits, positive_data)
    #
    #     # Step 4: Collect all metadata and results
    #     results = {
    #         # Original dataframes
    #         'df_bld': df_bld,
    #         'df_hlt': df_hlt,
    #
    #         # Patient IDs and indexes
    #         'unique_patient_ids': patient_splits['unique_patient_ids'],
    #         'train_patient_ids': patient_splits['train_patient_ids'],
    #         'valid_patient_ids': patient_splits['valid_patient_ids'],
    #         'test_patient_ids': patient_splits['test_patient_ids'],
    #         'train_patient_inds': patient_splits['train_patient_inds'],
    #         'valid_patient_inds': patient_splits['valid_patient_inds'],
    #         'test_patient_inds': patient_splits['test_patient_inds'],
    #
    #         # Patient masks
    #         'patient_id_masks': patient_splits['patient_id_masks'],
    #         'train_masks': patient_splits['train_masks'],
    #         'valid_masks': patient_splits['valid_masks'],
    #         'test_masks': patient_splits['test_masks'],
    #         'train_inds': positive_data['train_inds'],
    #
    #         # Positive sequences
    #         'positive_seqs': positive_data['positive_seqs'],
    #         'train_pos_seqs': positive_data['train_pos_seqs'],
    #         'valid_pos_seqs': positive_data['valid_pos_seqs'],
    #         'test_pos_seqs': positive_data['test_pos_seqs'],
    #
    #         # Negative sequences - maintain original attribute name 'neg_seqs' for backwards compatibility
    #         'neg_seqs': negative_data['train_neg_seqs'],  # For backwards compatibility
    #         'train_neg_seqs': negative_data['train_neg_seqs'],
    #         'valid_neg_seqs': negative_data['valid_neg_seqs'],
    #         'test_neg_seqs': negative_data['test_neg_seqs']
    #     }
    #
    #     return results
    #
    # def _split_patients(self, df_bld, num_test_patients):
    #     """
    #     Split patients into train, validation, and test sets.
    #
    #     Args:
    #         df_bld (DataFrame): Disease dataset
    #         num_test_patients (int): Number of patients for test+validation
    #
    #     Returns:
    #         dict: Dictionary with patient splits and masks
    #     """
    #     # Get unique patient IDs and shuffle them
    #     unique_patient_ids = df_bld["patient_id"].unique()
    #     unique_patient_ids = np.random.permutation(unique_patient_ids)
    #
    #     # Split patients into test, validation, and train
    #     test_patient_ids = unique_patient_ids[:num_test_patients]
    #     test_patient_ids, valid_patient_ids = (
    #         test_patient_ids[:num_test_patients // 2],
    #         test_patient_ids[num_test_patients // 2:]
    #     )
    #     train_patient_ids = unique_patient_ids[num_test_patients:]
    #
    #     # Get indices for each patient group
    #     test_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in test_patient_ids])
    #     valid_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in valid_patient_ids])
    #     train_patient_inds = np.array([np.where(unique_patient_ids == pid)[0][0] for pid in train_patient_ids])
    #
    #     return {
    #         'unique_patient_ids': unique_patient_ids,
    #         'test_patient_ids': test_patient_ids,
    #         'valid_patient_ids': valid_patient_ids,
    #         'train_patient_ids': train_patient_ids,
    #         'test_patient_inds': test_patient_inds,
    #         'valid_patient_inds': valid_patient_inds,
    #         'train_patient_inds': train_patient_inds,
    #         'patient_id_masks': None,  # Will be filled later
    #         'train_masks': None,  # Will be filled later
    #         'valid_masks': None,  # Will be filled later
    #         'test_masks': None  # Will be filled later
    #     }
    #
    # def _process_positive_sequences(self, df_bld, df_hlt, patient_splits, cell_type):
    #     """
    #     Process positive sequences for each split.
    #
    #     Args:
    #         df_bld (DataFrame): Disease dataset
    #         df_hlt (DataFrame): Healthy dataset
    #         patient_splits (dict): Patient split information
    #         cell_type (str): Cell type being used
    #
    #     Returns:
    #         dict: Positive sequences for each split
    #     """
    #     # Step 1: Get initial positive and negative sets
    #     train_df = df_bld[df_bld['patient_id'].isin(patient_splits['train_patient_ids'])]
    #     valid_df = df_bld[df_bld['patient_id'].isin(patient_splits['valid_patient_ids'])]
    #     test_df = df_bld[df_bld['patient_id'].isin(patient_splits['test_patient_ids'])]
    #
    #     # Step 2: Calculate common sequences separately for each set
    #     # For train set
    #     train_positive_seqs, _ = self.get_positive_negative(
    #         train_df, df_hlt, self.dataset_type, cell_type, num_of_patients=3
    #     )
    #     train_common_seqs = self.find_all_common_sequences(train_df, num_of_patients=3)
    #     valid_seqs_healthy = self.find_all_common_sequences(df_hlt, num_of_patients=3)
    #     train_common_seqs = train_common_seqs - valid_seqs_healthy
    #     train_positive_seqs.update(train_common_seqs)
    #
    #     # For validation set (ensuring no overlap with train)
    #     valid_positive_seqs, _ = self.get_positive_negative(
    #         valid_df, df_hlt, self.dataset_type, cell_type, num_of_patients=3
    #     )
    #     valid_common_seqs = self.find_all_common_sequences(valid_df, num_of_patients=3)
    #     valid_common_seqs = valid_common_seqs - valid_seqs_healthy
    #     valid_positive_seqs.update(valid_common_seqs)
    #     # Remove any sequences that are in train set
    #     valid_positive_seqs = valid_positive_seqs - train_positive_seqs
    #
    #     # For test set (ensuring no overlap with train or validation)
    #     test_positive_seqs, _ = self.get_positive_negative(
    #         test_df, df_hlt, self.dataset_type, cell_type, num_of_patients=3
    #     )
    #     test_common_seqs = self.find_all_common_sequences(test_df, num_of_patients=3)
    #     test_common_seqs = test_common_seqs - valid_seqs_healthy
    #     test_positive_seqs.update(test_common_seqs)
    #     # Remove any sequences that are in train or validation sets
    #     test_positive_seqs = test_positive_seqs - train_positive_seqs - valid_positive_seqs
    #
    #     # Combine all positive sequences and create masks
    #     all_positive_seqs = list(train_positive_seqs | valid_positive_seqs | test_positive_seqs)
    #     all_positive_seqs.sort()
    #     np.random.shuffle(all_positive_seqs)
    #
    #     # Create patient masks for the positive sequences
    #     masks = []
    #     for patient in patient_splits['unique_patient_ids']:
    #         patient_seqs = set(df_bld.loc[df_bld["patient_id"] == patient, "AASeq"])
    #         mask = np.array([1 if seq in patient_seqs else 0 for seq in all_positive_seqs])
    #         if 1 in mask:
    #             masks.append(mask)
    #
    #     patient_id_masks = np.array(masks)
    #
    #     # Update patient_splits with the masks
    #     patient_splits['patient_id_masks'] = patient_id_masks
    #     patient_splits['test_masks'] = patient_id_masks[patient_splits['test_patient_inds']]
    #     patient_splits['valid_masks'] = patient_id_masks[patient_splits['valid_patient_inds']]
    #     patient_splits['train_masks'] = patient_id_masks[patient_splits['train_patient_inds']]
    #
    #     # Create index masks for each split
    #     test_inds = patient_splits['test_masks'].any(axis=0)
    #     valid_inds = patient_splits['valid_masks'].any(axis=0)
    #
    #     # Handle any potential overlap
    #     valid_test_inds = test_inds & valid_inds
    #     if sum(valid_inds) > sum(test_inds):
    #         test_inds = test_inds | valid_test_inds
    #         valid_inds = valid_inds & ~valid_test_inds
    #     else:
    #         valid_inds = valid_inds | valid_test_inds
    #         test_inds = test_inds & ~valid_test_inds
    #
    #     train_inds = (~test_inds) & (~valid_inds)
    #
    #     # Get positive sequences for each split
    #     train_pos_seqs = np.array(all_positive_seqs)[train_inds]
    #     valid_pos_seqs = np.array(all_positive_seqs)[valid_inds]
    #     test_pos_seqs = np.array(all_positive_seqs)[test_inds]
    #
    #     # Print statistics
    #     print(f"Number of Positive Sequences in General: {len(all_positive_seqs)}")
    #     print(f"Number of Positive Sequences in Test: {sum(test_inds)}, "
    #           f"Percentage: {sum(test_inds) / len(all_positive_seqs) * 100:.2f}%")
    #     print(f"Number of Positive Sequences in Valid: {sum(valid_inds)}, "
    #           f"Percentage: {sum(valid_inds) / len(all_positive_seqs) * 100:.2f}%")
    #     print(f"Number of Positive Sequences in Train: {sum(train_inds)}, "
    #           f"Percentage: {sum(train_inds) / len(all_positive_seqs) * 100:.2f}%\n")
    #
    #     print(f"Number of patients in General: {df_bld['patient_id'].nunique() + df_hlt['patient_id'].nunique()}")
    #     print(f"Number of Disease patients: {df_bld['patient_id'].nunique()}")
    #     print(f"Number of Healthy patients: {df_hlt['patient_id'].nunique()}")
    #
    #     return {
    #         'positive_seqs': all_positive_seqs,
    #         'train_pos_seqs': train_pos_seqs,
    #         'valid_pos_seqs': valid_pos_seqs,
    #         'test_pos_seqs': test_pos_seqs,
    #         'train_inds': train_inds
    #     }
    #
    # def _process_negative_sequences(self, df_bld, patient_splits, positive_data):
    #     """
    #     Process negative sequences ensuring no intersections with positive sets.
    #
    #     Args:
    #         df_bld (DataFrame): Disease dataset
    #         patient_splits (dict): Patient split information
    #         positive_data (dict): Positive sequence information
    #
    #     Returns:
    #         dict: Negative sequences for each split
    #     """
    #     # Get negative sequences from each patient split
    #     train_neg_seqs = set(df_bld[df_bld['patient_id'].isin(patient_splits['train_patient_ids'])]['AASeq'].unique())
    #     valid_neg_seqs = set(df_bld[df_bld['patient_id'].isin(patient_splits['valid_patient_ids'])]['AASeq'].unique())
    #     test_neg_seqs = set(df_bld[df_bld['patient_id'].isin(patient_splits['test_patient_ids'])]['AASeq'].unique())
    #
    #     # Remove any sequences that are in positive sets
    #     train_pos_set = set(positive_data['train_pos_seqs'])
    #     valid_pos_set = set(positive_data['valid_pos_seqs'])
    #     test_pos_set = set(positive_data['test_pos_seqs'])
    #
    #     # Ensure no intersection between negative and positive sets
    #     train_neg_seqs = train_neg_seqs - train_pos_set - valid_pos_set - test_pos_set
    #     valid_neg_seqs = valid_neg_seqs - valid_pos_set - train_pos_set - test_pos_set
    #     test_neg_seqs = test_neg_seqs - test_pos_set - train_pos_set - valid_pos_set
    #
    #     # Ensure no intersection between negative sets
    #     valid_test_neg_seqs = valid_neg_seqs & test_neg_seqs
    #     if len(test_neg_seqs) < len(valid_neg_seqs):
    #         valid_neg_seqs = valid_neg_seqs - valid_test_neg_seqs
    #     else:
    #         test_neg_seqs = test_neg_seqs - valid_test_neg_seqs
    #
    #     train_valid_neg_seqs = train_neg_seqs & valid_neg_seqs
    #     train_neg_seqs = train_neg_seqs - train_valid_neg_seqs
    #
    #     train_test_neg_seqs = train_neg_seqs & test_neg_seqs
    #     train_neg_seqs = train_neg_seqs - train_test_neg_seqs
    #
    #     return {
    #         'train_neg_seqs': np.array(list(train_neg_seqs)),
    #         'valid_neg_seqs': np.array(list(valid_neg_seqs)),
    #         'test_neg_seqs': np.array(list(test_neg_seqs))
    #     }
    #
    # def prepare_data(self, cell_type='ALL', num_test_patients=8):
    #     """
    #     Main method to prepare all data and set as class attributes.
    #
    #     Args:
    #         cell_type (str): Cell type to filter by ('CD8', 'CD4', or 'ALL')
    #         num_test_patients (int): Number of patients to use for test+validation
    #     """
    #     # Get all split data
    #     results = self.split_datasets(cell_type, num_test_patients)
    #
    #     # Set all class attributes
    #     for key, value in results.items():
    #         setattr(self, key, value)
    #
    #     return results
    #
    # def prepare_with_different_test_count(self, num_test_patients):
    #     """
    #     Update data with a different number of test patients.
    #
    #     Args:
    #         num_test_patients (int): New number of patients for test+validation
    #     """
    #     self.num_test_patients = num_test_patients
    #     return self.prepare_data(self.cell_type, num_test_patients)
