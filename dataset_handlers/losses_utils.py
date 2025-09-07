import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import hashlib
from collections import Counter
from dataset_handlers.constants import *
from utils.utils import pairwise_scores, levenshtein_dist_non_bin


class LossPreprocessor:
    def __init__(self, df_bld, df_hlt, train_pos_seqs, train_patient_ids, test_patient_ids, valid_patient_ids,
                 filter_num_of_patients, ratio, dataset_type, dist_loss_type, use_nneighbors_loss, cache_path):
        self.aaseq_to_ratio = None
        self.aaseq_to_distance = None
        self.aaseq_to_nneighbors = None
        self.cache_path = cache_path

        if ratio and dataset_type not in ['article', 'article_sle']:
            self.aaseq_to_ratio = self.build_aaseq_to_ratio_func(df_bld, df_hlt, dataset_type)

        if dist_loss_type != 'none':
            self.aaseq_to_distance = self.build_aaseq_to_distance_func(df_bld, df_hlt, train_pos_seqs, dataset_type, dist_loss_type)

        if use_nneighbors_loss:
            self.aaseq_to_nneighbors = self.build_aaseq_to_nneighbors_func(
                df_bld,
                train_patient_ids,
                test_patient_ids,
                valid_patient_ids,
                dist_loss_type,
                filter_num_of_patients=filter_num_of_patients
            )

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

    def build_aaseq_to_ratio_func(self, df_bld, df_hlt):
        """Compute ratio of disease to healthy frequency for each AA sequence."""

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
        return aaseq_to_ratio

    def build_distance_df(self, df_bld, train_pos_seqs, dataset_type='ms', batch_size=1000):
        sorted_seqs = sorted(list(df_bld["AASeq"].unique()))
        # create hash of the df_dist in order to save or load it

        hash_str = "_".join(sorted_seqs)
        hash_of_df = hashlib.sha256(hash_str.encode()).hexdigest()

        # check if the df_dist already exists
        df_dist_filename = os.path.join(self.cache_path, f"distance_df_cache/{dataset_type}/df_dist_{hash_of_df}.pkl")
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

    def build_aaseq_to_distance_func(self, df_bld, df_hlt, train_pos_seqs, dataset_type, dist_loss_type):
        """Compute distance metrics between sequences (e.g., Levenshtein)."""

        self.build_distance_df(df_bld, train_pos_seqs, dataset_type)

        def distance_func(x, a=2.0, k=3.5, x_0=2.0, b=1.0):
            sig = 1 - 1 / (1 + torch.exp(-k * (x - x_0)))
            return a * sig + b

        self._dist_a = {"none": 0, "v1": 1, "v2": 2, "v3": 4, "v4": 6}[dist_loss_type.lower()]

        def aaseq_to_distance(aaseq_array, default_value=1.0, dont_use_function=False):
            lookup_series = self.df_aaseq_to_distance.set_index('AASeq')['distance']
            result = pd.Series(aaseq_array).map(lookup_series).fillna(default_value)
            return result if dont_use_function else distance_func(torch.tensor(result), a=self._dist_a)
        return aaseq_to_distance

    def build_nnegihbours_df(self, dfs):
        """
            Builds self.df_aaseq_to_nneighbors: a DataFrame with unique AASeqs and their
            number of neighbors (i.e., how many times they appear) across a list of DataFrames.
            Counts are aggregated across all provided DataFrames, but duplicates within a
            single patient in a DataFrame are only counted once.

            Args:
                dfs (list of pd.DataFrame): Each DataFrame must have columns ['patient_id', 'AASeq']
        """
        # Step 1: Count unique AASeqs per patient in each DF
        all_counts = Counter()
        for df in dfs:
            # For each patient, get the unique AASeqs and count each one once per patient
            patient_groups = df.groupby('patient_id')['AASeq'].unique()
            for aaseq_list in patient_groups:
                all_counts.update(aaseq_list)

        # Step 2: Convert Counter to DataFrame
        self.df_aaseq_to_nneighbors = (
            pd.DataFrame.from_dict(all_counts, orient='index', columns=['nneighbors'])
            .reset_index()
            .rename(columns={'index': 'AASeq'})
        )

    def build_aaseq_to_nneighbors_func(self, df_bld, train_patient_ids, test_patient_ids, valid_patient_ids,
                                       loss_version, filter_num_of_patients):
        """Compute nearest neighbors for each sequence."""

        # Option for AASeq to nneighbors incorporated into loss:
        self.build_nnegihbours_df([df_bld[df_bld['patient_id'].isin(train_patient_ids)],
                                   df_bld[df_bld['patient_id'].isin(test_patient_ids)],
                                   df_bld[df_bld['patient_id'].isin(valid_patient_ids)]])

        def nneighbors_func(x, gamma, k, a=0.7):
            # return gamma ** (x - k)
            m = gamma
            return m - (m - 1) * (2 / (1 + torch.exp(-a * (x - k))))

        # gamma_options = {0 : 1.2, 1 : 1.3, 2 : 1.4, 3 : 1.5, 4 : 1.6}
        gamma_options = {0: 1.0, 1: 0.5, 2: 0.0, 3: -1.0, 4: -2.0}
        default_gamma = gamma_options[loss_version] if loss_version in gamma_options else gamma_options[0]

        def aaseq_to_nneighbors(aaseq_array, default_value=filter_num_of_patients, gamma=default_gamma):
            lookup_series = self.df_aaseq_to_nneighbors.set_index('AASeq')['nneighbors']
            result = torch.tensor(pd.Series(aaseq_array).map(lookup_series).fillna(default_value).values)
            result = nneighbors_func(result, gamma=gamma, k=default_value)
            result[result < 1.0] = 1.0
            return result
        return aaseq_to_nneighbors
