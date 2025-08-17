import os
import pandas as pd
from numpy.f2py.auxfuncs import throw_error

from utils import *
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import random
import functools
import operator
from transformers import AutoTokenizer
from trainer import build_datasets
from definitions import *
import numpy as np
import shutil


class TCRdb():
    """Curation of data from TCRdb http://bioinfo.life.hust.edu.cn/TCRdb/#/"""
    def __init__(self):
        """Constructor for TCRdb"""
        _dir = TCR_DB_PATH
        with open(os.path.join(_dir, INDEX), 'r') as f:
            _index = json.load(f)


class Study:
    def __init__(self, study_id, to_rebuild=False):
        # set name from variable name. http://stackoverflow.com/questions/1690400/getting-an-instance-name-inside-class-init
        self.name = study_id

        if to_rebuild:
            save_dir = os.path.join(STUDY_SAVE_DIR, self.name)
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir)

        try:
            if to_rebuild:
                raise Exception
            else:
                self.load()
        except:
            self._id = study_id
            self._desc = ''
            self._samples = {'usable': [], 'uncertain': [], 'background': []}
            # self._columns = {'seq': 'AASeq', 'study': 'RunId', 'study_id': 'study_id', 'patient_id': 'patient_id', 'tissue': 'tissue', 'cell_type': 'cell_type'}
            self._columns = {'seq': 'AASeq', 'v': 'Vregion', 'd': 'Dregion', 'j': 'Jregion', 'study': 'RunId'}

        self._data_path = os.path.join(os.path.dirname(__file__), TCR_DB2_PATH)

    def __str__(self):
        return str(self.__dict__)

    def __add__(self, val):
        if isinstance(val, str):
            self._samples['usable'].append(val)
            return self
        if isinstance(val, Sample):
            self._samples['usable'].append(val.sample_id)
            return self

    def __sub__(self, val):
        if isinstance(val, str):
            self._samples['background'].append(val)
            return self
        if isinstance(val, Sample):
            self._samples['uncertain'].append(val.sample_id)
            return self

    def __xor__(self, val):
        if isinstance(val, str):
            self._samples['uncertain'].append(val)
            return self
        if isinstance(val, Sample):
            self._samples['uncertain'].append(val.sample_id)
            return self

    def save(self):
        """save class as self.name.txt"""
        save_dir = os.path.join(STUDY_SAVE_DIR, self.name)
        with open(save_dir + '.txt', 'w') as file:
            json.dump(self.__dict__, file)

    def load(self):
        """try load self.name.txt"""
        load_dir = os.path.join(STUDY_SAVE_DIR, f"{self.name}.txt")
        with open(load_dir, 'r') as file:
            data = file.read()
            self.__dict__ = json.loads(data)

    def _calculate_merged_df(self, df, sample_ids, condition, is_immunoseq_data):
        # get info about each sample from the study and merge to the large df with info about each sample
        samples = [Sample(self._id, sample_id) for sample_id in sample_ids]
        samples_df = pd.DataFrame([s.__dict__ for s in samples])
        if is_immunoseq_data:
            if 'immunoSEQ21' == self._id:
                df['tissue'] = samples_df['tissue'][0]
                df['cell_type'] = samples_df['cell_type'][0]
                df['condition'] = samples_df['condition'][0]
                # make runid as patient id but split by '_' and take [1]
                df['RunId'] = df['patient_id'].apply(lambda x: x.split('_')[1] if '_' in x else x)
                df['patient_id'] = df['patient_id'].apply(lambda x: f"patient{x.split('_')[0]}" if '_' in x else x)
                merged_df = df
            elif 'immunoSEQ54' == self._id:
                df = df[df['patient_id'].str.contains('healthy_control_')]
                df['tissue'] = samples_df['tissue'][0]
                df['cell_type'] = df['patient_id'].apply(lambda x: 'CD8' if 'CD8' in x else ('CD4' if 'CD4' in x else 'Unknown'))
                df = df[df['cell_type'] != 'Unknown']
                df['condition'] = samples_df['condition'][0]
                # make runid as patient id but split by '_' and take [1]
                df['RunId'] = df['patient_id']
                df['patient_id'] = df['patient_id'].apply(lambda x: f"control{x.split('_')[2]}" if '_' in x else x)
                merged_df = df
            else:
                merged_df = df.merge(samples_df, left_on='patient_id', right_on='patient_id', how='left')
                merged_df['RunId'] = merged_df['sample_id']
                merged_df = merged_df[merged_df[self._columns['study']].isin(sample_ids)]
        else:
            temp = df[df[self._columns['study']].isin(sample_ids)]
            merged_df = temp.merge(samples_df, left_on='RunId', right_on='sample_id', how='left')
        merged_df.drop(columns=[col for col in ['study_id', 'sample_id'] if col in merged_df.columns], inplace=True)
        if condition:
            merged_df = merged_df[merged_df['condition'] == condition]
        merged_df['study_id'] = self._id
        return merged_df

    def read_sample(self, sample_ids, condition=None, ret_columns=None, data_source='tcrdb', top_percent=None, top_n_seqs=None):
        """
        :param unique: bool if True will only return unique rows. Will apply df.unique() of ret_columns only
        :param sample_ids: string or iterable of all samples ids to retrieve
        :param ret_columns: optional list or string ['Vregion' | 'Dregion' | 'Jregion' | 'AASeq' | 'cloneFraction' | 'RunId']
                            if specified will only return the given columns
        :return: DataFrame containing all records with the sample_ids Note may have duplicates
        """
        if top_percent is not None or top_n_seqs is not None:
            data_source = 'tcrdb2'  # if top_percent is specified, we use tcrdb2 data source
        # from pandas.core.common import SettingWithCopyWarning
        # warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
        if data_source == 'tcrdb':
            self._data_path = os.path.join(os.path.dirname(__file__), TCR_DB_PATH)
        elif data_source == 'tcrdb2':
            self._data_path = os.path.join(os.path.dirname(__file__), TCR_DB2_PATH)
        else:
            raise ValueError("data_source must be 'tcrdb' or 'tcrdb2'!")

        sample_ids = [sample_ids] if isinstance(sample_ids, str) else sample_ids
        if data_source == 'tcrdb':
            df = pd.read_table(os.path.join(self._data_path, f"{self._id}.tsv"))
            merged_df = self._calculate_merged_df(df, sample_ids, condition, False)
        elif data_source == 'tcrdb2':
            # df_old = pd.read_table(os.path.join(self._data_path[:-1], f"{self._id}.tsv"))
            is_immunoseq_data = 'immunoSEQ' in self._id
            df = self.load_study_df_tcrdb2(is_immunoseq_data)

            merged_df = self._calculate_merged_df(df, sample_ids, condition, is_immunoseq_data)
            # merged_df_first = self._calculate_merged_df(df, sample_ids, condition, is_immunoseq_data)

            # Filter DF
            merged_df = filter_df(merged_df, top_percent, top_n_seqs, self._data_path, self._id)

            # Check that the old df contain the same parameters as the new df:
            # print(f"Study: {self._id}")
            # merged_df_old = self._calculate_merged_df(df_old, sample_ids, condition, False)
            # for patient_id in merged_df['patient_id'].unique():
            #     patient_seqs_raw = merged_df_first[merged_df_first['patient_id'] == patient_id]
            #     # patient_seqs = merged_df[merged_df['patient_id'] == patient_id]
            #     patient_seqs_filtered = merged_df_old[merged_df_old['patient_id'] == patient_id]
            #     # intersection = set(patient_seqs['AASeq']).intersection(patient_seqs_old['AASeq'])
            #     print(f"Patient {patient_id}: Raw len {len(patient_seqs_raw['AASeq'].unique())}, Filtered len {len(patient_seqs_filtered['AASeq'].unique())}, "
            #           f"Filtered - Raw = {len(set(patient_seqs_filtered['AASeq']) - set(patient_seqs_raw['AASeq']))}.")
            # print()

            # TODO: TESTING!
            # patient_id = 'MS3'
            # patient_seqs_unfilter = merged_df_first[merged_df_first['patient_id'] == patient_id]
            # patient_seqs_filtered = merged_df[merged_df['patient_id'] == patient_id]
            # patient_seqs_old = merged_df_old[merged_df_old['patient_id'] == patient_id]
            #
            # # Intersection of old and unfiltered
            # seqs_in_old_and_unfilter = set(patient_seqs_old['AASeq']).intersection(patient_seqs_unfilter['AASeq'])
            # patient_seqs_unfilter_and_old = patient_seqs_unfilter[patient_seqs_unfilter['AASeq'].isin(seqs_in_old_and_unfilter)]
            # # sort by cloneFraction:
            # patient_seqs_unfilter_and_old = patient_seqs_unfilter_and_old.sort_values(by='cloneFraction', ascending=False)
            # smallest_clone_fraction = patient_seqs_unfilter_and_old.iloc[-1, :]

            # TODO: End TESTING.

        # return only the specified columns (or all columns if not specified)
        if ret_columns:
            if isinstance(ret_columns, str):
                ret_columns = [ret_columns]
            assert isinstance(ret_columns, list), 'ret_columns must be in [list | str]'
        else:
            ret_columns = merged_df.columns
        return merged_df[ret_columns]

    @staticmethod
    def do_tcrdb2_threshold_filtering(df, top_percent, top_n_seqs, data_path='', id='', to_save=False):
        patient_ids = df['patient_id'].unique()
        df_filtered = []
        for patient_id in patient_ids:
            df_per_patient = df[df['patient_id'] == patient_id]
            df_per_patient = Study.tcrdb2_threshold_filtering(df_per_patient, patient_id, top_percent, top_n_seqs,
                                                              data_path=data_path, id=id, to_save=to_save)
            df_filtered.append(df_per_patient)
        df = pd.concat(df_filtered, ignore_index=True)
        return df

    def load_study_df_tcrdb2(self, is_immunoseq_data):
        last_col = 'patient_id' if is_immunoseq_data else 'RunId'

        # check if there is a folder under TCRDB2_PATH named study_id:
        study_folder = os.path.join(self._data_path, self._id)
        if not os.path.isdir(study_folder):
            assert False, f"Study folder not found: {study_folder}"

        csv_files = [f for f in os.listdir(study_folder) if f.endswith(".csv")]
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {study_folder}")

        # Read all .csv files from the dir and append to study_dfs
        study_dfs = []
        for filename in csv_files:
            file_path = os.path.join(study_folder, filename)
            df = pd.read_csv(file_path)
            if is_immunoseq_data:
                if self._id == 'immunoSEQ139':
                    df['patient_id'] = filename[:-4]
                elif self._id == 'immunoSEQ21':
                    df['patient_id'] = filename[:-4]
                elif self._id == 'immunoSEQ54':
                    df['patient_id'] = filename[:-4]  # TODO: This does not work here!
                else:
                    df['patient_id'] = filename[:-4].split('-')[0].split('_')[0]
            study_dfs.append(df)
        concat_df = pd.concat(study_dfs, ignore_index=True)

        # Keep only beta chains
        if 'Chain' in concat_df.columns:
            concat_df = concat_df[concat_df['Chain'] == 'TRB'].copy()
        return concat_df[['AASeq', 'cloneFraction', 'Vregion', 'Dregion', 'Jregion', last_col]]

    def tcrdb2_filtering(self, df, patient_id):
        study_folder = os.path.join(self._data_path, self._id, "cache")
        os.makedirs(study_folder, exist_ok=True)
        save_path = os.path.join(study_folder, f"{self._id}_{patient_id}.parquet")

        # If file exists, load and validate it
        if os.path.exists(save_path):
            cached_df = pd.read_parquet(save_path)  # TODO: Fix this part of the code!
            # Check that all AASeqs in cached_df exist in current df
            input_aaseqs = set(df['AASeq'].unique())
            cached_aaseqs = set(cached_df['AASeq'].unique())
            # if not cached_aaseqs.issubset(input_aaseqs):
                # print("Cached AASeqs are not all present in the input DataFrame. Incompatible input.")
                # raise ValueError("Cached AASeqs are not all present in the input DataFrame. Incompatible input.")
            return cached_df

        # Keep only CDR3 sequences that start with C and end with F and don't contain stop codons (*)
        df = df[df['AASeq'].str.match(r'^C[ACDEFGHIKLMNPQRSTVWY]*F$')]

        # Normalize V and J region by removing alleles (e.g., TRBV7-9*01 → TRBV7-9)
        df['Vregion'] = df['Vregion'].str.extract(r'^(TRBV[\d\-]+)')
        df['Jregion'] = df['Jregion'].str.extract(r'^(TRBJ[\d\-]+)')

        # Get index of row with max cloneFraction per AASeq
        idx = df.groupby('AASeq')['cloneFraction'].idxmax()
        # Use loc to select those rows
        df_grouped = df.loc[idx].reset_index(drop=True)

        # Filter by cloneFraction ratio threshold (> 0.00001)
        # # recalculate cloneFraction as the ratio of the max cloneFraction to the sum of cloneFractions for each AASeq
        # df_grouped['cloneFraction'] = df_grouped['cloneFraction'] / df_grouped['cloneFraction'].sum()
        df_grouped = df_grouped[df_grouped['cloneFraction'] > 0.00001]

        # Ensure column order and presence
        required_cols = ['AASeq', 'cloneFraction', 'Vregion', 'Dregion', 'Jregion',
                         'RunId', 'patient_id', 'tissue', 'cell_type', 'condition', 'study_id']
        for col in required_cols:
            if col not in df_grouped.columns:
                df_grouped[col] = pd.NA

        df_grouped = df_grouped[required_cols]

        # Save to disk
        df_grouped.to_parquet(save_path, index=False)
        return df_grouped

    @staticmethod
    def tcrdb2_threshold_filtering(df, patient_id, top_percent, top_n_seqs, data_path='', id='', to_save=True):
        if data_path != '' and id != '':
            study_folder = os.path.join(data_path, id, "cache")
            os.makedirs(study_folder, exist_ok=True)
        else:
            to_save = False
        if to_save and top_n_seqs is None:
            save_path = os.path.join(study_folder, f"{id}_{patient_id}_top_{top_percent}.parquet")

            # If file exists, load and validate it
            if os.path.exists(save_path):
                cached_df = pd.read_parquet(save_path)  # TODO: Fix this part of the code!
                # Check that all AASeqs in cached_df exist in current df
                # input_aaseqs = set(df['AASeq'].unique())
                # cached_aaseqs = set(cached_df['AASeq'].unique())
                # if not cached_aaseqs.issubset(input_aaseqs):
                    # print("Cached AASeqs are not all present in the input DataFrame. Incompatible input.")
                    # raise ValueError("Cached AASeqs are not all present in the input DataFrame. Incompatible input.")
                return cached_df

        # Keep only CDR3 sequences that start with C and end with F and don't contain stop codons (*)
        df = df[df['AASeq'].str.match(r'^C[ACDEFGHIKLMNPQRSTVWY]*F$')]

        # Ensure column order and presence
        required_cols = ['AASeq', 'cloneFraction', 'Vregion', 'Dregion', 'Jregion',
                         'RunId', 'patient_id', 'tissue', 'cell_type', 'condition', 'study_id']
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan

        df = df[required_cols]

        if top_n_seqs is None and top_percent is None:
            return df

        # calculate the topxk sequences by cloneFraction
        if top_n_seqs is not None:
            topk = int(top_n_seqs * 1000)
            if len(df) < topk:
                return df
            else:
                df_top = df.nlargest(topk, 'cloneFraction')
                threshold = df_top['cloneFraction'].min()
                threshold_df = df[df['cloneFraction'] >= threshold]
                # if len(threshold_df) > topk + 10000:
                #     return df.nlargest(topk + 10000, 'cloneFraction')
                return threshold_df

        if 100 > top_percent > 0:
            threshold = df['cloneFraction'].quantile((100 - top_percent) / 100)
            df = df[df['cloneFraction'] >= threshold]

        # Save to disk
        if to_save and top_n_seqs is None:
            df.to_parquet(save_path, index=False)
        return df

    def build_train_test_classification(self, pos_examples=None, neg_examples=None, seq_identity_threshold=1.0,
                                        validation_ration=0.1, test_ratio=0.1, save=True, path=None):
        """
        splits study into untokenized train and test sets. pos_labels will be labeled with 1 and neg_labels with 0.
        :param path: str to save results default is ./
        :param save: bool whether to save results
        :param test_ratio: float [0,1]
        :param validation_ration: float [0,1]
        :param pos_labels: iterable default is 'usable' samples from study
        :param neg_labels: iterable default is 'background' samples from study
        :param seq_identity_threshold: test samples with sequence identity over the threshold will be removed float[0,1]
        :param train_ratio: float [0,1]
        :return: tain_sequences, validation_sequences, test_sequences, train_labels, validation_labels, test_labels
        """
        pos_examples = self._samples['usable'] if pos_examples is None else pos_examples
        neg_examples = self._samples['uncertain'] if neg_examples is None else neg_examples
        pos_seqs = self.read_sample(pos_examples, ret_columns=self._columns['seq']).tolist()
        neg_seqs = self.read_sample(neg_examples, ret_columns=self._columns['seq']).tolist()
        pos_labels, neg_labels = [1] * len(pos_seqs), [0] * len(neg_seqs)
        sequences, labels = pos_seqs + neg_seqs, pos_labels + neg_labels
        train_sequences, test_sequences, train_labels, test_labels = train_test_split(sequences, labels,
                                                                                      test_size=test_ratio,
                                                                                      shuffle=True)
        train_sequences, validation_sequences, train_labels, validation_labels = train_test_split(train_sequences,
                                                                                                  train_labels,
                                                                                                  test_size=validation_ration / (
                                                                                                              1 - test_ratio),
                                                                                                  shuffle=True)
        if save:
            df = pd.DataFrame()
            df['seqs'] = train_sequences
            df['labels'] = train_labels
            df.to_csv(f'{self._id}_train.csv')
            df = pd.DataFrame()
            df['seqs'] = validation_sequences
            df['labels'] = validation_sequences
            df.to_csv(f'{self._id}_validation.csv')
            df = pd.DataFrame()
            df['seqs'] = test_sequences
            df['labels'] = test_labels
            df.to_csv(f'{self._id}_test.csv')

        return train_sequences, validation_sequences, test_sequences, train_labels, validation_labels, test_labels

    def build_train_cl(self, classes=None, return_all=False):
        """
        builds training set for contrastive learning
        :param classes: list of tuples each tuple will be an augmentation class if None will use usable samples
                        as classes
        :param tokenizer: sequence sequence tokenizer default is the identity transformation
        :param return_all: if True will return all classes sefault is to leave two out for validation and test
        :return: training set
        """
        classes = [[s] for s in self._samples['usable']] if classes is None else classes
        validation_sequences, test_sequences = [], []
        if not return_all:
            assert len(classes) > 2, "not enough classes for test and validation use with return_all=True"
            validation_samples = classes.pop(random.randrange(len(classes)))
            test_samples = classes.pop(random.randrange(len(classes)))
            validation_sequences = self.read_sample(list(validation_samples), ret_columns=self._columns['seq']).tolist()
            test_sequences = self.read_sample(list(test_samples), ret_columns=self._columns['seq']).tolist()

        validation_labels, test_labels = [1] * len(validation_sequences), [1] * len(test_sequences)

        #  create list of lists of train samples
        train_sequences = [self.read_sample(list(train_samples), ret_columns=self._columns['seq']).tolist() for
                           train_samples in classes]
        train_labels = [[i] * len(seqs) for i, seqs in enumerate(train_sequences)]
        transformer = {i: seqs for i, seqs in enumerate(train_sequences)}
        #  reduce lists
        train_sequences, train_labels = shuffle(train_sequences, train_labels)
        train_sequences = functools.reduce(operator.iconcat, train_sequences, [])
        train_labels = functools.reduce(operator.iconcat, train_labels, [])

        return train_sequences, validation_sequences, test_sequences, train_labels, validation_labels, test_labels, transformer

    def build_train_representations(self, samples=None, save=True, path=None):
        """
        :param samples: iterable of Samples default is 'usable' Samples from study
        :param save: bool
        :return: pandas DataFrame
        """
        samples = self._samples['usable'] if samples is None else samples
        sequences = self.read_sample(samples, ret_columns=['AASeq'])
        sequences.drop_duplicates(inplace=True)
        path = path if path else f'{self._id}_rep_seqs.npy'
        if save:
            np.save(path, sequences)
        return sequences


class Sample:
    """holds data about individual samples in a study"""
    # def __init__(self, id, study_id, origin=''):
    def __init__(self, study_id, sample_id, patient_id='', tissue='', cell_type='', condition='', to_rebuild=False):
        # set sample_id from variable sample_id. http://stackoverflow.com/questions/1690400/getting-an-instance-name-inside-class-init
        self.study_id = study_id
        self.sample_id = sample_id
        try:
            if not to_rebuild:
                self.load()
            else:
                self.patient_id = patient_id
                self.tissue = tissue
                self.cell_type = cell_type
                self.condition = condition
                self.save()
        except:
            self.patient_id = patient_id
            self.tissue = tissue
            self.cell_type = cell_type
            self.condition = condition
            self.save()

    def save(self):
        """save class as self.study_id.txt"""
        save_dir = os.path.join(STUDY_SAVE_DIR, self.study_id, f"{self.sample_id}.txt")
        if os.path.exists(save_dir):
            os.remove(save_dir)
        if not os.path.exists(os.path.dirname(save_dir)):
            os.makedirs(os.path.dirname(save_dir))
        with open(save_dir, 'w') as file:
            json.dump(self.__dict__, file)

    def load(self):
        """try load self.study_id.txt"""
        load_dir = os.path.join(STUDY_SAVE_DIR, self.study_id, f"{self.sample_id}.txt")
        with open(load_dir, 'r') as file:
            data = file.read()
            self.__dict__ = json.loads(data)


def build_study(study_id, study_df, study_desc, usable, uncertain, background):
    """
    builds a new study entry
    :param study_id: str
    :param study_df: pandas df
    :param study_desc: str
    :return: Study
    """
    if study_id == 'PRJNA393498':
        return build_study_PRJNA393498(study_id, study_df, study_desc, usable, uncertain, background)
    if study_id == 'immunoSEQ47':
        return build_study_immunoSEQ47(study_id, study_df, study_desc, usable, uncertain, background)
    if study_id == 'immunoSEQ77':
        return build_study_immunoSEQ77(study_id, study_df, study_desc, usable, uncertain, background)
    if study_id == 'PRJNA258001':
        return build_study_PRJNA258001(study_id, study_df, study_desc, usable, uncertain, background)
    if study_id == 'PRJNA390125':
        return build_study_PRJNA390125(study_id, study_df, study_desc, usable, uncertain, background)
    if study_id == 'PRJNA495603':
        return build_study_PRJNA495603(study_id, study_df, study_desc, usable, uncertain, background)
    if study_id == 'PRJNA579190':
        return build_study_PRJNA579190(study_id, study_df, study_desc, usable, uncertain, background)
    if study_id == 'PRJNA280417':
        return build_study_PRJNA280417(study_id, study_df, study_desc, usable, uncertain, background)
    if study_id == 'PRJNA427746':
        return build_study_PRJNA427746(study_id, study_df, study_desc, usable, uncertain, background)
    if study_id == 'PRJNA318421':
        return build_study_PRJNA318421(study_id, study_df, study_desc, usable, uncertain, background)
    if study_id == 'PRJNA473147':
        return build_study_PRJNA473147(study_id, study_df, study_desc, usable, uncertain, background)
    if study_id == 'PRJNA273698':
        return build_study_PRJNA273698(study_id, study_df, study_desc, usable, uncertain, background)
    if study_id == 'immunoSEQ139':
        return build_study_immunoSEQ139(study_id, study_df, study_desc, usable, uncertain, background)
    if study_id == 'immunoSEQ21':
        return build_study_immunoSEQ21(study_id, study_df, study_desc, usable, uncertain, background)
    if study_id == 'immunoSEQ54':
        return build_study_immunoSEQ54(study_id, study_df, study_desc, usable, uncertain, background)
    if study_id == 'immunoSEQ03':
        return build_study_immunoSEQ03(study_id, study_df, study_desc, usable, uncertain, background)
    if study_id == 'immunoSEQ68':
        return build_study_immunoSEQ68(study_id, study_df, study_desc, usable, uncertain, background)
    # if study_id == 'immunoSEQ48':
    #     return build_study_immunoSEQ48(study_id, study_df, study_desc, usable, uncertain, background)
    throw_error('study_id not found!')


def build_study_PRJNA393498(study_id, study_df, study_desc, usable, uncertain, background):
    columns = ['study_id', 'sample_id', 'patient_id', 'tissue', 'cell_type']

    found_usable = []
    found_uncertain = []
    found_background = []
    study = Study(study_id, to_rebuild=True)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        condition = row['Condition']

        if tissue == 'Synovial fluid':
            patient_id = comment.split(' ')[-1].split('_')[0]
            if comment[-1] == '4' or comment[-1] == '8':
                cell_type = "CD" + comment[-1]
                sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
                study += sample
                found_usable.append(sample_id)
            elif comment.endswith('TRBV9'):
                cell_type = "Other (TRBV9)"
                sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
                study ^= sample
                found_usable.append(sample_id)
            else:
                # adding to uncertain if there is no cell type (we want only CD4 or CD8)
                cell_type = "Other"
                sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
                study += sample
                found_uncertain.append(sample_id)
        else:
            if comment[-2:] == '_4' or comment[-2:] == '_8':
                patient_id = comment.split(' ')[-1].split('_')[0]
                cell_type = "CD" + comment[-1]
                sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
                study += sample
                found_background.append(sample_id)
            else:
                cell_type = 'Other'
                if '-' in comment:
                    patient_id = comment.split(' ')[-1].split('-')[0]
                elif '_' in comment:
                    patient_id = comment.split(' ')[-1].split('_')[0]
                    cell_type = f"Other ({comment.split('_')[-1]})"
                else:
                    patient_id = comment.split(' ')[-1]
                sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
                study += sample
                found_background.append(sample_id)

    study.save()
    return study


def build_study_immunoSEQ47(study_id, study_df, study_desc, usable, uncertain, background):
    columns = ['study_id', 'sample_id', 'patient_id', 'tissue', 'cell_type']

    found_usable = []
    found_uncertain = []
    found_background = []

    study = Study(study_id, to_rebuild=True)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        cell_type = row['Cell Type'][:-1]
        patient_id = comment.split('_')[0]
        condition = row['Condition']

        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
        study += sample
        found_usable.append(sample_id)

    study.save()
    return study


def build_study_immunoSEQ77(study_id, study_df, study_desc, usable, uncertain, background):
    columns = ['study_id', 'sample_id', 'patient_id', 'tissue', 'cell_type']

    found_usable = []
    found_uncertain = []
    found_background = []

    study = Study(study_id, to_rebuild=True)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        cell_type = row['Cell Type']
        # TODO: Find out if it is correct to cluster '-' and '+' cell types together!
        if '+' in cell_type or '-' in cell_type:
            cell_type = cell_type[:-1]
        patient_id = comment
        if '-' in comment:
            patient_id = comment.split('-')[0]
        condition = row['Condition']

        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
        study += sample
        found_usable.append(sample_id)

    study.save()
    return study


def build_study_PRJNA258001(study_id, study_df, study_desc, usable, uncertain, background):
    columns = ['study_id', 'sample_id', 'patient_id', 'tissue', 'cell_type']

    found_usable = []
    found_uncertain = []
    found_background = []

    study = Study(study_id, to_rebuild=True)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        cell_type = row['Cell Type']
        condition = row['Condition']
        if '+' in cell_type:
            cell_type = cell_type[:-1]

        patient_id = comment.split(' ')[-1]
        if '_' in patient_id:
            patient_id = patient_id.split('_')[0]
        if 'v' in patient_id:
            patient_id = patient_id.split('v')[0]

        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
        study += sample
        found_usable.append(sample_id)

    study.save()
    return study


def build_study_PRJNA390125(study_id, study_df, study_desc, usable, uncertain, background):
    columns = ['study_id', 'sample_id', 'patient_id', 'tissue', 'cell_type']

    found_usable = []
    found_uncertain = []
    found_background = []

    study = Study(study_id, to_rebuild=True)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        cell_type = row['Cell Type'][:3]
        condition = row['Condition']

        patient_id = comment.split('_')[0]
        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
        study += sample
        found_usable.append(sample_id)

    study.save()
    return study


def build_study_PRJNA495603(study_id, study_df, study_desc, usable, uncertain, background):
    columns = ['study_id', 'sample_id', 'patient_id', 'tissue', 'cell_type']

    found_usable = []
    found_uncertain = []
    found_background = []

    study = Study(study_id, to_rebuild=True)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        cell_type = row['Cell Type']
        condition = row['Condition']
        if '+' in cell_type:
            cell_type = cell_type[:-1]

        import re
        def extract_pattern(s):
            match = re.search(r'_(S\d+)_', s)
            return match.group(1) if match else None
        patient_id = extract_pattern(comment)

        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
        study += sample
        found_usable.append(sample_id)

    study.save()
    return study


def build_study_PRJNA579190(study_id, study_df, study_desc, usable, uncertain, background):
    columns = ['study_id', 'sample_id', 'patient_id', 'tissue', 'cell_type']

    found_usable = []
    found_uncertain = []
    found_background = []

    study = Study(study_id, to_rebuild=True)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        cell_type = row['Cell Type']
        condition = row['Condition']
        if '+' in cell_type:
            cell_type = cell_type[:-1]

        import re
        def extract_pattern(s):
            match = re.search(r'_(\d+)_', s)
            return match.group(1) if match else None
        patient_id = extract_pattern(comment)

        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
        study += sample
        found_usable.append(sample_id)

    study.save()
    return study


def build_study_PRJNA280417(study_id, study_df, study_desc, usable, uncertain, background):
    columns = ['study_id', 'sample_id', 'patient_id', 'tissue', 'cell_type']

    found_usable = []
    found_uncertain = []
    found_background = []

    study = Study(study_id, to_rebuild=True)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        cell_type = row['Cell Type']
        condition = row['Condition']
        patient_id = comment

        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
        study += sample
        found_usable.append(sample_id)

    study.save()
    return study


def build_study_PRJNA427746(study_id, study_df, study_desc, usable, uncertain, background):
    columns = ['study_id', 'sample_id', 'patient_id', 'tissue', 'cell_type']

    found_usable = []
    found_uncertain = []
    found_background = []

    comment_counter = {}

    study = Study(study_id, to_rebuild=True)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        cell_type = row['Cell Type'].split(' ')[0]
        condition = row['Condition']
        if comment in comment_counter:
            comment_counter[comment] += 1
        else:
            comment_counter[comment] = 1

        patient_id = f"p{comment_counter[comment]}_{study_id}"

        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
        study += sample
        found_usable.append(sample_id)

    study.save()
    return study


def build_study_PRJNA318421(study_id, study_df, study_desc, usable, uncertain, background):
    columns = ['study_id', 'sample_id', 'patient_id', 'tissue', 'cell_type']

    found_usable = []
    found_uncertain = []
    found_background = []

    study = Study(study_id, to_rebuild=True)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        cell_type = row['Cell Type']
        condition = row['Condition']
        patient_id = comment.split('-')[0]

        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
        study += sample
        found_usable.append(sample_id)

    study.save()
    return study


def build_study_PRJNA473147(study_id, study_df, study_desc, usable, uncertain, background):
    columns = ['study_id', 'sample_id', 'patient_id', 'tissue', 'cell_type']

    found_usable = []
    found_uncertain = []
    found_background = []

    import re
    study = Study(study_id, to_rebuild=True)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        cell_type = row['Cell Type']
        condition = row['Condition']

        # Regex pattern to extract the substring between ' P' and '_'
        match = re.search(r' ([NP]\d+)_', comment)
        if match:
            patient_id = f"{match.group(1)}_{study_id}"
        else:
            assert False, f"Pattern not found in comment: {comment}"

        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
        study += sample
        found_usable.append(sample_id)

    study.save()
    return study


def build_study_PRJNA273698(study_id, study_df, study_desc, usable, uncertain, background):
    columns = ['study_id', 'sample_id', 'patient_id', 'tissue', 'cell_type']

    found_usable = []
    found_uncertain = []
    found_background = []

    import re
    study = Study(study_id, to_rebuild=True)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        cell_type = row['Cell Type']
        condition = row['Condition']

        # Regex pattern to extract the substring between ' P' and '_'
        match = re.search(r'Individual_[\d]+', comment)
        if match:
            patient_id = match.group(0)
        else:
            continue

        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
        study += sample
        found_usable.append(sample_id)

    study.save()
    return study


def build_study_immunoSEQ139(study_id, study_df, study_desc, usable, uncertain, background):
    found_usable = []

    import re
    study = Study(study_id, to_rebuild=True)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        cell_type = row['Cell Type']
        condition = row['Condition']
        patient_id = comment

        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
        study += sample
        found_usable.append(sample_id)

    study.save()
    return study


def build_study_immunoSEQ21(study_id, study_df, study_desc, usable, uncertain, background):
    found_usable = []

    import re
    study = Study(study_id, to_rebuild=True)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        cell_type = row['Cell Type']
        condition = row['Condition']
        patient_id = comment

        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
        study += sample
        found_usable.append(sample_id)

    study.save()
    return study


def build_study_immunoSEQ54(study_id, study_df, study_desc, usable, uncertain, background):
    found_usable = []

    import re
    study = Study(study_id, to_rebuild=True)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        cell_type = row['Cell Type']
        if cell_type[-1] == '+':
            cell_type = cell_type[:-1]
        condition = row['Condition']

        if 'Control' in comment and 'T cells' in comment:
            patient_id = sample_id
        elif 'Subject' in comment and 'T cells' in comment:
            continue
            # match = re.search(r'Subject [\d]+', comment)
            # if match:
            #     patient_id = match.group(0)
            # else:
            #     continue
        else:
            continue

        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
        study += sample
        found_usable.append(sample_id)

    study.save()
    return study


def build_study_immunoSEQ03(study_id, study_df, study_desc, usable, uncertain, background):
    found_usable = []

    import re
    study = Study(study_id, to_rebuild=True)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source'].lower()
        cell_type = row['Cell Type']
        if cell_type[-1] == '+':
            cell_type = cell_type[:-1]
        condition = row['Condition']

        patient_id = 'JIA_'+'_'.join(comment.split('_')[:2])

        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
        study += sample
        found_usable.append(sample_id)

    study.save()
    return study


def build_study_immunoSEQ68(study_id, study_df, study_desc, usable, uncertain, background):
    found_usable = []

    import re
    study = Study(study_id, to_rebuild=True)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source'].lower()
        cell_type = row['Cell Type'].split(' ')[-1]
        if cell_type[-1] == '+':
            cell_type = cell_type[:-1]
        condition = row['Condition']

        patient_id = comment.split('_')[0]

        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
        study += sample
        found_usable.append(sample_id)

    study.save()
    return study


# def build_study_immunoSEQ48(study_id, study_df, study_desc, usable, uncertain, background):
#     found_usable = []
#
#     import re
#     study = Study(study_id, to_rebuild=True)
#     study._desc = study_desc
#     for row_ind, row in study_df.iterrows():
#         sample_id = row['Sample ID']
#         comment = row['Comment']
#         tissue = row['Cell Source'].lower()
#         cell_type = row['Cell Type']#.split(' ')[-1]
#         if cell_type[-1] == '+':
#             cell_type = cell_type[:-1]
#         condition = row['Condition']
#
#         patient_id = comment#.split('_')[0]
#
#         sample = Sample(study_id, sample_id, patient_id, tissue, cell_type, condition, to_rebuild=True)
#         study += sample
#         found_usable.append(sample_id)
#
#     study.save()
#     return study


def filter_df(df, top_percent, top_n_seqs, data_path='', id=''):
    patient_ids = df['patient_id'].unique()
    df_filtered = []
    for patient_id in patient_ids:
        df_per_patient = df[df['patient_id'] == patient_id]
        df_per_patient = Study.tcrdb2_threshold_filtering(df_per_patient, patient_id, top_percent, top_n_seqs, data_path=data_path, id=id)
        # df_per_patient = self.tcrdb2_filtering(df_per_patient, patient_id)
        df_filtered.append(df_per_patient)
    df = pd.concat(df_filtered, ignore_index=True)
    return df


if __name__ == '__main__':
    study = Study('PRJNA330606')
    model_checkpoint = "facebook/esm2_t4815B__UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_sequences, validation_sequences, test_sequences, train_labels, validation_labels, test_labels, trans = study.build_train_cl()
    train, validation, test = build_datasets(tokenizer, train_sequences, validation_sequences, test_sequences,
                                             train_labels, validation_labels, test_labels)
    # train_test = calculate_distance_matrix(list(train_sequences), list(test_sequences), chunks=34000, name_to_save='PRJNA330606_test_identity')
    # train_validation = calculate_distance_matrix(list(train_sequences), list(validation_sequences), chunks=34000, name_to_save='PRJNA330606_validation_identity')

