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
    def __init__(self, study_id):
        # set name from variable name. http://stackoverflow.com/questions/1690400/getting-an-instance-name-inside-class-init
        # TODO: name is basically _id which is actually study_id, so why not just call it study_id?
        self.name = study_id

        try:
            self.load()
        except:
            self._id = study_id
            save_dir = os.path.join(STUDY_SAVE_DIR, self.name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            else:
                shutil.rmtree(save_dir)
                os.makedirs(save_dir)
            self._desc = ''
            self._samples = {'usable': [], 'uncertain': [], 'background': []}
            # self._columns = {'seq': 'AASeq', 'study': 'RunId', 'study_id': 'study_id', 'patient_id': 'patient_id', 'tissue': 'tissue', 'cell_type': 'cell_type'}
            self._columns = {'seq': 'AASeq', 'v': 'Vregion', 'd': 'Dregion', 'j': 'Jregion', 'study': 'RunId'}

        self._data_path = os.path.join(os.path.dirname(__file__), TCR_DB_PATH)

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

    def read_sample(self, sample_ids, ret_columns=None):
        """
        :param unique: bool if True will only return unique rows. Will apply df.unique() of ret_columns only
        :param sample_ids: string or iterable of all samples ids to retrieve
        :param ret_columns: optional list or string ['Vregion' | 'Dregion' | 'Jregion' | 'AASeq' | 'cloneFraction' | 'RunId']
                            if specified will only return the given columns
        :return: DataFrame containing all records with the sample_ids Note may have duplicates
        """
        sample_ids = [sample_ids] if isinstance(sample_ids, str) else sample_ids
        df = pd.read_table(os.path.join(self._data_path, f"{self._id}.tsv"))

        # get info about each sample from the study and merge to the large df with info about each sample
        samples = [Sample(self._id, sample_id) for sample_id in sample_ids]
        samples_df = pd.DataFrame([s.__dict__ for s in samples])
        temp = df[df[self._columns['study']].isin(sample_ids)]
        merged_df = temp.merge(samples_df, left_on='RunId', right_on='sample_id', how='left')
        merged_df.drop(columns=['study_id', 'sample_id'], inplace=True)

        # return only the specified columns (or all columns if not specified)
        if ret_columns:
            if isinstance(ret_columns, str):
                ret_columns = [ret_columns]
            assert isinstance(ret_columns, list), 'ret_columns must be in [list | str]'
        else:
            ret_columns = merged_df.columns
        return merged_df[ret_columns]

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
    def __init__(self, study_id, sample_id, patient_id='', tissue='', cell_type=''):
        # set sample_id from variable sample_id. http://stackoverflow.com/questions/1690400/getting-an-instance-name-inside-class-init
        self.study_id = study_id
        self.sample_id = sample_id
        try:
            self.load()
        except:
            self.patient_id = patient_id
            self.tissue = tissue
            self.cell_type = cell_type
            self.save()

    def save(self):
        """save class as self.study_id.txt"""
        save_dir = os.path.join(STUDY_SAVE_DIR, self.study_id, f"{self.sample_id}.txt")
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
    throw_error('study_id not found!')


def build_study_PRJNA393498(study_id, study_df, study_desc, usable, uncertain, background):
    columns = ['study_id', 'sample_id', 'patient_id', 'tissue', 'cell_type']

    found_usable = []
    found_uncertain = []
    found_background = []

    study = Study(study_id)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        if tissue == 'Synovial fluid':
            patient_id = comment.split(' ')[-1].split('_')[0]
            if comment[-1] == '4' or comment[-1] == '8':
                cell_type = "CD" + comment[-1]
                sample = Sample(study_id, sample_id, patient_id, tissue, cell_type)
                study += sample
                found_usable.append(sample_id)
            elif comment.endswith('TRBV9'):
                cell_type = "Other (TRBV9)"
                sample = Sample(study_id, sample_id, patient_id, tissue, cell_type)
                study ^= sample
                found_usable.append(sample_id)
            else:
                # adding to uncertain if there is no cell type (we want only CD4 or CD8)
                cell_type = "Other"
                sample = Sample(study_id, sample_id, patient_id, tissue, cell_type)
                study ^= sample
                found_uncertain.append(sample_id)
        else:
            if comment[-2:] == '_4' or comment[-2:] == '_8':
                patient_id = comment.split(' ')[-1].split('_')[0]
                cell_type = "CD" + comment[-1]
                sample = Sample(study_id, sample_id, patient_id, tissue, cell_type)
                study -= sample
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
                sample = Sample(study_id, sample_id, patient_id, tissue, cell_type)
                study -= sample
                found_background.append(sample_id)

    study.save()
    return study


def build_study_immunoSEQ47(study_id, study_df, study_desc, usable, uncertain, background):
    columns = ['study_id', 'sample_id', 'patient_id', 'tissue', 'cell_type']

    found_usable = []
    found_uncertain = []
    found_background = []

    study = Study(study_id)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        cell_type = row['Cell Type'][:-1]
        patient_id = comment.split('_')[0]

        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type)
        study += sample
        found_usable.append(sample_id)

    study.save()
    return study


def build_study_immunoSEQ77(study_id, study_df, study_desc, usable, uncertain, background):
    columns = ['study_id', 'sample_id', 'patient_id', 'tissue', 'cell_type']

    found_usable = []
    found_uncertain = []
    found_background = []

    study = Study(study_id)
    study._desc = study_desc
    for row_ind, row in study_df.iterrows():
        sample_id = row['Sample ID']
        comment = row['Comment']
        tissue = row['Cell Source']
        cell_type = row['Cell Type']
        if '+' in cell_type:
            cell_type = cell_type[:-1]
        patient_id = comment
        condition = row['Condition']

        sample = Sample(study_id, sample_id, patient_id, tissue, cell_type)
        if condition == 'Healthy':
            study += sample
            found_usable.append(sample_id)
        else:
            study ^= sample
            found_uncertain.append(sample_id)

    study.save()
    return study


if __name__ == '__main__':
    study = Study('PRJNA330606')
    model_checkpoint = "facebook/esm2_t4815B__UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    train_sequences, validation_sequences, test_sequences, train_labels, validation_labels, test_labels, trans = study.build_train_cl()
    train, validation, test = build_datasets(tokenizer, train_sequences, validation_sequences, test_sequences,
                                             train_labels, validation_labels, test_labels)
    # train_test = calculate_distance_matrix(list(train_sequences), list(test_sequences), chunks=34000, name_to_save='PRJNA330606_test_identity')
    # train_validation = calculate_distance_matrix(list(train_sequences), list(validation_sequences), chunks=34000, name_to_save='PRJNA330606_validation_identity')

'''
    build_study('PRJNA273698', "T-cell Receptor Beta Chain sequences from blood of systemically healthy individuals of various ages", ['AASeq', 'Vregion', 'Dregion', 'Jregion', 'RunId'],
                [],
                [],
                [['SRR1777800', 'blood'], ['SRR1777798', 'blood'], ['SRR1777796', 'blood'], ['SRR1777792', 'blood'], ['SRR1777788', 'blood'], ['SRR1777784', 'blood'], ['SRR1777779', 'blood'], ['SRR1777776', 'blood'], ['SRR1777775', 'blood'], ['SRR1777774', 'blood'], ['SRR1777772', 'blood'], ['SRR1777770', 'blood'], ['SRR1777769', 'blood'], ['SRR1777765', 'blood'], ['SRR1777764', 'blood'], ['SRR1777766', 'blood'], ['SRR1777767', 'blood'], ['SRR1777768', 'blood'], ['SRR1777771', 'blood'], ['SRR1777778', 'blood'], ['SRR1777781', 'blood'], ['SRR1777785', 'blood'], ['SRR1777786', 'blood'], ['SRR1777787', 'blood'], ['SRR1777789', 'blood'], ['SRR1777790', 'blood'], ['SRR1777791', 'blood'], ['SRR1777795', 'blood'], ['SRR1777797', 'blood'], ['SRR1777799', 'blood'], ['SRR1777801', 'blood'], ['SRR1777802', 'blood'], ['SRR1777803', 'blood'], ['SRR1777773', 'blood'], ['SRR1777777', 'blood'], ['SRR1777793', 'blood'], ['SRR1777794', 'blood'], ['SRR1777804', 'blood']]
)
'''
