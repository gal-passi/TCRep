import random
import numpy as np
from datasets import Dataset




def CLTransformer(transformer, Y):
    """
    augment sequences using sequences from the same class
    :param transformer: dictionary {label: [sequences...]}
    :param Y: ndarray of shape (N,)
    :return:
    """
    return np.array(random.choice(transformer[y]) for y in Y)


def build_datasets(tokenizer, train_seqs, validation_seqs, test_seqs, train_labels, validation_labels, test_labels):
    """
    :param tokenizer: sequence tokenizer
    :param train_seqs: list
    :param train_labels: list
    :param validation_seqs: list
    :param validation_labels: list
    :param test_seqs: list
    :param test_labels: list
    :return: datasets.Dataset objects (train, validation, test)
    """

    train_tokenized, validation_tokenized, test_tokenized = \
        tokenizer(train_seqs), tokenizer(validation_seqs), tokenizer(test_seqs)
    train_dataset, validation_dataset, test_dataset = Dataset.from_dict(train_tokenized), \
                                                      Dataset.from_dict(validation_tokenized), \
                                                      Dataset.from_dict(test_tokenized)
    train_dataset = train_dataset.add_column("labels", train_labels)
    validation_dataset = validation_dataset.add_column("labels", validation_labels)
    test_dataset = test_dataset.add_column("labels", test_labels)

    return train_dataset, validation_dataset, test_dataset


def create_sequence_embeddings(sequences, max_seq_len, embedding_model, batch_converter, device, batch_size=1024):
    """
    :param sequences: sequence
    :param max_seq_len:
    :param embedding_model:
    :param batch_converter:
    :param device:
    :param batch_size:
    :return: np array of shape(n_sequences, max_seq_length, esm_features_num)
    """
    embedded_data = np.zeros((len(sequences), max_seq_len, ESM_FEATURE_NUM))
    n_batches = (len(sequences) // batch_size) + 1
