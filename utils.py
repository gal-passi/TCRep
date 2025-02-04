from Levenshtein import ratio as levenshtein_ratio
from scipy.spatial.distance import pdist
import numpy as np
import math
from Bio import pairwise2
from Bio.Seq import Seq
from multiprocessing import cpu_count, Pool
from itertools import combinations_with_replacement, product
import json
import requests
from requests.adapters import HTTPAdapter, Retry


def seq_identity(seq_a, seq_b):
     return pairwise2.align.globalxx(seq_a, seq_b, score_only=True) / float(max(len(seq_a), len(seq_b)))


def format_runids(runids, desc=''):
    return [[r, d] for r, d in zip(runids, [desc] * len(runids))]


def calculate_distance_matrix(train_seq, test_seq, matric=seq_identity, chunks = 10000, name_to_save=''):
    """
    calculates pairwise distance between sequences
    :param sequences_1: list of sequences
    :param sequences_2: list of sequence
    :param matric: distance matrix f(string, string) --> float
    :param chunks: for large sets will perform the proces in chunks of chunk size
    :return: ndarray of shape (len(sequences_1), len(sequences_1))
    """
    chunks = max(len(train_seq), len(test_seq)) if not chunks else chunks
    overlapping_seqs = {1: [], 0.95: [], 0.9: [], 0.85: [], 0.8: []}
    n_1 = (len(train_seq) // chunks) + math.ceil((len(train_seq) % chunks) / chunks)
    n_2 = (len(test_seq) // chunks) + math.ceil((len(test_seq) % chunks) / chunks)
    print(f"n1: {n_1}, n2: {n_2}")
    # break process into chunks
    for i in range(n_1):
        for j in range(n_2):
            # preform distance calculation
            D = pairwise_scores(train_seq[chunks*i: chunks*(i+1)], test_seq[chunks*j: chunks*(j+1)], score=matric)

            # remove samples over threshold
            for thr in overlapping_seqs.keys():
                values = np.argwhere(D >= thr)[:, 1] + (chunks*j)
                overlapping_seqs[thr] += list(np.unique(values))
            del D

    # with open(f'{name_to_save}.txt', 'w') as file:
    #     json.dump(overlapping_seqs, file)

    return overlapping_seqs


def pairwise_scores(group_a, group_b=None, score=seq_identity, workers=0):
    """
    """
    workers = workers if workers else cpu_count()

    # pairwise within group
    if group_b is None:
        pairs_to_check = combinations_with_replacement(group_a, 2)
        if workers == 1:
            scores = [score(*pair) for pair in pairs_to_check]
        else:
            with Pool(workers) as p:
                scores = p.starmap(score, pairs_to_check)

        pair_scores = np.zeros((len(group_a), len(group_a)))
        pair_scores[np.triu_indices_from(pair_scores)] = scores
        pair_scores = pair_scores + np.tril(pair_scores.T, -1)

    # pairwise between two groups
    else:
        pairs_to_check = product(group_a, group_b)
        if workers == 1:
            pair_scores = np.array([score(*pair) for pair in pairs_to_check]).reshape((len(group_a), len(group_b)))
        else:
            with Pool(workers) as p:
                pair_scores = np.array(p.starmap(score, pairs_to_check)).reshape((len(group_a), len(group_b)))
    return pair_scores


def create_session(header, retries=5, wait_time=0.5, status_forcelist=None):
    """
    Creates a session using pagination
    :param header: str url header session eill apply to
    :param retries: int number of retries on failure
    :param wait_time: float time (sec) between attempts
    :param status_forcelist: list HTTP status codes that we should force a retry on
    :return: requests session
    """
    s = requests.Session()
    retries = Retry(total=retries,
                    backoff_factor=wait_time,
                    status_forcelist=status_forcelist)

    s.mount(header, HTTPAdapter(max_retries=retries))
    return s
