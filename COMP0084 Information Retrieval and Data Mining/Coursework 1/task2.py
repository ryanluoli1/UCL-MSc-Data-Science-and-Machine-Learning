import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

from task1 import *


def get_vocab():
    # obtaint the vocabulary
    passage_collection = load_collection('passage-collection.txt')
    cleaned_passages = process_passage(passage_collection, remove_sw=True, lemma=True)
    freq_dict = get_freq_dict(cleaned_passages, order=True)
    vocab = [item[0] for item in freq_dict.items()]
    print('vocabulary loaded!')
    return vocab

def passage_to_id():
    # tokenize the passages and map to pid
    candidate_passages = pd.read_csv('candidate-passages-top1000.tsv', sep='\t', header=None, names=['qid','pid','query','passage'])
    cleaned_candidate_passages = process_passage(candidate_passages['passage'], remove_sw=True, lemma=True)
    pid_passage_dict = dict(zip(candidate_passages['pid'], cleaned_candidate_passages))
    print('pid-to-passage mapped!')
    return candidate_passages, pid_passage_dict

def get_inverted_index(vocab, pid_passage_dict):
    # initialization
    inverted_index = {}
    # store passage id, frequency, position information
    for pid, passages in tqdm(pid_passage_dict.items()):
        for term in passages:   
            if term not in vocab:   # skip the term is not in vocabulary
                continue
            freq = passages.count(term)
            idx = passages.index(term)
            if term not in inverted_index:
                inverted_index[term] = {pid:[freq,idx]}
            else:
                inverted_index[term].update({pid:[freq,idx]})
    return inverted_index


if __name__ == "__main__":
    
    vocab = get_vocab()
    candidate_passages, pid_passage_dict = passage_to_id()

    inverted_index = get_inverted_index(vocab, pid_passage_dict)
    # with open('inverted_index.pkl', 'wb') as f:
        # pickle.dump(inverted_index, f)