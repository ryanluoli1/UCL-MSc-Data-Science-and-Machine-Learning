import pickle
import numpy as np
import matplotlib.pyplot as plt

import time
import pickle 
from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


def load_collection(file_path):
    with open(file_path) as f:
        # remove the white space at the start and end of each line
        passage_collection = [x.strip() for x in f.readlines()]
    return passage_collection

def process_passage(passage_collection, remove_sw=False, lemma=False):
    # initialization
    if remove_sw:
        stop_words = stopwords.words('english')
    if lemma:
        lemmatizer = WordNetLemmatizer()
    tokenizer = RegexpTokenizer(r'\s+', gaps=True)
    cleaned_passages = []
    for passage in passage_collection:
        # tokenization, lowercase, remove non-alphabetic tokens (numbers, punctuations, etc.)
        tokens = [token.lower() for token in tokenizer.tokenize(passage) if token.isalpha()]
        # remove stopwords if required
        if remove_sw:
            tokens = [token for token in tokens if token not in stop_words]
        # lemmatization if required
        passage_cleaned = []
        if lemma:
            for token in tokens:
                passage_cleaned.append(lemmatizer.lemmatize(token))
        else:
            passage_cleaned = tokens
        cleaned_passages.append(passage_cleaned)
    return cleaned_passages

def get_freq_dict(cleaned_passages, order=True):
    flatten_passages = [word for passage in cleaned_passages for word in passage]
    freq_dict = dict(Counter(flatten_passages))
    freq_dict = dict(sorted(freq_dict.items(), key=lambda item: item[1], reverse=order))
    return freq_dict

def get_zipfian(vocab_size, s=1):
    I = [1/(i**(s)) for i in range(1, vocab_size+1)]
    total_I = np.sum(I)
    F = [(1/(k**(s)))/total_I for k in range(1,vocab_size+1)]
    return F


if __name__ == "__main__":
    # load the collection of passages
    file_path = 'passage-collection.txt'
    passage_collection = load_collection(file_path)

    # preprocess the passages (without removing stop words and lemmatization)
    cleaned_passages = process_passage(passage_collection, remove_sw=False, lemma=False)

    # obtain the frequecy dictionary
    freq_dict = get_freq_dict(cleaned_passages, order=True)

    # obtain the vocabulary
    vocab = [item[0] for item in freq_dict.items()]
    with open('vocabulary.pkl', 'wb') as f:
        pickle.dump(vocab, f)
    vocab_size = len(vocab)
    print('vocabulary size: ', vocab_size)

    # compute the normalized frequency
    freq = [item[1] for item in freq_dict.items()]
    n_words = np.sum(freq)
    norm_freq = [f/n_words for f in freq]

    # rank of occurance
    rank = np.arange(1, vocab_size+1)

    # plot the empirical distribution
    plt.figure(figsize=(6,4))
    plt.plot(rank[:1000], norm_freq[:1000])
    plt.title('Emperical Distribution')
    plt.xlabel('Frequency Ranking')
    plt.ylabel('Normalized Frequency')
    plt.show()

    # log-log plot comparing the empirial and the zipfian distribution
    F = get_zipfian(vocab_size, s=1)
    plt.plot(np.log10(rank), np.log10(F), label='Zipfian')
    plt.scatter(np.log10(rank), np.log10(norm_freq), color='r', alpha=0.3, label='Empirical')
    plt.title('Empirical vs Zipfian')
    plt.xlabel('Frequency Ranking (log)')
    plt.ylabel('Normalized Frequency (log)')
    plt.legend()
    plt.show()

    # plot the difference
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1,51), (np.array(norm_freq)-np.array(F))[:50], label='Emperical-Zipfian')
    plt.legend()
    plt.title('Emperical vs Zipfian')
    plt.xlabel('Frequency Ranking')
    plt.ylabel('Normalized Frequency')
    plt.show()

    # remove stop words
    cleaned_passages_sw = process_passage(passage_collection, remove_sw=True, lemma=False)
    freq_dict_sw = get_freq_dict(cleaned_passages_sw, order=True)
    freq_sw = [item[1] for item in freq_dict_sw.items()]
    n_words_sw = np.sum(freq_sw)
    norm_freq_sw = [f/n_words_sw for f in freq_sw]
    vocab_size_sw = len(freq_dict_sw)
    rank_sw = np.arange(1, vocab_size_sw+1)
    F_sw = get_zipfian(vocab_size_sw, s=1)
    plt.plot(np.log10(rank_sw), np.log10(F_sw), label='Zipfian')
    plt.scatter(np.log10(rank_sw), np.log10(norm_freq_sw), color='r', alpha=0.3, label='Empirical')
    plt.title('Empirical vs Zipfian (stop words removed)')
    plt.xlabel('Frequency Ranking (log)')
    plt.ylabel('Normalized Frequency (log)')
    plt.legend()
    plt.show()
    plt.figure(figsize=(6,4))
    plt.plot(np.arange(1,51), (np.array(norm_freq_sw)-np.array(F_sw))[:50], label='Emperical-Zipfian')
    plt.legend()
    plt.title('Emperical vs Zipfian (stop words removed)')
    plt.xlabel('Frequency Ranking')
    plt.ylabel('Normalized Frequency')
    plt.show()