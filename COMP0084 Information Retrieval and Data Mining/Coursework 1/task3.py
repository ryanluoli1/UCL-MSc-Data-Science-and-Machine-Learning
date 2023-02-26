import numpy as np
import pandas as pd
from IPython.display import display

from task1 import *
from task2 import *


def get_passage_tf_idf(pid_passage_dict, inverted_index):
    passage_tf_idf = {}
    N_doc = len(pid_passage_dict)
    for pid, passage in tqdm(pid_passage_dict.items()):
        pid_tf_idf = {}
        for term in passage:
            freq = inverted_index[term][pid][0]
            tf = freq / len(passage)
            n_term = len(inverted_index[term])
            idf = np.log10(N_doc/n_term)
            pid_tf_idf[term] = [tf, idf, tf*idf]
        passage_tf_idf[pid] = pid_tf_idf
    return passage_tf_idf

def query_to_id():
    test_queries = pd.read_csv('test-queries.tsv', sep='\t', header=None, names=['qid','query'])
    cleaned_test_queries = process_passage(test_queries['query'], remove_sw=True, lemma=True)
    qid_query_dict = dict(zip(test_queries['qid'], cleaned_test_queries))
    return test_queries, qid_query_dict

def get_query_tf_idf(qid_query_dict, query_inverted_index, vocab, query_vocab, inverted_index, passage_tf_idf):
    query_tf_idf = {}
    N_doc = len(qid_query_dict)
    for qid, query in tqdm(qid_query_dict.items()):
        qid_tf_idf = {}
        for term in query:
            if term not in vocab:
                continue
            if term not in query_vocab:
                continue
            freq = query_inverted_index[term][qid][0]
            tf = freq / len(query)
            pid = list(inverted_index[term].keys())[0]
            idf_passage = passage_tf_idf[pid][term][1]
            qid_tf_idf[term] = [tf, idf_passage, tf*idf_passage]
        query_tf_idf[qid] = qid_tf_idf
    return query_tf_idf

def cosine_similarity(v_query, v_passage):
    terms = set(v_query.keys()).intersection(set(v_passage.keys()))
    inner_product = 0.0
    for term in terms:
        inner_product += v_query[term][2] * v_passage[term][2]
    X = np.linalg.norm([v[2] for k,v in v_query.items()])
    Y = np.linalg.norm([v[2] for k,v in v_passage.items()])
    if X == 0 or Y == 0:
        return 0
    else:
        return inner_product / (X * Y)

def vector_space_model(test_queries, candidate_passages, query_tf_idf, passage_tf_idf, top=100):
    df = pd.DataFrame()
    qids, pids, scores = np.array([]), np.array([]), np.array([])
    for qid, query in tqdm(test_queries.values):
        qids_i, pids_i, scores_i = np.array([]), np.array([]), np.array([])
        for pid in candidate_passages[candidate_passages['qid']==qid]['pid']:
            v_query = query_tf_idf[qid]
            v_passage = passage_tf_idf[pid]
            score = cosine_similarity(v_query, v_passage)
            qids_i = np.append(qids_i, qid)
            pids_i = np.append(pids_i, pid)
            scores_i = np.append(scores_i, score)
        descending_idx = np.argsort(scores_i)[::-1]
        qids_i = qids_i[descending_idx]
        pids_i = pids_i[descending_idx]
        scores_i = scores_i[descending_idx]
        qids = np.append(qids, qids_i[:top])
        pids = np.append(pids, pids_i[:top])
        scores = np.append(scores, scores_i[:top])
    df['qid'] = qids
    df['pid'] = pids
    df['score'] = scores
    df['qid'] = df['qid'].astype(int)
    df['pid'] = df['pid'].astype(int)
    return df

def BM25_model(query_params, passage_params, model_params, top=100):
    test_queries, qid_query_dict, q_inverted_index = query_params[0], query_params[1], query_params[2]
    candidate_passages, pid_passage_dict, p_inverted_index = passage_params[0], passage_params[1], passage_params[2]
    k1, k2, b = model_params[0], model_params[1], model_params[2]
    N = len(pid_passage_dict)
    avdl = np.sum([len(passage) for pid, passage in pid_passage_dict.items()]) / len(pid_passage_dict)
    df = pd.DataFrame()
    qids, pids, scores = np.array([]), np.array([]), np.array([])
    for qid, _ in tqdm(test_queries.values):
        query = qid_query_dict[qid]
        qids_i, pids_i, scores_i = np.array([]), np.array([]), np.array([])
        for pid in candidate_passages[candidate_passages['qid']==qid]['pid']:
            passage = pid_passage_dict[pid]
            dl = len(passage)
            K = k1 * ((1-b) + b*dl/avdl)
            score = 0
            for term in query:
                qfi = query.count(term)
                if term in passage:
                    fi = p_inverted_index[term][pid][0]
                    ni = len(inverted_index[term])
                else:
                    fi = 0
                    ni = 0
                score += np.log(((0+0.5)/(0-0+0.5))/((ni-0+0.5)/(N-ni-0+0+0.5)))*((k1+1)*fi/((K+fi)))*((k2+1)*qfi/(k2+qfi))
            qids_i = np.append(qids_i, qid)
            pids_i = np.append(pids_i, pid)
            scores_i = np.append(scores_i, score)
        descending_idx = np.argsort(scores_i)[::-1]
        qids_i = qids_i[descending_idx]
        pids_i = pids_i[descending_idx]
        scores_i = scores_i[descending_idx]
        qids = np.append(qids, qids_i[:top])
        pids = np.append(pids, pids_i[:top])
        scores = np.append(scores, scores_i[:top])
    df['qid'] = qids
    df['pid'] = pids
    df['score'] = scores
    df['qid'] = df['qid'].astype(int)
    df['pid'] = df['pid'].astype(int)
    return df


if __name__ == "__main__":

    # obatin the inverted_index for passages
    vocab = get_vocab()
    candidate_passages, pid_passage_dict = passage_to_id()
    inverted_index = get_inverted_index(vocab, pid_passage_dict)

    # obtain the TF-IDF transformation of passages
    passage_tf_idf = get_passage_tf_idf(pid_passage_dict, inverted_index)
    print('Passages TF-IDF obtained!')

    # obatain the inverted_index for queries
    test_queries, qid_query_dict = query_to_id()
    terms = [v for k, v in qid_query_dict.items()]
    query_vocab = list(set([item for sublist in terms for item in sublist]))
    query_inverted_index = get_inverted_index(query_vocab, qid_query_dict)

    # obtain the TF-IDF transformation of queries
    query_tf_idf = get_query_tf_idf(qid_query_dict, query_inverted_index, vocab, query_vocab, inverted_index, passage_tf_idf)
    print('Queries TF-IDF obtained!')

    # implement the vector space model with cosine similarity scoring
    df_tf_idf = vector_space_model(test_queries, candidate_passages, query_tf_idf, passage_tf_idf, top=100)
    display(df_tf_idf.head())
    df_tf_idf.to_csv('tfidf.csv', header=False, sep=',', index=False)
    print('Top 100 vector space model retrieved passages saved!')

    # implement the BM25 mdeol
    query_params = [test_queries, qid_query_dict, query_inverted_index]
    passage_params = [candidate_passages, pid_passage_dict, inverted_index]
    model_params = [1.2, 100, 0.75]
    df_bm25 = BM25_model(query_params, passage_params, model_params, top=100)
    display(df_bm25.head())
    df_bm25.to_csv('bm25.csv', header=False, sep=',', index=False)
    print('Top 100 BM25 model retrieved passages saved!')