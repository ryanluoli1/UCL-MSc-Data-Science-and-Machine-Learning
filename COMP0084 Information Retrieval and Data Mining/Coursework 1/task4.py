import numpy as np
import pandas as pd
from IPython.display import display

from task1 import *
from task2 import *
from task3 import *


def implement_language_model(test_queries, qid_query_dict, pid_passage_dict, inverted_index, model_params, top=100):
    # initialization
    df = pd.DataFrame()
    qids, pids, scores = np.array([]), np.array([]), np.array([])
    # vocabulary size of passages
    V = len(inverted_index)
    # total number of terms in passages
    N = np.sum([len(passage) for pid, passage in pid_passage_dict.items()])
    for qid, query in tqdm(test_queries.values):
        query = qid_query_dict[qid]
        qids_i, pids_i, scores_i = np.array([]), np.array([]), np.array([])
        for pid in candidate_passages[candidate_passages['qid']==qid]['pid']:
            passage = pid_passage_dict[pid]
            D = len(passage)    # passage length
            score = 0
            for term in query:
                # term frequency in passage and total occurance
                m = inverted_index[term][pid][0] if term in passage else 0
                # compute the score
                if model_params['model'] == 'laplace':
                    num = (m+1)
                    dom = (D+V)
                elif model_params['model'] == 'lidstone':
                    num = (m+model_params['epsilon'])
                    dom = (D+V*model_params['epsilon'])
                elif model_params['model'] == 'dirichlet':
                    # compute the background probability of the term
                    if term in inverted_index:
                        cqi = np.sum([v[0] for k,v in inverted_index[term].items()])
                    else:
                        cqi = 0
                    p_term = cqi / N
                    num = (m+model_params['mu']*p_term)
                    dom = (D+model_params['mu'])
                # check whether there exists a zero probability
                if num/dom == 0:
                    score = -float('inf')
                    break
                else:
                    score += np.log(num/dom)
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

    # obtain the inverted_index of the passages
    vocab = get_vocab()
    candidate_passages, pid_passage_dict = passage_to_id()
    inverted_index = get_inverted_index(vocab, pid_passage_dict)

    # obtain the qid to processed query mapping
    test_queries, qid_query_dict = query_to_id()

    # implement laplace smoothing
    model_params = {'model': 'laplace',
                    'epsilon': 0.1,
                    'mu': 50}
    df_laplace = implement_language_model(test_queries, qid_query_dict, pid_passage_dict, inverted_index, model_params, top=100)
    print('laplaced smoothing: ')
    display(df_laplace.head())
    df_laplace.to_csv('laplace.csv', header=False, sep=',', index=False)

    # implement lidstone correction 
    model_params = {'model': 'lidstone',
                    'epsilon': 0.1,
                    'mu': 50}
    df_lidstone = implement_language_model(test_queries, qid_query_dict, pid_passage_dict, inverted_index, model_params, top=100)
    print('lidstone correlation: ')
    display(df_lidstone.head())
    df_lidstone.to_csv('lidstone.csv', header=False, sep=',', index=False)

    # implement dirichlet smoothing
    model_params = {'model': 'dirichlet',
                    'epsilon': 0.1,
                    'mu': 50}
    df_dirichlet = implement_language_model(test_queries, qid_query_dict, pid_passage_dict, inverted_index, model_params, top=100)
    print('dirichlet smoothing: ')
    display(df_dirichlet.head())
    df_dirichlet.to_csv('dirichlet.csv', header=False, sep=',', index=False)