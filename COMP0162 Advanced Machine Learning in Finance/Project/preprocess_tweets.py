import os
import csv
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import warnings
warnings.filterwarnings('ignore')


if __name__ == '__main__':

    tweet_path = 'Data/stocknet dataset/tweet/preprocessed'

    stocks, dates, tweets = [], [], []
    for stock in tqdm(os.listdir(tweet_path)):
        if stock == '.DS_Store':
            continue
        folder_path = tweet_path + '/' + stock
        for file_name in os.listdir(folder_path):
            if '(1)' in file_name:
                print(file_name)
            tweet = ''
            file_path = folder_path + '/' + file_name
            with open(file_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    tweet += ' '.join(data['text']) + '<sep>'
            stocks.append(stock)
            dates.append(file_name)
            tweets.append(tweet)

    df_tweet = pd.DataFrame(columns=['stock', 'target_date', 'tweet'])
    df_tweet['stock'] = stocks
    df_tweet['target_date'] = dates
    df_tweet['tweet'] = tweets

    print(df_tweet.head())

    df_tweet.to_csv('Data/df_tweet.csv', index=False)



