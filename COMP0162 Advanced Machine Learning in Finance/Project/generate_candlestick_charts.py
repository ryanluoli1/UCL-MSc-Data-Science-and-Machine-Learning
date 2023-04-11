import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from PIL import Image
from mplfinance.original_flavor import candlestick2_ochl

import warnings
warnings.filterwarnings('ignore')


def plot_candlestick(df, save_path, show_plot=True, fig_size=100, dpi=96):
    fig = plt.figure(figsize=(fig_size / (11 / 15) / dpi, fig_size / (11 / 15) / dpi), dpi=dpi)
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_facecolor('black')
    ax1.figure.set_facecolor('k')
    candlestick2_ochl(ax1,
                      df['Open'], df['Close'], df['High'], df['Low'],
                      width=0.8, colorup='#77d879', colordown='#db3f3f')
    ax1.grid(False)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax1.xaxis.set_visible(False)
    ax1.yaxis.set_visible(False)
    ax1.axis('off')

    if show_plot:
        plt.show()

    fig.savefig(save_path, pad_inches=0, transparent=False)
    plt.close(fig)



if __name__ == '__main__':

    raw_price_path = 'Data/stocknet dataset/price/raw'

    stock_names, N_raw = [], []
    first = True
    for file_name in tqdm(os.listdir(raw_price_path)):
        dates, open_prices, high_prices, low_prices, close_prices = [], [], [], [], []
        file_path = os.path.join(raw_price_path, file_name)
        df = pd.read_csv(file_path)
        df = df[df['Date'].between('2014-01-01', '2016-01-01')]
        stock_names.append(file_name.split('.')[0])
        df['Stock'] = len(df) * [file_name.split('.')[0]]
        if first:
            df_price = df
            first = False
        else:
            df_price = pd.concat([df_price, df])
        N_raw.append(len(df))
    df_price = df_price.reset_index(drop=True)

    df_price_path = 'Data/df_prices.csv'
    df_price.to_csv(df_price_path, index=False)

    print('Historical price dataframe loaded!')

    df = df_price[df_price['Stock'] == 'CSCO'][20:25]
    save_path = 'Data/sample.png'
    plot_candlestick(df, save_path, show_plot=True)

    labels = {}

    for stock in tqdm(stock_names):
        df_stock = df_price[df_price['Stock'] == stock]
        valid_dates, valid_labels = [], []
        for i in range(len(df_stock)):
            if i + 5 >= len(df_stock):
                break
            diff = 100 * (df_stock.iloc[i + 5]['Adj Close'] - df_stock.iloc[i + 4]['Adj Close']) / df_stock.iloc[i + 4][
                'Adj Close']
            if diff <= -0.5:
                valid_label = 0
            elif diff >= 0.55:
                valid_label = 1
            else:
                continue
            valid_labels.append(valid_label)
            target_date = df_stock.iloc[i + 5]['Date']
            valid_dates.append(target_date)
            save_path = f'Data/candlestick charts/{stock}@{target_date}#{valid_label}.png'
            plot_candlestick(df_stock[i:i + 5], save_path, show_plot=False)
        info = {'date': valid_dates, 'label': valid_labels}
        labels[stock] = info





