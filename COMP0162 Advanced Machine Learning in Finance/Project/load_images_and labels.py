import os
import csv
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt

from PIL import Image

import warnings
warnings.filterwarnings('ignore')



def load_candlestick(image_path, show_plot=True):
    img = Image.open(image_path)
    img = img.convert('RGB')
    crop_size = img.size[0] * 2/15
    img = img.crop((crop_size, crop_size, img.size[0]-crop_size, img.size[0]-crop_size))
    img = np.asarray(img)/255
    if show_plot:
        plt.imshow(img)
        plt.grid(False)
        plt.axis('off')
        plt.show()
    return img


if __name__ == '__main__':

    image_path = 'Data/candlestick charts/' + os.listdir('Data/candlestick charts')[0]
    img = load_candlestick(image_path, show_plot=True)

    print('Sample candlestick chart plotted!')

    df_image = pd.DataFrame(columns=['stock', 'target_date'])

    stock_names, target_dates, image_names, labels, features = [], [], [], [], []
    for file_name in tqdm(os.listdir('Data/candlestick charts')):
        stock_name = file_name.split('.')[0].split('@')[0]
        stock_names.append(stock_name)
        target_date = file_name.split('.')[0].split('@')[1].split('#')[0]
        target_dates.append(target_date)
        image_names.append(file_name)
        label = int(file_name.split('.')[0].split('#')[1])
        labels.append(label)
        # image_path = 'Data/candlestick charts/' + file_name
        # feature = load_candlestick(image_path, show_plot=False)
        # features.append(feature)
    print('Image Data extracted!')

    df_image['stock'] = stock_names
    df_image['target_date'] = target_dates
    df_image['image_name'] = image_names
    df_image['label'] = labels
    df_img_features_path = 'Data/df_image.csv'
    df_image.to_csv(df_img_features_path, index=False)

    print(df_image.head())

    # np.save('Data/img_features.npy', features)
    # print('Image Data saved!')

