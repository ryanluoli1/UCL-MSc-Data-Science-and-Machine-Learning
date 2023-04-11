# pip install transformers

import gc
import os
import csv
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoModel, AutoTokenizer



if __name__ == '__main__':

    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    # load the pre-trained RoBERTa and its tokenizer from HuggingFace
    roberta = AutoModel.from_pretrained('roberta-large').to(device)
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')

    # load the labels
    label_path = "Data/seven_classes_id_labels.txt"
    labels, file_names = [], []
    with open(label_path, 'r') as f:
        for row in csv.reader(f, delimiter='\t'):
            for item in row:
                if item.isdigit():
                    labels.append(int(item))
                else:
                    file_names.append(item)
    labels = np.array(labels)
    file_names = np.array(file_names)
    print('Labels loaded!')

    # load the text data
    text_path = "Data/transcripts.csv"
    df_text = pd.read_csv(text_path, names=['file_name', 'transcript'])
    df_text_sub = df_text[df_text['file_name'].isin(file_names)].copy()
    text_data = df_text_sub['transcript'].values
    print('Text data loaded!')

    # preprocess the text data using the tokenizer
    encoded_text = tokenizer.batch_encode_plus(list(text_data),
                                               padding=True,
                                               truncation=True,
                                               return_tensors='pt').to(device)
    print('Text data encoded!')

    # prepare the data loader
    text_dataset = TensorDataset(encoded_text['input_ids'], encoded_text['attention_mask'])
    text_loader = DataLoader(text_dataset, batch_size=32)

    # extract features using RoBERTa
    first = True
    for input_ids, attention_mask in tqdm(text_loader):
        with torch.no_grad():
            outputs = roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if first:
            text_features = outputs
            first = False
        else:
            text_features = torch.concat((text_features, outputs), dim=0)
        del outputs

    # save the extracted text features
    torch.save(text_features, "Data/text_features.pt")
    print('Text features saved!')
    print('Text features shape: ', text_features.shape)























