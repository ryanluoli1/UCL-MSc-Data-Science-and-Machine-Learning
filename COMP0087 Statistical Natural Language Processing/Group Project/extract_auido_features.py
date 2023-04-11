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

from transformers import Wav2Vec2Model, Wav2Vec2Processor



if __name__ == '__main__':

    # set device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is available.")
    else:
        device = torch.device("cpu")
        print("Using CPU.")

    # load the pre-trained Wav2Vec 2.0 model and its processor from HuggingFace
    wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

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

    # load the audio data into an array
    audio_path = "Data/Audio"
    audio_data = []
    for folder_name in tqdm(os.listdir(audio_path)):
        if folder_name == '.DS_Store':
            continue
        folder_path = os.path.join(audio_path, folder_name)
        for file_name in os.listdir(folder_path):
            if file_name == '.DS_Store':
                continue
            if file_name.split('.')[0] in file_names:
                file = os.path.join(folder_path, file_name)
                audio_data.append(librosa.load(file)[0])
    print('Audio data loaded!')

    # preprocess the audio data using the processor
    encoded_audio = processor(list(audio_data),
                              sampling_rate=16000,
                              padding=True,
                              truncation=True,
                              max_length=160000,
                              return_tensors="pt").input_values.to(device)
    print('Audio data encoded! Shape: ', encoded_audio.shape)

    # prepare the data loader
    audio_loader = DataLoader(TensorDataset(encoded_audio), batch_size=4)

    # extract audio features using Wav2Vec 2.0
    first = True
    for data in tqdm(audio_loader):
        with torch.no_grad():
            outputs = wav2vec(data[0]).last_hidden_state
        if first:
            audio_features = outputs
            first = False
        else:
            audio_features = torch.concat((audio_features, outputs), dim=0)
        torch.cuda.empty_cache()
        del outputs

    # save the extracted audio features
    torch.save(audio_features, "Data/Audio Features/audio_features.pt")
    print('Fold 1 audio features saved!')
    print('Audio features shape: ', audio_features.shape)

