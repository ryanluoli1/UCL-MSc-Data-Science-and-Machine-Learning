The self-contain notebook: Framework.ipynb, including: loading data, feature extraction, model building, making inference, analyse predictions.


Python Files:

1. models: the proposed self and cross-MHA models
2. models_text_audio: single modal models (text & audio)
3. models_noSP: the proposed models with the statistical pooling layer removed
4. models_cls: the proposed models with the <CLS> token as the text input
5. extract_text_features: run to extract features from the raw text data using RoBERTa
6. extract_audio_features: run to extract features from the raw audio data using Wav2Vec 2.0
 

Jupyter Notebooks:

1. Training.ipynb: notebook for training the proposed self-MHA and cross-MHA models
2. Training (single modality).ipynb: notebook for training the single modality models
3. Training (noSP).ipynb: notebook for training the proposed self-MHA and cross-MHA models without the SP layer
4. Training (CLS).ipynb: notebook for training the proposed self-MHA and cross-MHA models using the <CLS> token as text input
5. Training Results Analysis.ipynb: notebook to visualise the experimental results of the models and the baselines
6. Framework.ipynb: a self-contain notebook that can be used for making inference


Other Files:

1. Folds: sample ids after shuffling for each fold of cross-validation
2. model_sa.pt: the best self-MHA model checkpoint, can be loaded when making inference
3. model_ca.pt: the best cross-MHA model checkpoint, can be loaded when making inference
4. Data/Inference/sample text.txt: a TXT file containing 14 sentences of text data
5. Data/Inference/sample labels.txt: a TXT file containing the labels


Folders:

1. Data: download the IEMOCAP dataset into this folder for training
2. Data/Folds: contains the fold ids for cross-validation splits
3. Data/Inference/Models: contains the trained models checkpoints
4. Data/Inference/sample audio: contains 14 sample audio files for testing the framework