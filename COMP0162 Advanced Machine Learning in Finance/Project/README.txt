A self-contained notebook "Framework.ipynb" is provided to demonstrate the proposed method. More detailed instructions are given inside the notebook. You need to run this notebook with the same directory as the "Inference" folder (can be found within the Source Code.zip" file) which contains the sample data and trained model weights.


Implementing codes in the "Source Codes" folder (preprocessing, feature extraction, training, evaluation):

1. Download the stocknet dataset from https://github.com/yumoxu/stocknet-dataset into the "Data" folder, unzip it to a folder with name "stockinet dataset" 

2. Run the "preprocess_tweets.py" file to preprocess the tweets, the processed tweets will be store as a DataFrame name "df_tweet.csv" in the "Data" folder

3. Run the "generate_candlestick_charts.py" to transform the historical stock prices into 5-day candlestick charts, all the generated image will be stored in the "candlestick charts" folder (zip this folder to be uploaded to Colab for training) under the "Data" folder.

4. Run the "load_images_and_labels.py" file to generate a DataFrame called "df_iamge.csv" which will be saved in the "Data" folder

5. Run the "Load Data.ipynb" notebook to load candlestick charts as lumpy arrays and store it as "image_data.npy" in the "Data" folder

6. Run the "Extract Text Features.ipynb" to extract text features using FinBERT, the extracted features will be stored as "text_features_256.pt" in the "Data" folder

7. Run the "Extract Image Features.ipynb" to extract image features using ResNet50, the extracted features will be stored as "image_features.pt" in the "Data" folder

8. Run the "Training (xxx).ipynb" notebooks to train the corresponding models and the trained weights will be stored in the "Models" folder

9. Run the "Training Results Analysis.ipynb" to print the result table and the confusion matrices