from models.lstm import Single_LSTM
from functions import *
import numpy as np
import sys
import csv
import pickle as pkl

episodes = 500
epochs = 5
model_name = "model"

#* Read data
data = pkl.load(open("data/final_data.pkl", "rb"))
emotion_type = pkl.load(open("data/emotion_types.pkl", "rb"))
glove_dict = load_glove_dict("/media/edward/c40f4a81-55c2-49d2-8b86-91d871810cf2/glove_dict.pkl")
X = []; Y = []
for item in data:
    X.append(item[0]); Y.append([item[1]])

#* Build LSTM
lstm = Single_LSTM(200, len(emotion_type))
print("Built LSTM Models.")

#* Train with data
lstm.train(X, Y, episodes, epochs, glove_dict, emotion_type)
lstm.save("params/" + model_name)
