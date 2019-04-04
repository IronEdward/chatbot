from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from keras.models import model_from_json
from functions import *
from constants import *
import numpy as np
from tqdm import tqdm
from Models.lstm import *
import pickle as pkl

glove_dict = pkl.load(open("/media/edward/c40f4a81-55c2-49d2-8b86-91d871810cf2/GloVe_Tester/glove_dict.pkl", "rb"))
emotion_type = pkl.load(open("Models/emotion_types.pkl", "rb"))

model = Single_LSTM(200, len(emotion_type))
model.load("params/model")
while True:
    try:
        input_sentence = input("Give Input: ")
        vec = np.array([encode_sentence(input_sentence, glove_dict)])
        print(vec)
        result = model.predict(vec)
        print(result)
        class_result = np.argmax(result,axis=-1)[0]
        print(emotion_type[int(class_result)])
    except KeyboardInterrupt:
        break
