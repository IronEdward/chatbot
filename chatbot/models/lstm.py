from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from keras.models import model_from_json
from functions import *
import numpy as np
from tqdm import tqdm

embed_dim_1 = 150
embed_dim_2 = 100
embed_dim_3 = 75
embed_dim_4 = 50

class Single_LSTM:
    """Creates an LSTM network.
    in_shape: Input dimentions.
    output_count: Output dimentions.
    """

    def __init__(self, in_shape, output_count):
        self.model = Sequential()
        self.model.add(LSTM(units=embed_dim_3, input_shape=(None, in_shape)))
        self.model.add(Dense(units=output_count, activation='softmax'))
        self.model.summary()
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, x_train, y_train, episode_count, epoch_count, glove_dict, emotion_types):
        for j in tqdm(range(episode_count), desc="Episode Progress"):
            for i in tqdm(range(len(x_train)), desc="Epoch Progress"):
                try:
                    self.model.train_on_batch(np.array([x_train[i]]), np.array([y_train[i]], dtype=np.uint8))
                except KeyboardInterrupt:
                    return
            #* Calculate
            print("Test!")
            index = np.random.randint(len(x_train)-1); test_X = x_train[index]; test_Y = y_train[index]
            #print(np.array([x_train[i]]), np.array([test_X]))
            results = self.predict(np.array([test_X]))
            class_result = np.argmax(results,axis=-1)[0]
            print(class_result, results)
            sentence = ""
            for word_vec in test_X:
                for key, val in glove_dict.items():
                    if val == list(word_vec):
                        sentence += key + " "
            print("Sentence:", sentence)
            print("Results: ", emotion_types[class_result])
            print("Expected Result: ", emotion_types[test_Y[0]])

    def predict(self, x):
        return self.model.predict_on_batch(x)

    def save(self, name):
        model_json = self.model.to_json()
        file_name = name + ".json"
        with open(file_name, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(name + ".h5")
        print("Saved model to disk.")
        
    def load(self, name):
        file_name = name + ".json"
        json_file = open(file_name, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(name + ".h5")
        print("Loaded model from disk.")


