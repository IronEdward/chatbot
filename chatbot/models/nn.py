# Create your first MLP in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy
from keras.models import model_from_json

class NN():
    def __init__(self, input_dim, output_dim):
        self.model = Sequential()
        self.model.add(Dense(100, input_dim=input_dim))
        #self.model.add(Dense(50))
        #self.model.add(Dense(100))
        self.model.add(Dense(output_dim))
        self.model.compile(loss='mse', optimizer='adam')
        self.model.summary()

    def train(self, train_x, train_y):
        for index in range(len(train_x)):
            self.model.fit(train_x[index], train_y[index], epochs=50, verbose=True)

    def predict(self, x):
        return self.model.predict(x)
    
    def save(self, path):
        model_json = self.model.to_json()
        file_name = path + ".json"
        with open(file_name, "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(path + ".h5")

    def load(self, path):
        file_name = path + ".json"
        json_file = open(file_name, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(path + ".h5")