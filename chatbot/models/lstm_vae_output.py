from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
from keras.models import model_from_json
import numpy as np

def train_generator(x):
    while True:
        for data in x:
            yield np.array([data]), np.array([data])

class LSTM_VAE_Output():
    def __init__(self, input_dim, timesteps, batch_size, intermediate_dim, latent_dim, epsilon_std=1.):
        x = Input(shape=(timesteps, input_dim,))

        h = LSTM(intermediate_dim)(x)
        z_mean = Dense(latent_dim)(h)
        z_log_sigma = Dense(latent_dim)(h)

        def sampling(args):
            z_mean, z_log_sigma = args
            epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=epsilon_std)
            return z_mean + z_log_sigma * epsilon
        
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
        decoder_h = LSTM(intermediate_dim, return_sequences=True)
        decoder_mean = LSTM(input_dim, return_sequences=True)
        h_decoded = RepeatVector(timesteps)(z)
        h_decoded = decoder_h(h_decoded)
        x_decoded_mean = decoder_mean(h_decoded)
        decoder_input = Input(shape=(latent_dim,))
        _h_decoded = RepeatVector(timesteps)(decoder_input)
        _h_decoded = decoder_h(_h_decoded)
        _x_decoded_mean = decoder_mean(_h_decoded)

        self.vae = Model(x, x_decoded_mean)
        self.encoder = Model(x, z_mean)
        self.decoder = Model(decoder_input, _x_decoded_mean)

        def loss_function(x, x_decoded_mean):
            xent_loss = objectives.mse(x, x_decoded_mean)
            kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
            loss = xent_loss + kl_loss
            return loss

        self.vae.compile(optimizer='rmsprop', loss=loss_function)
        self.vae.summary()

    def enc_predict(self, x):
        return self.encoder.predict(x)

    def vae_predict(self, x):
        return self.vae.predict(x)

    def dec_predict(self, x):
        return self.decoder.predict(x)

    def train(self, x):
        self.vae.fit_generator(train_generator(x), steps_per_epoch=50, epochs=150, verbose=1)

    def save(self, path_vae, path_enc, path_dec):
        vae_model_json = self.vae.to_json()
        vae_file_name = path_vae + ".json"
        with open(vae_file_name, "w") as json_file:
            json_file.write(vae_model_json)
        self.vae.save_weights(path_vae + ".h5")

        enc_model_json = self.encoder.to_json()
        enc_file_name = path_enc + ".json"
        with open(enc_file_name, "w") as json_file:
            json_file.write(enc_model_json)
        self.encoder.save_weights(path_enc + ".h5")

        dec_model_json = self.decoder.to_json()
        dec_file_name = path_dec + ".json"
        with open(dec_file_name, "w") as json_file:
            json_file.write(dec_model_json)
        self.decoder.save_weights(path_dec + ".h5")

    def load(self, path_vae, path_enc, path_dec):
        file_name = path_vae + ".json"
        json_file = open(file_name, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.vae = model_from_json(loaded_model_json)
        self.vae.load_weights(path_vae + ".h5")

        file_name = path_enc + ".json"
        json_file = open(file_name, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.encoder = model_from_json(loaded_model_json)
        self.encoder.load_weights(path_enc + ".h5")

        file_name = path_dec + ".json"
        json_file = open(file_name, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.decoder = model_from_json(loaded_model_json)
        self.decoder.load_weights(path_dec + ".h5")