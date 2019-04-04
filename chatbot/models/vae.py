from keras.layers import Lambda, Input, Dense
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import numpy as np

def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))                               # by default, random_normal has mean=0 and std=1.0

    return z_mean + K.exp(0.5 * z_log_var) * epsilon

class VAE:
    def __init__(self, input_dim, output_dim, intermediate_dim, latent_dim, batch_size, epochs):
        # VAE model = encoder + decoder
        #* Build encoder model
        input_shape = (input_dim, )
        inputs = Input(shape=input_shape, name='encoder_input')
        x = Dense(intermediate_dim, activation='relu')(inputs)
        z_mean = Dense(latent_dim, name='z_mean')(x); z_log_var = Dense(latent_dim, name='z_log_var')(x)
        z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
        self.encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')

        #* Build decoder model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        x = Dense(intermediate_dim, activation='relu')(latent_inputs)
        outputs = Dense(output_dim, activation='sigmoid')(x)
        self.decoder = Model(latent_inputs, outputs, name='decoder')

        #? Instantiate VAE model
        outputs = self.decoder(self.encoder(inputs)[2])
        self.vae = Model(inputs, outputs, name='vae')

        # VAE loss = mse_loss or xent_loss + kl_loss
        reconstruction_loss = binary_crossentropy(inputs, outputs); reconstruction_loss *= input_dim
        kl_loss = K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) * -0.5
        vae_loss = K.mean(reconstruction_loss + kl_loss); self.vae.add_loss(vae_loss)
        vae.compile(optimizer='adam')
        vae.summary()

    def save(self, name):
        vae_model_json = self.vae.to_json()
        vae_file_name = "chatbot_vae.json"
        with open(vae_file_name, "w") as json_file:
            json_file.write(vae_model_json)
        self.vae.save_weights("chatbot_vae.h5")

        enc_model_json = self.encoder.to_json()
        enc_file_name = "chatbot_enc.json"
        with open(enc_file_name, "w") as json_file:
            json_file.write(enc_model_json)
        self.encoder.save_weights("chatbot_enc.h5")

        dec_model_json = self.dec.to_json()
        dec_file_name = "chatbot_dec.json"
        with open(dec_file_name, "w") as json_file:
            json_file.write(dec_model_json)
        self.decoder.save_weights("chatbot_dec.h5")
        print("Saved models to disk.")
        
    def load(self, name):
        file_name = name + ".json"
        json_file = open(file_name, "r")
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights(name + ".h5")
        print("Loaded model from disk.")

    def train(self, x, y, epochs):
        for _ in range(5):
            for i in tqdm(range(len(x))):
                try:
                    vae.fit(np.array([x[i]]), np.array([y[i]]), epochs=epochs, verbose=False)
                except KeyboardInterrupt:
                    save_models(vae, gen, enc)
                    break
