import sys
from models.lstm_vae import *
from functions.functions import *
from keras.models import model_from_json
from tqdm import tqdm
from scipy import spatial
import pickle

model = LSTM_VAE(input_dim=300, batch_size=1, intermediate_dim=100, latent_dim=200, epsilon_std=1.)

glove_dict = load_glove_dict("/media/edward/c40f4a81-55c2-49d2-8b86-91d871810cf2/GloVe_Tester/glove_dict_300d.pkl")
glove_dict_values = pickle.load(open("/media/edward/c40f4a81-55c2-49d2-8b86-91d871810cf2/GloVe_Tester/reverse_glove_dict_300d_values.pkl", "rb"))
glove_dict_words = pickle.load(open("/media/edward/c40f4a81-55c2-49d2-8b86-91d871810cf2/GloVe_Tester/reverse_glove_dict_300d_words.pkl", "rb"))
glove_dict_values_KDTree = spatial.KDTree(glove_dict_values)

model.load("params/vae_test2", "params/enc_test2", "params/dec_test2")

while True:
    try:
        print("Enter input:")
        sent = input()
        sent = encode_sentence_intent(sent, glove_dict)
        print(sent)
        output = model.enc_predict(np.array([sent]))
        sent = model.vae_predict(np.array([sent]))
        print("Output:")
        print(output.shape, output)
        print(decode_sentence(sent, glove_dict_words, glove_dict_values_KDTree))
    except KeyboardInterrupt:
        break