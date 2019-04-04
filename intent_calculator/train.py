import sys
from models.lstm_vae import *
from functions.functions import *
from tqdm import tqdm

dataset = "easy_train.txt"
loop_count = 50
model = LSTM_VAE(input_dim=300, batch_size=1, intermediate_dim=100, latent_dim=200, epsilon_std=1.)

glove_dict = load_glove_dict("/media/edward/c40f4a81-55c2-49d2-8b86-91d871810cf2/GloVe_Tester/glove_dict_300d.pkl")
speaker_data = []; listener_data = []; data = []


with open(dataset, "r") as dataFile:
    for conv in dataFile.readlines():
        original_conv = conv
        conv = conv.replace("\n", "")
        [speaker, listener] = conv.split("\t")
        speaker_vec = encode_sentence_intent(speaker, glove_dict); listener_vec = encode_sentence_intent(listener, glove_dict)
        speaker_data.append(speaker_vec); listener_data.append(listener_vec); data.append(speaker_vec); data.append(listener_vec)

model.train(speaker_data)
model.save("params/vae", "params/enc", "params/dec")
print("Saved.")
