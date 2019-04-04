import sys
from models.lstm_vae import *
from models.lstm import *
from models.lstm_vae_output import *
from models.nn import *
from keras.models import model_from_json
import pickle as pkl
from functions import *
from constants import *
from scipy import spatial
from rake_nltk import Rake
import numpy as np

glove_paths = "/media/edward/c40f4a81-55c2-49d2-8b86-91d871810cf2/GloVe_Tester/"
class Chatbot():
    def __init__(self, mode):
        #* mode == 1: Train NN; mode == 2: Train LSTM; mode == 3: Test

        #* Change the PATH to the respected destination!
        print_load("Loading [200D GloVe] Module...")
        self.glove_dict = pkl.load(open(glove_paths + "glove_dict.pkl", "rb"))
        print_complete("Loaded [200D GloVe] Module.")

        print_load("Loading [300D GloVe] Module...")
        self.glove_dict_300d = pkl.load(open(glove_paths + "glove_dict_300d.pkl", "rb"))
        print_complete("Loaded [300D GloVe] Module.")
        if mode == 3:
            print_load("Loading [300D GloVe Words] Module...")
            self.glove_dict_words_300d = pkl.load(open(glove_paths + "reverse_glove_dict_300d_words.pkl", "rb"))
            print_complete("Loaded [300D GloVe Words] Module.")

            print_load("Loading [300D GloVe Values] Module...")
            self.glove_dict_values_300d = pkl.load(open(glove_paths + "reverse_glove_dict_300d_values.pkl", "rb"))
            print_complete("Loaded [300D GloVe Values] Module.")

            end = np.zeros(len(self.glove_dict_300d["the"])); end[0] += 1
            self.glove_dict_values_300d.append(end)
            self.glove_dict_words_300d.append("___END___")

            print_load("Loading [300D GloVe KDTree] Module...")
            self.glove_dict_values_KDTree_300d = spatial.KDTree(self.glove_dict_values_300d)
            print_complete("Loaded [300D GloVe KDTree] Module.")

        #* Emotion classifier:
        print_load("Loading [Emotion Classifier] Module...")
        self.emotion_type = pkl.load(open("sentence_dimentionalizer/emotion_classifier/emotion_types.pkl", "rb"))
        self.emotion_classifier_model = Single_LSTM(200, len(self.emotion_type))
        self.emotion_classifier_model.load("sentence_dimentionalizer/emotion_classifier/model")
        print_complete("Loaded [Emotion Classifier] Module.")

        #* Intent classifier:
        print_load("Loading [Query Type Classifier] Module...")
        self.intent_classifier = LSTM_VAE(input_dim=300, batch_size=1, intermediate_dim=100, latent_dim=200, epsilon_std=1.)
        self.intent_classifier.load("sentence_dimentionalizer/intent_calculator/vae_test2", "sentence_dimentionalizer/intent_calculator/enc_test2")
        print_complete("Loaded [Query Type Classifier] Module.")

        #* Keyword extractor:
        print_load("Loading [Keyword Extractor] Module...")
        self.r = Rake()
        print_complete("Loaded [Keyword Extractor] Module.")

        #* Conversation Saver:
        print_load("Loading [Conversation Saver] Module...")
        self.previous_conversations = []
        print_complete("Loaded [Conversation Saver] Module.")

        """-----------------------------------------------------------------------------------------------------------------------------------------------------------------"""

        #* Build chatbot module:
        print_load("Loading [Chatbot Self-assertion] Module...")
        self.chatbot_lstm = LSTM_VAE(input_dim=200, batch_size=1, intermediate_dim=128, latent_dim=64, epsilon_std=1.)
        if mode != 2:
            self.chatbot_lstm.load("params/chatbot_vae", "params/chatbot_enc")
        print_complete("Loaded [Chatbot Self-assertion] Module.")

        #* Build nn module:
        print_load("Loading [Latent Dimention Mapper] Module...")
        self.chatbot_nn = NN(input_dim=64, output_dim=200)
        if mode == 3:
            self.chatbot_nn.load("params/chatbot_nn_small")
        print_complete("Loaded [Latent Dimention Mapper] Module.")

        if mode != 2:
            #* Build output module for training:
            print_load("Loading [Training output] Module...")
            self.chatbot_lstm_y = LSTM_VAE_Output(input_dim=300, batch_size=1, timesteps=padding, intermediate_dim=100, latent_dim=200, epsilon_std=1.)
            self.chatbot_lstm_y.load("output_module/vae_padding", "output_module/enc_padding", "output_module/dec_padding")
            print_complete("Loaded [Training output] Module.")

    def train_chatbot_lstm(self, train_x, train_count):
        train_data = []
        for index in range(len(train_x)):
            sentence_x = train_x[index]

            #* Convert sentences into GloVe vectorizers:
            vec_x = encode_sentence(sentence_x, self.glove_dict)
            vec_x_300d = encode_sentence_intent(sentence_x, self.glove_dict_300d)

            #* Compute values for each module:
            #* Emotion classifier:
            ECM_value = self.emotion_classifier_model.predict(np.array([vec_x]))
            ECM_value = int(np.argmax(ECM_value,axis=-1)[0])
            #* Convert emotion into GloVe matrix:
            ECM_value = encode_sentence(self.emotion_type[ECM_value], self.glove_dict)

            #* Intent classifier:
            ENC_value = self.intent_classifier.enc_predict(np.array([vec_x_300d]))

            

            PRE_value = []
            self.r.extract_keywords_from_text(sentence_x)
            for keywords in self.r.get_ranked_phrases():
                for keyword in keywords.split():
                    PRE_value.append(encode_keyword(keyword, self.glove_dict))

            #* Sum all vectors into one matrix:
            final_matrix = []
            final_matrix.append(ECM_value[0])
            final_matrix.append(ENC_value[0])
            for i in PRE_value:
                final_matrix.append(i)
            final_matrix = np.array(final_matrix)
            

            train_data.append(final_matrix)
            self.previous_conversations.append(train_data)
        train_data = np.array(train_data)

        #* Conversation Memorizer
        for _ in tqdm(range(train_count), desc="Episode: "):
                try:
                    self.chatbot_lstm.train(train_data)
                except KeyboardInterrupt:
                    return

    def train_chatbot_nn(self, train_x, train_y, train_count):
        x = []; y = []
        for index in range(len(train_x)):
            sentence_x = train_x[index]
            sentence_y = train_y[index]
            print(sentence_x, sentence_y)

            #* Convert sentences into GloVe vectorizers:
            vec_x = encode_sentence(sentence_x, self.glove_dict)
            vec_x_300d = encode_sentence_intent(sentence_x, self.glove_dict_300d)
            vec_y_300d = encode_sentence_intent_padding(sentence_y, self.glove_dict_300d, padding)

            #* Compute values for each module:
            #* Emotion classifier:
            ECM_value = self.emotion_classifier_model.predict(np.array([vec_x]))
            ECM_value = int(np.argmax(ECM_value,axis=-1)[0])
            #* Convert emotion into GloVe matrix:
            ECM_value = encode_sentence(self.emotion_type[ECM_value], self.glove_dict)

            #* Intent classifier:
            ENC_value = self.intent_classifier.enc_predict(np.array([vec_x_300d]))

            PRE_value = []
            self.r.extract_keywords_from_text(sentence_x)
            for keywords in self.r.get_ranked_phrases():
                for keyword in keywords.split():
                    PRE_value.append(encode_keyword(keyword, self.glove_dict))

            #* Sum all vectors into one matrix:
            final_matrix = []
            final_matrix.append(ECM_value[0])
            final_matrix.append(ENC_value[0])
            for i in PRE_value:
                final_matrix.append(i)
            final_matrix = np.array(final_matrix)
            x_output = self.chatbot_lstm.enc_predict(np.array([final_matrix]))
            y_output = self.chatbot_lstm_y.enc_predict(np.array([vec_y_300d]))
            x.append(x_output)
            y.append(y_output)
        x = np.array(x); y = np.array(y)
        for _ in tqdm(range(train_count), desc="Episode: "):
                try:
                    self.chatbot_nn.train(x, y)
                except KeyboardInterrupt:
                    return

    def predict(self, x):
        vec_x = encode_sentence(x, self.glove_dict); vec_x_300d = encode_sentence(x, self.glove_dict_300d)

        #* Compute values for each module:
        #* Emotion classifier:
        ECM_value = self.emotion_classifier_model.predict(np.array([vec_x]))
        ECM_value = int(np.argmax(ECM_value,axis=-1)[0])
        #* Convert emotion into GloVe matrix:
        ECM_value = encode_sentence(self.emotion_type[ECM_value], self.glove_dict)

        #* Intent classifier:
        ENC_value = self.intent_classifier.enc_predict(np.array([vec_x_300d]))

        PRE_value = []
        self.r.extract_keywords_from_text(x)
        for keywords in self.r.get_ranked_phrases():
            for keyword in keywords.split():
                PRE_value.append(encode_keyword(keyword, self.glove_dict))

        #* Sum all vectors into one matrix:
        final_matrix = []
        final_matrix.append(ECM_value[0])
        final_matrix.append(ENC_value[0])
        for i in PRE_value:
            final_matrix.append(i)
        final_matrix = np.array(final_matrix)

        lstm_y = self.chatbot_lstm.enc_predict(np.array([final_matrix]))
        nn_y = self.chatbot_nn.predict(np.array(lstm_y))
        output = self.chatbot_lstm_y.dec_predict(np.array(nn_y))
        
        return decode_sentence_padding(output, self.glove_dict_words_300d, self.glove_dict_values_KDTree_300d)

    def predict_latent_mapping(self, x):
        vec_x = encode_sentence(x, self.glove_dict); vec_x_300d = encode_sentence(x, self.glove_dict_300d)

        #* Compute values for each module:
        #* Emotion classifier:
        ECM_value = self.emotion_classifier_model.predict(np.array([vec_x]))
        ECM_value = int(np.argmax(ECM_value,axis=-1)[0])
        #* Convert emotion into GloVe matrix:
        ECM_value = encode_sentence(self.emotion_type[ECM_value], self.glove_dict)

        #* Intent classifier:
        ENC_value = self.intent_classifier.enc_predict(np.array([vec_x_300d]))

        PRE_value = []
        self.r.extract_keywords_from_text(x)
        for keywords in self.r.get_ranked_phrases():
            for keyword in keywords.split():
                PRE_value.append(encode_keyword(keyword, self.glove_dict))

        #* Sum all vectors into one matrix:
        final_matrix = []
        final_matrix.append(ECM_value[0])
        final_matrix.append(ENC_value[0])
        for i in PRE_value:
            final_matrix.append(i)
        final_matrix = np.array(final_matrix)

        return self.chatbot_lstm.enc_predict(np.array([final_matrix]))

