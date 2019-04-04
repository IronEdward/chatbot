import pickle
from tqdm import tqdm
import numpy as np
from nltk.corpus import stopwords
from constants import *

def construct_dictionary(PATH):
    """Construct the GloVe dictionary."""
    glove_raw = open("/media/edward/c40f4a81-55c2-49d2-8b86-91d871810cf2/" + PATH)
    glove_dict = {}
    count = 0
    for i in tqdm(glove_raw):
        word2vec = i.split()
        glove_dict[word2vec[0]] = [float(i) for i in word2vec[1:]]
    return glove_dict

def load_glove_dict(PATH):
    print("Loading GloVe Dictionary...")
    with open(PATH, "rb") as file:
        print("Loaded GloVe Dictionary!")
        return pickle.load(file)

def encode_sentences(data, glove_dict):
    """Encode the sentences into GLoVe vectors."""
    vec_data = []
    for i in data:
        vec_sentence_data = []
        sent = i.lower()
        for j in sent:
            if 97>ord(j) or 122<ord(j) and j != " ":
                sent = sent.replace(j, "")
        sent = sent.split()
        if sent != None:
            for j in sent:
                try:
                    vec_sentence_data.append(glove_dict[j])
                except KeyError:
                    vec_sentence_data.append(list(np.zeros(len(glove_dict["the"]))-1.))
            vec_data.append(vec_sentence_data)
    return vec_data


def encode_sentence(sentence, glove_dict):
    """Encode the sentences into GLoVe vectors."""
    original_sentence = sentence
    vec_sentence = []
    try:
        sentence = sentence.lower()
    except AttributeError:
        print(sentence)
    for j in sentence:
        if (97>ord(j) or 122<ord(j)) and j != " " and j not in ok_symbols:
            sentence = sentence.replace(j, "")
    sentence = sentence.split()
    if sentence != None:
        for j in sentence:
            try:
                vec_sentence.append(glove_dict[j])
            except KeyError:
                vec_sentence.append(list(np.zeros(len(glove_dict["the"]))-1.))
    return vec_sentence

def make_outputs(reader_val, reader_arou, reader_domi):
    y = []
    for i in range(len(reader_val)):
        y.append([[[reader_val[i], reader_arou[i], reader_domi[i]]]])
    return y