import pickle
from tqdm import tqdm
import numpy as np
from rake_nltk import Rake

ok_symbols = "?!., "
r = Rake()

def load_glove_dict(PATH):
    print("Loading GloVe Dictionary...")
    with open(PATH, "rb") as file:
        print("Loaded GloVe Dictionary!")
        return pickle.load(file)

def encode_sentence_intent(sentence, glove_dict):
    vec_sentence = []
    sentence = sentence.lower()
    for j in sentence:
        if (97>ord(j) or 122<ord(j)) and j not in ok_symbols:
            sentence = sentence.replace(j, "")
    sentence = sentence.split()
    if sentence != None:
        for j in sentence:
            try:
                vec_sentence.append(np.array(glove_dict[j]))
            except KeyError:
                vec_sentence.append(np.array(list(np.zeros(len(glove_dict["the"]))-1.)))
    return np.array(vec_sentence)

def encode_sentence_intent_remove_keywords(sentence, glove_dict):
    
    vec_sentence = []
    sentence = sentence.lower()
    for j in sentence:
        if (97>ord(j) or 122<ord(j)) and j not in ok_symbols:
            sentence = sentence.replace(j, "")
    sentence = sentence.split()
    if sentence != None:
        for j in sentence:
            try:
                vec_sentence.append(np.array(glove_dict[j]))
            except KeyError:
                vec_sentence.append(np.array(list(np.zeros(len(glove_dict["the"]))-1.)))
    return np.array(vec_sentence)

def decode_sentence(sentence_vec, glove_dict_words, glove_dict_values_KDTree):
    sentence_vec = sentence_vec[0]
    result = []
    for word_vec in sentence_vec:
        closest_index = glove_dict_values_KDTree.query(word_vec)[1]
        print(closest_index)
        result.append(glove_dict_words[closest_index])
    return " ".join(result)