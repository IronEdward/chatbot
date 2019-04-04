import pickle
from tqdm import tqdm
import numpy as np
from nltk.corpus import stopwords

ok_symbols = "?!., "

def encode_keyword(word, glove_dict):
    try:
        return glove_dict[word]
    except KeyError:
        return list(np.zeros(len(glove_dict["the"]))-1.)

def encode_sentence(sentence, glove_dict):
    """Encode the sentences into GLoVe vectors."""
    original_sentence = sentence
    vec_sentence = []
    sentence = sentence.lower()
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
    return np.array(vec_sentence)

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
    return np.array(vec_data)

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

def encode_sentence_intent_padding(sentence, glove_dict, padding_length):
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
    end = np.zeros(len(glove_dict["the"]))
    end[0] += 1
    vec_sentence.append(end)
    while len(vec_sentence) < padding_length:
        vec_sentence.append(np.zeros(len(glove_dict["the"])))
    #print(np.array(vec_sentence).shape)
    return np.array(vec_sentence)

def decode_sentence(sentence_vec, glove_dict_words, glove_dict_values_KDTree):
    #* (Assuming sentence_vec is an array with 200d arrays inside...)
    result = []
    for word_vec in sentence_vec:
        closest_index = glove_dict_values_KDTree.query(word_vec)[1]
        print(closest_index)
        result.append(glove_dict_words[closest_index[0]])
    return " ".join(result)
    
def decode_sentence_padding(sentence_vec, glove_dict_words, glove_dict_values_KDTree):
    sentence_vec = sentence_vec[0]
    result = []
    for word_vec in sentence_vec:
        #print(word_vec)
        closest_index = glove_dict_values_KDTree.query(word_vec)[1]
        word = glove_dict_words[closest_index]
        if word == "___END___":
            return " ".join(result)
        #print(closest_index)
        result.append(word)
    return " ".join(result)

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


def make_outputs(reader_val, reader_arou, reader_domi):
    y = []
    for i in range(len(reader_val)):
        y.append([[[reader_val[i], reader_arou[i], reader_domi[i]]]])
    return y

OKBLUE = '\033[96m'
OKGREEN = '\033[92m'
WARNING = '\033[93m'
RED = '\033[91m'
ERROR = '\033[41m'
WAIT = '\033[43m'
ENDC = '\033[0m'

def print_complete(mes):
    print( OKBLUE + str(mes) + ENDC)

def print_load(mes):
    print( OKGREEN + str(mes) + ENDC)

def print_warn(mes):
    print( WARNING + str(mes) + ENDC)

def print_red(mes):
    print( RED + str(mes) + ENDC)

def print_error(mes):
    print( ERROR + str(mes) + ENDC)

def print_wait(mes):
    print( WAIT + str(mes) + ENDC)
