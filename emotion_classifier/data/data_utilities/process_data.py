"""Read files, organize the text, convert them to GloVe format, and save to pkl file"""
import pickle
import csv
import numpy as np
from Functions.constants import *
from Functions.functions import *
from tqdm import tqdm
import re
import sys
from nltk.corpus import stopwords

main_file = "Data/Normal/files/ise_processed.txt"
neutral_file = "Data/Normal/files/new_neutral.tsv"
data_file = "Data/final.tsv"

glove_dict = load_glove_dict("glove/glove_dict.pkl")
stopwords_list = stopwords.words("english")

dataset = []
emotion_types = []

with open(main_file, "r") as i:
    for line in tqdm(i.readlines()):
        line_split = line.split()
        emotion = line_split[1]
        sentence = line_split[2:]
        for word in sentence:
            for letter in word:
                if letter in symbols:
                    sentence.remove(word)
        if len(sentence) > 1:
            vectorized_sentence = []
            for word in sentence:
                if word not in stopwords_list:
                    try:
                        vectorized_sentence.append(glove_dict[word])
                    except KeyError:
                        vectorized_sentence.append(list(np.zeros(len(glove_dict["the"]))))
            dataset.append([vectorized_sentence, emotion])

        if emotion not in emotion_types:
            emotion_types.append(emotion)

emotion_types.append("neutral")
neutral_dataset = {}
with open(neutral_file, "r") as i:
    tsv = list(csv.reader(i, delimiter = '\t'))
    for line in tqdm(tsv):
        if line != []:
            error = 0
            line_split = line[0].split()
            for word in line_split:
                for letter in word:
                    if letter in symbols:
                        line_split.remove(word)
                        break
            vectorized_sentence = []
            for word in line_split:
                if word not in stopwords_list:
                    try:
                        vectorized_sentence.append(glove_dict[word])
                    except KeyError:
                        error += 1
                        vectorized_sentence.append(list(np.zeros(len(glove_dict["the"]))))
            if error not in neutral_dataset.keys():
                neutral_dataset[error] = [vectorized_sentence]
            else:
                neutral_dataset[error].append(vectorized_sentence)
new_neutral_dataset = neutral_dataset[0] + neutral_dataset[1]
final_neutral_dataset = []
for i in new_neutral_dataset:
    final_neutral_dataset.append([i, "neutral"])


dataset += final_neutral_dataset

pickle.dump(dataset, open("data.pkl", "wb"))
pickle.dump(emotion_types, open("emotion_types.pkl", "wb"))

"""tsv_writer = csv.writer(open(data_file, "w"), delimiter='\t')
for i in tqdm(dataset):
    tsv_writer.writerow(i)"""