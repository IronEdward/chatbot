import pickle as pkl
from Functions.functions import *

data = pkl.load(open("Data(new)/data.pkl", "rb"))
emotion_type = pkl.load(open("Data(new)/emotion_types.pkl", "rb"))

new_data = []

for i in data:
    new_data.append([i[0], emotion_type.index(i[1])])

pkl.dump(new_data, open("Data(new)/final_data.pkl", "wb"))