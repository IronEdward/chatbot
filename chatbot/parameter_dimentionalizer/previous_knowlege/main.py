from rake_nltk import Rake
import pickle as pkl
import re
from tqdm import tqdm

dataset = "/media/edward/9010d49a-bf5b-4b68-8aa0-a7d03a8d12e9/Datas/train.txt"

data = []
knowlege = {}
print("Building Dataset....")
with open(dataset, "r") as dataFile:
    for conv in dataFile.readlines():
        conv = conv.lower(); conv = conv[2:]; conv = conv.replace("\n", "")
        [speaker, listener] = conv.split("\t")
        speaker, listener = re.split('[?.!]', speaker), re.split('[?.!]', listener)
        for line in speaker:
            if line != "":
                data.append(line)
        for line in listener:
            if line != "":
                data.append(line)

"""For the sake of clarity, I divided the part to clean up the data and process the data, even though I could've mashed them up together"""

print("Creating storage.pkl...")
r = Rake()
for line in tqdm(data):
    r.extract_keywords_from_text(line)
    keywords = r.get_ranked_phrases()
    for keyword in keywords:
        if keyword not in knowlege.keys():
            #print(keyword, keywords)
            keywords.remove(keyword)
            if keywords != None:
                #print("KEYWORDS: ", eywords)
                knowlege[keyword] = keywords
            else:
                knowlege[keyword] = []
        else:
            for other_keyword in keywords:
                if other_keyword not in knowlege[keyword]:
                    knowlege[keyword].append(other_keyword)

with open("storage.pkl", "wb") as pklfile:
    pkl.dump(knowlege, pklfile)