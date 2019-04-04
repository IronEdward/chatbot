from chatbot import Chatbot
from functions import *
import sys

data_PATH = "easy_train.txt"

train_x = []; train_y = []
print_load("Loading Data...")

#* Fix and feed data:
with open(data_PATH, "r") as dataFile:
    for conv in dataFile.readlines():
        original_conv = conv
        conv = conv.replace("\n", "")
        [speaker, listener] = conv.split("\t")
        if len(speaker) != 0 and len(listener) != 0:
            train_x.append(speaker); train_y.append(listener)

print_complete("Loaded Data.")

bot = Chatbot(mode=1)

print_wait("Training LDM...")
bot.train_chatbot_nn(train_x, train_y, train_count=50)
print_complete("Chatbot LDM Training Complete.")

print_wait("Saving...")
bot.chatbot_nn.save("params/chatbot_nn_small")
print_complete("Chatbot Saved.")
