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

bot = Chatbot(mode=2)

print_wait("Training...")
bot.train_chatbot_lstm(train_x, train_count=100)
print_complete("Chatbot Training Complete.")

print_wait("Saving...")
bot.chatbot_lstm.save("params/chatbot_vae", "params/chatbot_enc")
print_complete("Chatbot Saved.")