from chatbot import Chatbot
from functions import *

bot = Chatbot(mode=3)

while True:
    try:
        print_wait("Enter input: \n")
        sent = input()
        output = bot.predict(sent)
        print_complete("Output: \n")
        print(output)
    except KeyboardInterrupt:
        break
