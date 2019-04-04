from chatbot import Chatbot
import numpy as np
from functions import *
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib import offsetbox

bot = Chatbot(mode=3)
file = open("easy_train.txt", "r")
output = []

for sentence in file.readlines():
    output.append(bot.predict_latent_mapping(sentence)[0])

def embedding_plot(X, title):
    x_min, x_max = np.min(X, axis=0), np.max(X, axis=0)
    X = (X - x_min) / (x_max - x_min)
    print(X)
    plt.figure()
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(X[:,0], X[:,1])
    for i in range(len(X)):
        ax.annotate(str(i), X[i])

    #shown_images = np.array([[1., 1.]])
    #for i in range(X.shape[0]):
    #    if np.min(np.sum((X[i] - shown_images) ** 2, axis=1)) < 1e-2: continue
    #    shown_images = np.r_[shown_images, [X[i]]]
        #ax.add_artist(offsetbox.AnchoredText(str(i), X[i]))

    plt.xticks([]), plt.yticks([])
    plt.title(title)

X_tsne = manifold.TSNE(n_components=2, init='pca').fit_transform(output)
embedding_plot(X_tsne,"t-SNE")

plt.show()