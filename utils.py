import numpy as np
import random
import math
import matplotlib.pyplot as plt


def getImages():
    data = np.load('face.npz')

    X = data['X']
    y = data['y']

    return X, y

def showImage(img, title):
    plt.title(title)
    plt.imshow(np.rot90(img.reshape(46,56), k=3), 'grey' )
    plt.show()
    return


def getAverageColumn(X):
    avg = np.zeros(X.shape[0])
    for i in range(X.shape[1]):
        avg += X[:, i]

    avg = avg/ X.shape[1]
    return avg

    
def separateTrainingTest(X, nb_of_training):

    #todo I should shuffle
    random.seed(10) #set the seed to be replicative

    testNumber = X.shape[1] - nb_of_training
    trainingIndexes = random.sample(range(X.shape[1]), nb_of_training)
    return X[: ,trainingIndexes], X[: ,list(set(range(X.shape[1])) - set(trainingIndexes))]


def show2images(x,y, title1, title2):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    # Show the first image
    axes[0].imshow(np.rot90(x.reshape(46,56), k=3), 'grey')
    axes[0].set_title(title1)
    axes[0].axis('off')

    # Show the second image
    axes[1].imshow(np.rot90(y.reshape(46,56), k=3), 'grey')
    axes[1].set_title(title2)
    axes[1].axis('off')

    # Adjust spacing between the plots
    plt.tight_layout()

    plt.show()


def showImages(ims, titles):
    n_images = ims.shape[1]
    n_cols = 4
    n_rows = math.ceil(n_images / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2, n_rows * 2))
    axes = axes.flatten()

    for i in range(n_images):
        img = ims[:, i].reshape(46, 56)
        axes[i].imshow(np.rot90(img, k=3), cmap='gray')
        axes[i].set_title(titles[i], fontsize=21)
        axes[i].axis('off')

    # Hide unused subplots
    for j in range(n_images, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()

def replicateImages(X, desired):
    while (X.shape[1] != desired):
        X = np.append(X, X[:,0:1], axis=1)

    print("Done replicating")

    return X


