import numpy as np
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
    return X[:, :nb_of_training], X[:, nb_of_training:]


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


def showImages(ims):
    fig, axes = plt.subplots(math.ceil(ims.shape[1]/4), 4, figsize=(8, 4))

    # Show the first image

    axes = axes.flatten()

    for i in range(ims.shape[1]):
        axes[i].imshow(np.rot90(ims[:,i].reshape(46,56), k=3), 'grey')
        #axes[i].set_title(title1)
        axes[i].axis('off')

    # Adjust spacing between the plots
    plt.tight_layout()

    plt.show()







