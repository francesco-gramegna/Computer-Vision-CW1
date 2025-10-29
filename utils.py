import numpy as np
import time
import NN
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

    print(trainingIndexes[113])
    return X[: ,trainingIndexes], X[: ,list(set(range(X.shape[1])) - set(trainingIndexes))]


def separateTrainingTestQ1(X, y):
    trainingIndexes = []
    testIndexes = []
    random.seed(12)
    for i in range(0,X.shape[1],10):
        samples = random.sample(range(10), 8)
        for j in range(len(samples)):
            samples[j]+=i
        trainingIndexes += samples

    print(len(trainingIndexes))
    return X[: ,trainingIndexes], X[: ,list(set(range(X.shape[1])) - set(trainingIndexes))], y[:,trainingIndexes], y[:, list(set(range(X.shape[1])) - set(trainingIndexes))]



def separateTrainingTestQ2(X, y):
    random.seed(34)

    indexes = [[] for i in range(4)] 
    
    for i in range(0,X.shape[1],8):

        temp = []
        for j in range(4):
            samples = random.sample(list(set(range(8)) - set(temp)), 2)
            temp += samples[:]

            for t in range(len(samples)):
                samples[t]+=i
            indexes[j] += samples

    return ([X[:,indexes[0]] , X[:,indexes[1]], X[:,indexes[2]], X[:,indexes[3]]],
            [y[:,indexes[0]], y[:,indexes[1]], y[:,indexes[2]], y[:,indexes[3]]])




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

def mse(x,y):
    e = 0
    for a,b in zip(x, y):
        e += (a-b)**2
    e = e/len(x)

    return e 



def findElbow(x, y):
    x = np.array(x)
    y = np.array(y)

    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    line_vec = p2 - p1
    line_vec_norm = line_vec / np.linalg.norm(line_vec)

    vecs = np.column_stack((x - x[0], y - y[0]))

    distances = np.abs(np.cross(line_vec, vecs)) / np.linalg.norm(line_vec)

    idx = np.argmax(distances)
    return x[idx]


def findBestK(X, mean, vec):
    results = []
    for k in range(vec.shape[1]-1):
        keptVec = vec[:,:k+1]

        W =  keptVec.T @ X

        #reconstruction error

        meanMatrix = np.repeat(mean[:, np.newaxis], X.shape[1], axis=1)
        R = meanMatrix + keptVec @ W

        #compute MSE

        totalError = 0
        for i in range(W.shape[1]):
            totalError += mse(W[:,i], R[:,i])

        totalError /= W.shape[1]

        results.append(totalError)
    print(results)

    elbow = findElbow(range(1,len(results)+1), results)
            
    print("Best value : " + str(elbow-1))

    plt.plot(range(1, len(results)+1), results) 
    plt.xlabel("k (nb of eigenvectors)", fontsize=21)
    plt.ylabel("MSE", fontsize=21)

    plt.plot(elbow, results[elbow-1], 'rx', markersize=20, mew=2, label='elbow = ' + str(elbow))

    plt.legend(fontsize=22)

    plt.show()

    return elbow, results


def findTestAccuracy(classifier, testX, testY):
    total = 0
    for i in range(testX.shape[1]):
        if (classifier.classify(testX[:,i]) == testY[:,i]):
            total+=1
         
    return 100*total/testX.shape[1]



def testDifferentKValues(maxK,phiTrain,trainY, testX,testY,pca00,pca01,pca02, pca03, total_pca):
    pcas = [pca00, pca01, pca02, pca03, total_pca]    
    results = []
    times = []


    for k in range(1,maxK+1):
        stime = time.time()
        pcbs = [ NN.NNPCAClassifier(vec[:,:k].T @ phiTrain, trainY, mean, vec[:,:k]) for mean, vec, val,_, _,_ in pcas]
        results.append(tuple([findTestAccuracy(cl, testX, testY) for cl in pcbs]))
        endTime = time.time() - stime
        times.append(endTime)

        print(results[-1])
    
        
    return results, times

    
def plotTestAccuracyTestQ2(maxK, results, times):
    res0 = [r[0] for r in results]
    res1 = [r[1] for r in results]
    res2 = [r[2] for r in results]
    res3 = [r[3] for r in results]
    res4 = [r[4] for r in results]

    fig, ax1 = plt.subplots()

    ax1.plot(range(1, maxK+1), res0, 'b', label='PCA (one subset of 104 images)')
    ax1.plot(range(1, maxK+1), res1, 'g', label='IPCA (2 subsets)')
    ax1.plot(range(1, maxK+1), res2, 'c', label='IPCA (3 subsets)')
    ax1.plot(range(1, maxK+1), res3, 'm', label='IPCA (4 subsets)')
    ax1.plot(range(1, maxK+1), res4, 'r', label='PCA (all training images)')
    ax1.set_xlabel('K=dimensions kept in the dimensionnality reduction')
    ax1.set_ylabel('Accuracy (%)')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()

    ax2.bar(range(1, maxK+1), times, color='y', alpha=0.3, label='Total testing time (s)')

    ax2.set_ylabel('Total testing time (s)')
    ax2.legend(loc='upper right')

    ax1.axvline(x=39, color='orange', linestyle=':')

    ax1.text(39 + 0.5, 80, f'K=39', color='red', rotation=90, va='bottom')
    
    
    plt.legend()
    plt.show()


