
import time
import numpy as np
import utils
import matplotlib.pyplot as plt



def main():
    X, y = utils.getImages()

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    training, test = utils.separateTrainingTest(X, int(X.shape[1]*0.85))

    #Separate into training and rest 
    meanFace = utils.getAverageColumn(training)

    #Separate the mean
    utils.showImage(meanFace, 'mean face')
    #now we subtract the mean
    meanMatrix = np.repeat(meanFace[:, np.newaxis], training.shape[1], axis=1)
    phi = training - meanMatrix


    #standard cov matrix computation
    stime = time.time()
    S = 1/phi.shape[1] * phi @  np.transpose(phi)
    eigvalsBig, eigvecsBig = np.linalg.eigh(S)
    total1 = time.time() - stime
    print("Time to compute normal cov matrix + eig decomposition : " + str(total1))
    axes = axes.flatten()

    #we do the smal one now

    #we compute the covariance matrix

    stime = time.time()
    S = 1/phi.shape[1] *  np.transpose(phi) @ phi
    print(S.shape)


    #eigvals = np.diag(S)
    #eigvecs = np.eye(S.shape[0])
    eigvals, eigvecs = np.linalg.eigh(S)

    # Compute actual eigenfaces
    eigvecs = phi @ eigvecs
    eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)
    
    total2 = time.time() - stime
    print("Time to compute the small cov matrix + eig decomposition : " + str(total2))

    print("Radio between the times : " + str(total2/total1))

    #sort them
    indexes = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:,indexes]
    eigvals = eigvals[indexes]

    #sort them
    indexes = np.argsort(eigvalsBig)[::-1]
    eigvecsBig = eigvecsBig[:,indexes]
    eigvalsBig = eigvalsBig[indexes]


    #how many non zeros for the big?

    count = 0
    for i in eigvalsBig:
        if(i != 0):
            count += 1

    print("Out of " + str(len(eigvalsBig))  + " eigenvals, only " + str(count) + " are non-zero")

    #how many non_zeros ?
    count = 0
    for i in eigvals:
        if(i != 0):
            count += 1

    print("Out of " + str(len(eigvals))  + " eigenvals, only " + str(count) + " are non-zero")

    for i in range(20):
        #utils.showImage(eigvecs[:,i], "Eigenface " + str(i))
        #utils.show2images(eigvecsBig[:,i] - eigvecs[:,i], eigvecs[:,i] - eigvecsBig[:,i], "Small ", "Standard ")
        utils.showImages(eigvecsBig[:, :20])


    #check if they are the same

    for i, j in zip(eigvals, eigvalsBig):
        if i != j:
            print("Different ! ")


    for i, j in zip(eigvecs, eigvecsBig):
        if i != j:
            print("Different ! " + i)







main()
