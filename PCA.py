
import NN
import math
import time
import numpy as np
import utils
import matplotlib.pyplot as plt



def main():
    X, y = utils.getImages()


    print("X shape:", X.shape)
    print("y shape:", y.shape)

    training, test, _,_ = utils.separateTrainingTestQ1(X, y)

    #training = utils.replicateImages(training, 2576)


    #Separate into training and rest 
    meanFace = utils.getAverageColumn(training)

    #Separate the mean
    utils.showImage(meanFace, 'mean face')
    #now we subtract the mean
    meanMatrix = np.repeat(meanFace[:, np.newaxis], training.shape[1], axis=1)

    phi = training - meanMatrix

    #mean analysis
    utils.showImages((np.array([meanFace, training[:, 355], phi[:, 355]])).T, ["mean face", "sample face", "face - mean"])


    #standard cov matrix computation
    stime = time.process_time()
    S = 1/phi.shape[1] * phi @  np.transpose(phi)
    eigvalsBig, eigvecsBig = np.linalg.eigh(S)
    total1 = time.process_time() - stime
    print("Time to compute normal cov matrix + eig decomposition : " + str(total1))


    #we do the smal one now

    #we compute the covariance matrix

    stime = time.process_time()
    S = 1/phi.shape[1] *  np.transpose(phi) @ phi
    print(S.shape)


    #eigvals = np.diag(S)
    #eigvecs = np.eye(S.shape[0])
    eigvals, eigvecs = np.linalg.eigh(S)

    # Compute actual eigenfaces
    eigvecs = phi @ eigvecs
    eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)
    
    total2 = time.process_time() - stime
    print("Time to compute the small cov matrix + eig decomposition : " + str(total2))

    print("Radio between the times : " + str(total2/total1))


    #make the eigvalues positivr
    for i in range(len(eigvals)):
        if eigvals[i] < 0:
            eigvals[i] *= -1
            eigvecs[:,i] = eigvecs[:,i] * -1

    for i in range(len(eigvalsBig)):
        if eigvalsBig[i] < 0:
            eigvalsBig[i] *= -1
            eigvecsBig[:,i] = eigvecsBig[:,i] * -1



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
        if not np.isclose(i, 0, atol=1e-8):
            count += 1

    print("Out of " + str(len(eigvalsBig))  + " eigenvals, only " + str(count) + " are non-zero")

    #how many non_zeros ?
    count = 0
    for i in eigvals:
        if not np.isclose(i, 0, atol=1e-8):
            count += 1

    print("Out of " + str(len(eigvals))  + " eigenvals, only " + str(count) + " are non-zero")

    #for i in range(20):
        #utils.showImage(eigvecs[:,i], "Eigenface " + str(i))
        #utils.show2images(eigvecs[:,i],  eigvecsBig[:,i], "Small ", "Standard ") #utils.showImages(eigvecsBig[:, :20])


    
    #check if they are the same

    sameEigenVals = 0
    for i, (val_small, val_big) in enumerate(zip(eigvals, eigvalsBig)):
        if np.isclose(val_small, val_big, atol=1e-8):
            sameEigenVals+=1
           


    invertedEig=0
    sameEigenVecs = 0
    for i in range(len(eigvals)):
        if np.allclose(eigvecs[:, i], eigvecsBig[:, i], atol=1e-8):
            sameEigenVecs+=1
        elif np.allclose(eigvecs[:, i], -eigvecsBig[:, i], atol=1e-8):
            invertedEig+=1

    print("They share : " + str(sameEigenVals) + " eigenvals (sorted ..) , and " + str(sameEigenVecs) + "  eigenvecs")
    print("Of the eigvectors they don't share, " + str(invertedEig) + " are just the same but inverted")


    #check if the sums eigvals are the same

    sum1 = 0
    for i in eigvals:
        sum1+=i
    sum2 = 0
    for i in eigvalsBig:
        sum2+=i
    print("The trace of the AAT is : " + str(sum2) + ", and of ATA is : "  + str(sum1) )



    #now we do the projections
    
    #how much variance do we keep?



    plt.bar(np.arange(0,30), eigvals[:30])
    plt.xlabel("Eigenvectors", fontsize=22)
    plt.ylabel("Eigenvalues (variance granted)", fontsize=22)
    plt.show()

    for k in range(30):
        sum2 = 0
        for i in range(k):
            sum2+= eigvals[i]
        print("The explained variacne of the first " + str(k) + " eigenvectors is of " + str(sum2/sum1) + "%")



    eigIndexes = [3,24,124]
    imageToPick = 355
    recons = [training[:,imageToPick]]
    titles = ["original (d = 2576) \nError = 0"]
    for i in range(len(eigIndexes)):
        tempEigvec = eigvecs[:, :eigIndexes[i]]
        #conversion
        w = np.transpose(phi[:, imageToPick]) @ tempEigvec
        recon = meanFace + tempEigvec @ w 
        recons.append(recon)

        sum2=0
        for j in range(eigIndexes[i]):
            sum2+= eigvals[j]

        #we define the error as eucliedian

        error = utils.mse(recon, training[:, imageToPick])

        titles.append("d = " + str(eigIndexes[i]) + "\n" + f"{100*sum2/sum1:.1f}%" + " Variance" + "\n" + "Error = " + f"{error:.1f}")

    #utils.showImages( (np.array(recons).T), titles)




    meanMatrix = np.repeat(meanFace[:, np.newaxis], test.shape[1], axis=1)
    phiTest = test-meanMatrix

    utils.findBestK(phiTest, meanFace, eigvecs)
    


main()
