import math
import NN
import numpy as np
import utils
import time


def pca(phi):

    S = 1/phi.shape[1] *  np.transpose(phi) @ phi

    #print(S.shape)


    stime = time.time()

    eigvals, eigvecs = np.linalg.eigh(S)

    # Compute actual eigenfaces
    eigvecs = phi @ eigvecs
    eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)

    total_time = time.time() - stime

    #sort? yes

    for i in range(len(eigvals)):
        if eigvals[i] < 0:
            eigvals[i] *= -1
            eigvecs[:,i] = eigvecs[:,i] * -1


    indexes = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:,indexes]
    eigvals = eigvals[indexes]



    return eigvecs, eigvals, total_time

def big_cov(eigvecs, eigvals):
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def fuse_pca(meanX, evecX, evalX,Cov1, N,time1, meanY, evecY, evalY,Cov2, M, time2):

    mean3 = (N*meanX + M*meanY)/(M+N)
    #step1 : build A : orthonrmal basis 

    bad_PHI = np.column_stack([evecX, evecY, (meanX - meanY)])

    #we have to orthonormalize and remove the zeros from bad_PHI

    Q, R = np.linalg.qr(bad_PHI)

    tol = 1e-10
    keep = np.abs(np.diag(R)) > tol
    PHI = Q[:, keep]


    mean_diff = (meanX - meanY).reshape(-1, 1)
    Cov3 = N/(M+N) * Cov1 + M/(M+N) * Cov2 + (N*M)/(M+N)**2 * (mean_diff @ mean_diff.T)

    B = PHI.T @ Cov3 @ PHI
    print("B shape")
    print(B.shape)

    stime = time.time()
    Bvals, Bvecs = np.linalg.eigh(B)
    end_time = time.time() - stime

    Rot = Bvecs

    idx = np.argsort(Bvals)[::-1]
    Bvals = Bvals[idx]
    Rot = Rot[:, idx]

    W = PHI @ Rot

    return (mean3, W, Bvals, Cov3, M+N, max(time1,time2) + end_time) 



def initial_pca(X):
    mean0 = utils.getAverageColumn(X)
    meanMatrix = np.repeat(mean0[:, np.newaxis], X.shape[1], axis=1)

    phi0 = X - meanMatrix
    vec0, val0, time = pca(phi0) 
    N0 = phi0.shape[1]

    cov = big_cov(vec0, val0)

    return mean0, vec0, val0, cov, X.shape[1], time

    

def main():
    X, y = utils.getImages() 

    training, test, trainingY, testY = utils.separateTrainingTestQ1(X, y)

    dataX, dataY = utils.separateTrainingTestQ2(training, trainingY)

    print(dataX[0].shape)

    pca0 = initial_pca(dataX[0])
    pca1 = initial_pca(dataX[1])
    pca2 = initial_pca(dataX[2])
    pca3 = initial_pca(dataX[3])

    #add pca1 to pca0

    trucation = 104

    pca0 = list(pca0)
    pca1 = list(pca1)
    pca2 = list(pca2)
    pca3 = list(pca3)

    pca0[1] = pca0[1][:, :trucation]
    pca1[1] = pca1[1][:, :trucation]
    pca2[1] = pca2[1][:, :trucation]
    pca3[1] = pca3[1][:, :trucation]
    

    pca01 = fuse_pca(*pca0, *pca1)
    pca02 = fuse_pca(*pca01, *pca2)
    pca03 = fuse_pca(*pca02, *pca3)

    mean, vec, val, cov, tot, total_time = pca03



    meanMatrix = np.repeat(mean[:, np.newaxis], test.shape[1], axis=1)
    phiTest = test - meanMatrix

    meanMatrix = np.repeat(mean[:, np.newaxis], training.shape[1], axis=1)
    phiTraining = training - meanMatrix

    #utils.showImages(np.array([vec[:,0],vec[:,0], mean0, meanMerged]).T, np.array(["","", "mean0", "meanMerged"]))

    #compute training representation

    pcaBIG = initial_pca(training)
    meanBig, bigVec, bigVal, _, _ , endTimeBig = pcaBIG #we compute the big pca so that we can compare

    print("The time for the eigendecompositions took " + str(total_time))
    print("The time for the eigendecompositions of the normal PCA took " + str(endTimeBig))
    
    #bestNumberForIPCA, results = utils.findBestK(phiTest, mean, vec)

    #bestNumberForPCA , resultsPCA = utils.findBestK(phiTest, meanBig, bigVec)
    bestNumberForPCA = 39
    bestNumberForIPCA = 39
    
    vecKept = vec[:, :bestNumberForIPCA]
    bigVecKept = bigVec[:,:bestNumberForPCA] 

    W = vecKept.T @ phiTraining


    meanMatrix = np.repeat(mean[:, np.newaxis], phiTraining.shape[1], axis=1)
    R = meanMatrix + vecKept @ W

    #utils.show2images(training[:,0],R[:,0], "", "")

    classifierIPCA = NN.NNPCAClassifier(W, trainingY, mean, vecKept)

    W = bigVecKept.T @ phiTraining
    classifierPCA = NN.NNPCAClassifier(W, trainingY, meanBig, bigVecKept)

    

    meanSmall, vecSmall, valSmall, _,_, _ = pca01

    meanMatrix = np.repeat(meanSmall[:, np.newaxis], training.shape[1], axis=1)
    phiTrainingSmall = training - meanMatrix

    vecSmallKept = vecSmall[:, :bestNumberForIPCA]
    W = vecSmallKept.T @ phiTraining 

    classifierIPCASmall = NN.NNPCAClassifier(W, trainingY, meanSmall, vecSmallKept)

    ipca_accuracy = utils.findTestAccuracy(classifierIPCA, test, testY)

    pca_accuracy = utils.findTestAccuracy(classifierPCA, test,testY)

    ipcasmall_accuracy = utils.findTestAccuracy(classifierIPCASmall, test,testY)

    print("IPCA : " + str(ipca_accuracy))
    print("IPCA Small : " + str(ipcasmall_accuracy))
    print("PCA : " + str(pca_accuracy))

    #analysis of difference in the eigenvectors

    #results,times= utils.testDifferentKValues(104, phiTraining, trainingY, test, testY, pca0, pca01, pca02, pca03, pcaBIG)

    #utils.plotTestAccuracyTestQ2(104, results, times)

    #tesing the different reconstruction error for k = 38

    pcas = [pca0, pca01,pca02,pca03, pcaBIG]

    recons = [test[:,10]]
    titles = ['original', '104', '208', '316', 'ipca total', 'batch']
    for pca in pcas:
        mean, W, _, _,_, _ = pca
        W = W[:, :100]
        #print("Reconstruction error : " + str(utils.findMeanReconstructionError(W,mean,training)))
        X = W.T @ test[:, 10]
        
        R = mean + W @ X

        recons.append(R) 

    utils.showImages(np.array(recons).T, titles)




if __name__ == "__main__":
    main()
