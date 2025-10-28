import math
import numpy as np
import utils
import time


def pca(phi):

    S = 1/phi.shape[1] *  np.transpose(phi) @ phi

    eigvals, eigvecs = np.linalg.eigh(S)
    # Compute actual eigenfaces
    eigvecs = phi @ eigvecs
    eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=0)

    #sort? yes

    for i in range(len(eigvals)):
        if eigvals[i] < 0:
            eigvals[i] *= -1
            eigvecs[:,i] = eigvecs[:,i] * -1


    indexes = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:,indexes]
    eigvals = eigvals[indexes]

    return eigvecs, eigvals

def big_cov(eigvecs, eigvals):
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def fuse_pca(meanX, evecX, evalX,Cov1, N, meanY, evecY, evalY,Cov2, M):

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

    Bvals, Bvecs = np.linalg.eigh(B)

    Rot = Bvecs

    idx = np.argsort(Bvals)[::-1]
    Bvals = Bvals[idx]
    Rot = Rot[:, idx]

    W = PHI @ Rot

    return (mean3, W, Bvals, Cov3, M+N) 



def initial_pca(X):
    mean0 = utils.getAverageColumn(X)
    meanMatrix = np.repeat(mean0[:, np.newaxis], X.shape[1], axis=1)

    phi0 = X - meanMatrix
    vec0, val0 = pca(phi0) 
    N0 = phi0.shape[1]

    cov = big_cov(vec0, val0)

    return mean0, vec0, val0, cov, X.shape[1]

    

def main():
    X, y = utils.getImages() 

    training, test, trainingY, testY = utils.separateTrainingTestQ1(X, y)

    dataX, dataY = utils.separateTrainingTestQ2(training, trainingY)

    pca0 = initial_pca(dataX[0])
    pca1 = initial_pca(dataX[1])
    pca2 = initial_pca(dataX[2])
    pca3 = initial_pca(dataX[3])

    #add pca1 to pca0


    pca01 = fuse_pca(*pca0, *pca1)
    pca02 = fuse_pca(*pca01, *pca2)
    pca03 = fuse_pca(*pca02, *pca3)

    mean, vec, val, cov, tot = pca03



    meanMatrix = np.repeat(mean[:, np.newaxis], test.shape[1], axis=1)
    phiTest = test - meanMatrix
    #utils.showImages(np.array([vec[:,0],vec[:,0], mean0, meanMerged]).T, np.array(["","", "mean0", "meanMerged"]))

    #compute training representation

    
    utils.findBestK(phiTest, mean, vec)
    

    

main()
