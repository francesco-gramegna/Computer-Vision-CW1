import math
import numpy as np
import utils


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
   


def fuse_pca(mean1, N1, P1, A1,Cov1, mean2, N2, P2, A2, Cov2):
    N3 = N1+N2
    mean3 = (N1*mean1 + N2*mean2)/(N3)

    Cov3 = N1/N3 * Cov1 + N2/N3 * Cov2 + (N1*N2)/N3**2 * np.array((mean1-mean2)) @ np.array((mean2-mean1)).T

    Concat = np.concat(P1, P2, mean1-mean2)

    Q, R = np.linalg.qr(Concat)

    tol = 1e-10
    mask = np.abs(np.diag(R)) > tol
    PHI = Q[:, mask]

    SPHI = PHI.T @ Cov3 @ PHI

    eig, R = np.linalg.eigh(SPHI)

    #convert to pos
    for i in range(len(eig)): #maybe shape here
        if eig[i] < 0:
            R[i] *= -1
            eig[:,i] *= -1

    sort_indices = np.argsort(eig)[::-1]

    A3 = eig[sort_indices]
    A3 = np.diag(A3)

    R = R[:, sort_indices]

    P3 = PHI @ R

    return mean3, N3, P3, A3, Cov3
    

def main():
    X, y = utils.getImages() 

    training, test, trainingY, testY = utils.separateTrainingTestQ1(X, y)

    dataX, dataY = utils.separateTrainingTestQ2(training, trainingY)

    mean0 = utils.getAverageColumn(dataX[0])
    meanMatrix = np.repeat(mean0[:, np.newaxis], dataX[0].shape[1], axis=1)

    phi0 = dataX[0] - meanMatrix
    val0, vec0 = pca(phi0) 

    mean1 = utils.getAverageColumn(dataX[1])
    meanMatrix = np.repeat(mean1[:, np.newaxis], dataX[1].shape[1], axis=1)

    phi1 = dataX[1] - meanMatrix
    
    val1, vec1 = pca(phi0) 


    meanC, _, vecC, valC, _ = fuse_pca(mean0, phi0.shape[1], vec0, val0, vec0 @ val0 @ vec0.T, mean1, phi1.shape[1], vec1 , val1, vec1 @ val1 @ vec1.T )





main()
