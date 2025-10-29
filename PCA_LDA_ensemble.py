import numpy as np
import incremental_PCA
import random
import NN
import utils
import PCA_LDA



def fisherFaceRandom(dataX, dataY, Mpca0, Mpca1 , Mlda, SB, SW, mean):
    
    phiX = dataX - mean[:, None]
    eigvecsPCA, eigvalsPCA, pcatime = incremental_PCA.pca(phiX)
    indexes = list(range(Mpca0))
    #now we add Mpca1 other random directions
    toAdd = random.sample(range(Mpca0, eigvecsPCA.shape[1]-1), Mpca1)
    indexes += toAdd
    eigvecsPCA = eigvecsPCA[:, indexes]

    #W = eigvecsPCA.T @ phiX

    Wpca = eigvecsPCA

    A = np.linalg.pinv(Wpca.T @ SW @ Wpca) #TODO watch out here
    B = Wpca.T @ SB @ Wpca

    eigvals, eigvecs = np.linalg.eig(A @ B)
    
    #Sw_pca = Wpca.T @ SW @ Wpca
    #Sb_pca = Wpca.T @ SB @ Wpca

    #eigvals, eigvecs = eigh(Sb_pca, Sw_pca)

    indx = np.argsort(eigvals.real)[::-1]
    Wlda = eigvecs[:, indx].real
    Wlda = Wlda[:, :Mlda]

    Wopt = Wpca @ Wlda

    return Wopt, mean, phiX, SB, SW, Wpca, eigvecs[:, indx].real


def get_machine(dataX,dataY,Mpca0, Mpca1, Mlda, SB, SW, mean):

    W, mean, trainingPhi, _ ,_ ,_ ,_ = fisherFaceRandom(dataX, dataY, Mpca0, Mpca1 , Mlda, SB, SW, mean)
    classifier = NN.NNPCAClassifier(W.T @ trainingPhi, dataY, mean, W)

    return classifier
            

def get_committee(dataX, dataY, n, Mpca0, Mpca1, Mlda):
    SB, SW, mean = PCA_LDA.getScatterImagesAndMean(dataX,dataY)

    return [get_machine(dataX,dataY,Mpca0,Mpca1, Mlda, SB,SW, mean) for i in range(n)]



def main():
    X, y = utils.getImages()

    training, test, trainingY, testY = utils.separateTrainingTestQ1(X,y)

    validation, validationY, test, testY = utils.separateTrainingTest(test, testY, test.shape[1]//2) #half becomes validation and half becomes test  

    comm = get_committee(training, trainingY, 20, 30,20, 36)


    classifier = NN.Committee(comm)

    print(utils.findTestAccuracy(classifier, test, testY))

main()



