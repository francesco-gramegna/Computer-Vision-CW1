import numpy as np
import incremental_PCA
import random
import NN
import utils
import PCA_LDA

from multiprocessing import Pool



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

def get_committee_fast(dataX, dataY, n, Mpca0, Mpca1, Mlda, SB,SW, mean):
    return [get_machine(dataX,dataY,Mpca0,Mpca1, Mlda, SB,SW, mean) for i in range(n)]

def test_Param(_in):
    dataX, dataY, n, Mpca0, Mpca1, Mlda, SB,SW, mean, validation, validationY = _in
    comm = get_committee_fast(dataX,dataY, n ,Mpca0, Mpca1, Mlda, SB,SW, mean)
    classifier = NN.Committee(comm)
    print("tested param")
    return (Mpca0, Mpca1, utils.findTestAccuracy(classifier, validation, validationY))

def get_comm_machines_all_params(dataX, dataY,n, Mlda,SB,SW,mean, validation, validationY): #here Mlda is set to be 37

    params = []
        
    MpcaTot = 50
    for sep in range(2, MpcaTot):
        Mpca0 = sep
        Mpca1 = MpcaTot - sep
        params.append((dataX, dataY, n, Mpca0, Mpca1, Mlda, SB,SW, mean, validation, validationY))

    with Pool(processes=18) as pool:
        results = pool.map(test_Param, params)

    np.save("Ensemble_fisher.npy", results)
    return results

           
def testRandom(i):

    X, y = utils.getImages()
    training, test, trainingY, testY = utils.separateTrainingTestQ1(X,y)

    validation, validationY, test, testY = utils.separateTrainingTest(test, testY, test.shape[1]//2) #half becomes validation and half becomes test  

    SB, SW, mean = PCA_LDA.getScatterImagesAndMean(training,trainingY)

    random.seed(i)

    for _ in range(20):
        Mtot = random.randint(10, 130)
        Mpca0 = random.randint(1, Mtot)
        Mpca1 = Mtot - Mpca0
        Mlda = random.randint(10, Mtot)

        n = random.randint(5, 30)

        comm = get_committee_fast(training, trainingY, n, Mpca0, Mpca1, Mlda, SB,SW,mean)

        classifier = NN.Committee(comm)

        r = (n , Mpca0, Mpca1, Mlda, utils.findTestAccuracy(classifier, validation, validationY))
        print("Found : " + str(r))



def main():
    X, y = utils.getImages()

    training, test, trainingY, testY = utils.separateTrainingTestQ1(X,y)

    validation, validationY, test, testY = utils.separateTrainingTest(test, testY, test.shape[1]//2) #half becomes validation and half becomes test  


    SB, SW, mean = PCA_LDA.getScatterImagesAndMean(training,trainingY)

    #with Pool(processes=18) as pool:
        #results = pool.map(testRandom, [i for i in range(17)])


    comm = get_committee_fast(training, trainingY, 26, 41, 57, 97, SB,SW,mean)
    classifier = NN.Committee(comm)

    # Evaluate on the validation set
        
    new_acc = utils.findTestAccuracy(classifier, test, testY)
        

    #SB, SW, mean = PCA_LDA.getScatterImagesAndMean(training,trainingY)
    #get_comm_machines_all_params(training, trainingY,20, 37,SB,SW,mean, validation, validationY)

    #r = np.load("Ensemble_fisher.npy", allow_pickle=True)

    #PCA_LDA.plotAllResults(r)



main()



