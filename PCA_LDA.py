import numpy as np
import matplotlib.pyplot as plt
import time
import NN
import utils
import incremental_PCA
from multiprocessing import Pool


from scipy.linalg import eigh



def getScatterImagesAndMean(dataX, dataY):
    means = {}
    
    dataHashMap = {}

    dataY = np.ravel(dataY)

    for img,c in zip(dataX.T , dataY):

        if c in dataHashMap:
            dataHashMap[c].append(img)
            val,count = means[c]
            means[c] = val+img, count+1
        else:
            dataHashMap[c] = [img]
            val,count = img, 1
            means[c] = (val,count)
    
    N = {}
    for c in means:
        val,count = means[c]
        N[c] = count
        means[c] = val/count


    #compute total mean

    mean = utils.getAverageColumn(dataX) 

    SB = np.zeros((dataX.shape[0], dataX.shape[0]))
    for c in N:
        SB += N[c]*np.array(means[c] - mean).reshape(-1,1) @ np.array(means[c]-mean).reshape(-1,1).T
        
    SW = np.zeros((dataX.shape[0], dataX.shape[0]))
    for c in N:
        for i in dataHashMap[c]:
            SW += (i - means[c]).reshape(-1,1) @ (i - means[c]).reshape(-1,1).T

    return SB, SW, mean


def fisherFace(dataX, dataY, Mpca, Mlda, SB, SW, mean):
    
    if(Mpca >= dataX.shape[1] - 1):
        print("Mpca value too high !")
        #compute class means
    #compute W , by pca
    phiX = dataX - mean[:, None]
        
    eigvecsPCA, eigvalsPCA, pcatime = incremental_PCA.pca(phiX)
    eigvecsPCA = eigvecsPCA[:, :Mpca]

    #W = eigvecsPCA.T @ phiX

    Wpca = eigvecsPCA

    #A = np.linalg.pinv(Wpca.T @ SW @ Wpca) #TODO watch out here
    #B = Wpca.T @ SB @ Wpca

    #eigvals, eigvecs = np.linalg.eig(A @ B)
    
    Sw_pca = Wpca.T @ SW @ Wpca
    Sb_pca = Wpca.T @ SB @ Wpca

    eigvals, eigvecs = eigh(Sb_pca, Sw_pca)

    #for i in range(len(eigvals)):
    #    if(eigvals[i] < 0):
    #        eigvals[i] *= -1
    #        eigvecs[:, i] *= -1


    indx = np.argsort(eigvals.real)[::-1]
    Wlda = eigvecs[:, indx].real
    Wlda = Wlda[:, :Mlda]

    Wopt = Wpca @ Wlda

    return Wopt, mean, phiX, SB, SW, Wpca, eigvecs[:, indx].real


def test24pca(_in):
    _from, _max, SB,SW, mean, training, trainingY, test, testY = _in
    res = []

    print("Starting from " + str(_from))
    stime = time.time()

    for pca in range(_from, min(_from+24, _max)):
        _, mean, trainingPhi, _,_,Wpca, Wlda = fisherFace(training, trainingY, pca, 52, SB, SW, mean)
        for lda in range(1, 52):
            W = Wpca @ Wlda[:,:lda]
            classifier = NN.NNPCAClassifier(W.T @ trainingPhi, trainingY, mean, W)
            acc = utils.findTestAccuracy(classifier, test, testY)
            res.append((pca,lda, acc))
        print("Finished all lda for pca : " + str(pca)+ " in " + str(int(time.time()-stime)))

    print("FINISHED 24 pca computation ! in " + str(time.time() - stime))

    return res

def checkAllParameters(maxPca, maxLda, SB,SW, mean, training, trainingY, test, testY):

    params_list = [(i,415, SB,SW, mean, training, trainingY, test, testY ) for i in range(1, 415, 24)]

    with Pool(processes=17) as pool:
        results = pool.map(test24pca, params_list)
    print(len(results))

    r = []
    for res in results:
        for pca, lda, acc in res:
            r.append((pca, lda, acc))

    np.save("fisherface_res2.npy", r)

    return r


def plotAllResults(results):


    # Extract unique sorted hyperparameters
    xs = sorted(set(x for x, _, _ in results))
    ys = sorted(set(y for _, y, _ in results))

    # Map values to indices for fast lookup
    x_index = {x: i for i, x in enumerate(xs)}
    y_index = {y: i for i, y in enumerate(ys)}

    # Create heatmap array
    Z = np.zeros((len(ys), len(xs)))

    # Fill it
    for x, y, v in results:
        Z[y_index[y], x_index[x]] = v


    max_val = np.max(Z)
    max_idx = np.unravel_index(np.argmax(Z), Z.shape)
    best_y = ys[max_idx[0]]
    best_x = xs[max_idx[1]]

    plt.scatter(best_x, best_y, color='red', s=160, marker='x', label='Best accuracy : ' + f"{max_val:.1f}" + f"at Mpca={max_idx[1]}, Mlda={max_idx[0]}")
    #plt.text(best_x, best_y, f'{max_val:.1f}', color='red', ha='left', va='bottom', fontsize=0)
    plt.legend(loc='upper right')

    plt.imshow(
        Z,
        origin='lower',
        aspect='auto',
        cmap='cividis',
        extent=[min(xs), max(xs), min(ys), max(ys)]
    )
    plt.colorbar(label='Accuracy %')
    plt.xlabel('Mpca')
    plt.ylabel('Mlda')
    plt.title('Hyperparameter Heatmap')
    plt.show()


def main():

    X, y = utils.getImages()

    training, test, trainingY, testY = utils.separateTrainingTestQ1(X,y)

    #separate test into real test and validation set

    #validation, validationY, test, testY = utils.separateTrainingTest(test, testY, test.shape[1]//2) #half becomes validation and half becomes test  

    #rank of SB = 52
    #rank of SW = 416
    
    SB, SW, mean = getScatterImagesAndMean(training,trainingY)

    stime = time.time()
    
    W, mean, trainingPhi, SB, SW, Wpca, Wlda = fisherFace(training, trainingY, 416, 52, SB, SW, mean)

    classifier = NN.NNPCAClassifier(W.T @ trainingPhi, trainingY, mean, W)

    #print(utils.findTestAccuracy(classifier, validation, validationY))

    end_time = time.time()-stime
    print(end_time)

    res = checkAllParameters(416, 52, SB,SW, mean, training, trainingY, test, testY)

    #res = np.load("fisherface_res.npy", allow_pickle=True)

    plotAllResults(res)



main()
