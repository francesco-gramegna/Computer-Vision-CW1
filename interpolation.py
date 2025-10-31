#here we try to test out interpolation via PCA
import incremental_PCA
import utils
import numpy as np


def main():
    X, y = utils.getImages() 


    mean,vec,val,_,_,_ = incremental_PCA.initial_pca(X)

    #utils.showImages(X[:, :10], ["" for i in range(10)])

    W = vec.T @ X

    #we now mant to create the intermediary results using the first two images

    W1 = W[:,42]
    W2 = W[:,507]

    inters = np.array([x * W1 + (1-x) * W2 for x in np.linspace(0, 1, 10)]).T

    recons = vec @ inters

    print(recons.shape)

    utils.showImages(recons, ["img 1"] + [str(f"{i:.2f}") + "* img 2"  for i in np.linspace(0,1,10)[1:][:-1]] + ["img 2"], n_cols=5)

main()

