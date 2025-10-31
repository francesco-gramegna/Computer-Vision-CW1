import math
import numpy as np
import utils


class NNClassifier():
    def __init__(self, A, Y):
        self.A = A
        self.Y = Y

    def classify(self, point):
        min_dist = float('inf')
        _class = self.Y[0]

        for i in range(self.A.shape[1]):
            dist = utils.mse(self.A[:, i], point)
            if dist < min_dist:
                _class = self.Y[:,i]
                min_dist = dist

        return _class


class NNPCAClassifier():
    #W is already the latent space representation here
    def __init__(self, W, Y, meanFace, eigenvecs):
        self.W = W
        self.Y = Y
        self.mean = meanFace
        self.eigenvecs = eigenvecs
        self.classifier = NNClassifier(W, Y)


    def classify(self, point):
        point = point - self.mean
        w = np.transpose(point) @ self.eigenvecs        
        return self.classifier.classify(w)
    

class Committee():
    def __init__(self, comm):
       self.comm = comm 

    def classify(self, point, reality):
        classes = {}

        individualTrue = 0
        for c in self.comm:
            h = c.classify(point)
            if(h == reality):

                individualTrue+=1
            h = h[0]

            if h in classes:
                classes[h] += 1
            else:
                classes[h] = 1

        return np.array([max(classes, key=classes.get)]), individualTrue / len(self.comm)



