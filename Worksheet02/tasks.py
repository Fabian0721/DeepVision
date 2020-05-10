import numpy as np
import sklearn.datasets
import sklearn.neighbors
import matplotlib.pyplot as plt


class KNN(object):
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, x, y):
        self.x = x
        self.y = y

    def kneighbors(self, xquery):
        # TODO return indices of nearest neighbor points and distances to them
        pass

    def predict(self, xquery):
        # TODO return predicted label for each query point
        pass


def task1():
    # get data
    n = 1000
    n_train = 900
    n_test = n - n_train
    x, y = sklearn.datasets.make_moons(n_samples=n, noise=0.2,
                                       random_state=0)
    xtrain, ytrain = x[:n_train,...], y[:n_train,...]
    xtest, ytest = x[n_train:,...], y[n_train:,...]

    # TODO visualize data via scatterplot

    # TODO for k=5 check that our implementation predicts the same as that of
    # sklearn.
    k = 5
    sknn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    knn = KNN(n_neighbors=k)

    # analyze different values of k
    ks = [2**i for i in range(10)]
    for k in ks:
        # TODO fit and evaluate accuracy on test data
        knn = KNN(n_neighbors=k)

        # TODO plot decision boundary
        N = 100
        x = np.linspace(-1.5, 2.5, N)
        y = np.linspace(-1.0, 1.5, N)


def task2():
    data = sklearn.datasets.load_digits()
    x, y = (data.images/16.0).reshape(-1, 8*8), data.target
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(
        x, y, test_size=0.25, shuffle=True, random_state=0)

    # TODO analyze accuracy for different values of k
    ks = [2**i for i in range(4)]

    # TODO plot nearest neighbors


def make_data(noise=0.2, outlier=1):
    prng = np.random.RandomState(0)
    n = 500

    x0 = np.array([0,0])[None,:] + noise*prng.randn(n, 2)
    y0 = np.ones(n)
    x1 = np.array([1,1])[None,:] + noise*prng.randn(n, 2)
    y1 = -1*np.ones(n)

    x = np.concatenate([x0,x1])
    y = np.concatenate([y0,y1]).astype(np.int32)

    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(
        x, y, test_size=0.1, shuffle=True, random_state=0)
    xplot, yplot = xtrain, ytrain

    outlier = outlier*np.array([1,1.75])[None,:]
    youtlier = np.array([-1])
    xtrain = np.concatenate([xtrain, outlier])
    ytrain = np.concatenate([ytrain, youtlier])
    return xtrain, xtest, ytrain, ytest, xplot, yplot


class LinearLeastSquares(object):
    def fit(self, x, y):
        # TODO find minimizer of least squares objective
        pass

    def predict(self, xquery):
        # TODO implement prediction using linear score function
        pass


def task3():
    # get data
    for outlier in [1, 2, 4, 8, 16]:
        # get data. xplot, yplot is same as xtrain, ytrain but without outlier
        xtrain, xtest, ytrain, ytest, xplot, yplot = make_data(outlier=outlier)
        # TODO visualize xtrain via scatterplot

        lls = LinearLeastSquares()
        lls.fit(xtrain, ytrain)
        # TODO evaluate accuracy and decision boundary of LLS
        N = 100
        x = np.linspace(-1.0, 2.0, N)
        y = np.linspace(-1.0, 2.0, N)

        svm = sklearn.svm.LinearSVC()
        svm.fit(xtrain, ytrain)
        # TODO evaluate accuracy and decision boundary of SVM


if __name__ == "__main__":
    #task1()
    #task2()
    #task3()