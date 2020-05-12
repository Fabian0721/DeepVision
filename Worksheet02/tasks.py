import numpy as np
import sklearn.datasets
import sklearn.neighbors
import matplotlib.pyplot as plt
from sklearn import metrics
import collections
from ipython_genutils.py3compat import xrange



class KNN(object):
    def __init__(self, n_neighbors):
        self.n_neighbors = n_neighbors

    def fit(self, x, y):
        self.x = x  # Training data set
        self.y = y

    def kneighbors(self, xquery):
        """
        :param xquery: Data set to be compared with the training data set
        :return: Nearest neighbors
        """
        num_test = xquery.shape[0]

        nearest_kindexes = np.empty(shape=(self.n_neighbors, num_test), dtype='uint8')
        nearest_kneighbors = np.empty(shape=(self.n_neighbors, num_test))

        # Get euclidean distances
        distances = []
        for i in xrange(0, num_test):
            euclidean_distance = np.linalg.norm(self.x - xquery[i, :], axis=1)
            distances.append(euclidean_distance)

        # Get n_neighbor indexes of each xquerie with the smallest distance
        # min_kindexes has the indexes of the neighbors until the position n_neighbors with the smallest distance
        min_kindexes = np.argpartition(distances, self.n_neighbors)

        # nearest_kindexes contains the n_neighbors for each point/data test
        nearest_kindexes = np.empty((num_test, self.n_neighbors), dtype=int)
        for i in xrange(num_test):
            nearest_kindexes[i] = min_kindexes[i, :self.n_neighbors]

        nearest_kneighbors = np.empty((num_test, self.n_neighbors, self.x.shape[1]))

        # Get the neighbors in the training data set
        for i in xrange(num_test):
            index_in_selfx = nearest_kindexes[i][:]
            nearest_kneighbors[i][:] = self.x[index_in_selfx]

        return nearest_kneighbors

    def predict(self, xquery):
        """
        :param xquery: data set to be classified
        :return: Predicted label to the query
        """
        neighbors = KNN.kneighbors(self, xquery)
        num_test = xquery.shape[0]
        yprediction = np.zeros(num_test, dtype=self.y.dtype)

        """
        [[[ , ]
          [ , ]]]
          
        Positions 0, 0, x and 0, 1, x
        second axis indicates position in y array
        Objective: count the values indicated by the second axis in the split array 
        """

        # array_of_indices has the indices of the values that we are looking for in y.
        # The first dimension of array_of_indices corresponds to the number of test.
        array_of_indices = np.empty((num_test, self.n_neighbors), dtype=int)
        for i in xrange(num_test):
            split = np.split(neighbors, num_test)[i]
            for j in xrange(self.n_neighbors):
                # test_neighbor is selecting iteratively on of the neighbors:
                # [[ , ] [ , ] ... [ , ]] -> [ , ]
                test_neighbor = split[0][j]
                indices = np.where(self.x == test_neighbor)
                array_of_indices[i][j] = indices[0][0]

        yarr = []
        for k in array_of_indices:
            yarr.append(collections.Counter(self.y[k]).most_common(1)[0][0])

        yprediction = np.array(yarr)
        return yprediction

    def score(self, prediction, reality):
        return metrics.accuracy_score(prediction, reality)




def task1():
    # get data
    n = 1000
    n_train = 900
    n_test = n - n_train
    x, y = sklearn.datasets.make_moons(n_samples=n, noise=0.2,
                                       random_state=0)

    xtrain, ytrain = x[:n_train, ...], y[:n_train, ...]
    xtest, ytest = x[n_train:, ...], y[n_train:, ...]

    # Visualize data via scatterplot
    plt.scatter(x[:, 0], x[:, 1], s=40, c=y)
    plt.show()

    k = 5
    sknn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k)
    knn = KNN(n_neighbors=k)

    # Fit the classifier to the data
    sknn.fit(xtrain, ytrain)
    knn.fit(xtrain, ytrain)

    # Show first five model predictions on the test data
    print("sknn prediction: ", sknn.predict(xtest)[0:5])
    print("knn prediction: ", knn.predict(xtest)[0:5])

    # Check accuracy of model on the test data
    print("Accuracy with sknn: ", sknn.score(xtest, ytest))
    print("Accuracy with knn: ", knn.score(knn.predict(xtest), ytest))

    # Analyze different values of k
    ks = [2 ** i for i in range(10)]
    accuracy_array = []
    for k in ks:
        knn = KNN(n_neighbors=k)

        # Fit the classifier to the training set
        knn.fit(xtrain, ytrain)

        # Compute accuracy.
        accuracy = knn.score(knn.predict(xtest), ytest)
        accuracy_array.append(accuracy)

    # # TODO plot decision boundary
    # N = 100
    # x = np.linspace(-1.5, 2.5, N)
    # y = np.linspace(-1.0, 1.5, N)

    # Plot accuracy
    plt.title("k-NN: Varying Number of Neighbors")
    plt.plot(ks, accuracy_array, label="Testing Accuracy")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Accuracy")
    plt.show()

    # # Plot decision boundary
    # plt.title("k-NN Decision Boundary")
    # xx, yy = np.meshgrid(x, y)
    # plt.contourf(ks, accuracy_array, predictions)
    # plt.show()


def task2():
    data = sklearn.datasets.load_digits()

    x, y = (data.images / 16.0).reshape(-1, 8 * 8), data.target
    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(
        x, y, test_size=0.25, shuffle=True, random_state=0)

    ks = [2 ** i for i in range(4)]
    accuracy_array = []
    for k in ks:
        knn = KNN(n_neighbors=k)

        # Fit the classifier to the training set
        knn.fit(xtrain, ytrain)

        # Compute accuracy.
        accuracy = knn.score(knn.predict(xtest), ytest)
        accuracy_array.append(accuracy)

    # Plot accuracy
    plt.title("k-NN: Varying Number of Neighbors")
    plt.plot(ks, accuracy_array, label="Testing Accuracy")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Accuracy")
    plt.show()

    # TODO plot nearest neighbors


def make_data(noise=0.2, outlier=1):
    prng = np.random.RandomState(0)
    n = 500

    x0 = np.array([0, 0])[None, :] + noise * prng.randn(n, 2)
    y0 = np.ones(n)
    x1 = np.array([1, 1])[None, :] + noise * prng.randn(n, 2)
    y1 = -1 * np.ones(n)

    x = np.concatenate([x0, x1])
    y = np.concatenate([y0, y1]).astype(np.int32)

    xtrain, xtest, ytrain, ytest = sklearn.model_selection.train_test_split(
        x, y, test_size=0.1, shuffle=True, random_state=0)
    xplot, yplot = xtrain, ytrain

    outlier = outlier * np.array([1, 1.75])[None, :]
    youtlier = np.array([-1])
    xtrain = np.concatenate([xtrain, outlier])
    ytrain = np.concatenate([ytrain, youtlier])
    return xtrain, xtest, ytrain, ytest, xplot, yplot


class LinearLeastSquares(object):
    def fit(self, x, y):
        self.x = x
        self.y = y

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
    # task1()
    # task2()
    task3()
