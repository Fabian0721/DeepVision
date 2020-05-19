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
    # plt.scatter(x[:, 0], x[:, 1], s=40, c=y)
    # plt.show()

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

        prediction = knn.predict(xtest)
        # Compute accuracy.
        accuracy = knn.score(prediction, ytest)
        accuracy_array.append(accuracy)

        # Plot decision boundary
        # TODO plot decision boundary
        N = 100
        x = np.linspace(-1.5, 2.5, N)
        y = np.linspace(-1.0, 1.5, N)

        plt.title("k-NN Decision Boundary")
        xx, yy = np.meshgrid(x, y)
        zz = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.contourf(xx, yy, zz)
        plt.scatter(xtest[:, 0], xtest[:, 1], c=ytest, alpha=0.8)
        plt.show()

    # Plot accuracy
    plt.title("k-NN: Varying Number of Neighbors")
    plt.plot(ks, accuracy_array)
    plt.xlabel("Number of neighbors")
    plt.ylabel("Accuracy")
    plt.show()


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

    # Plot nearest neighbors
    k = 8
    num_test = 10
    knn = KNN(n_neighbors=k)
    knn.fit(xtrain, ytrain)
    # print(knn.predict(xtest))
    # print(knn.kneighbors(xtest)[0])
    #
    # f, xarr = plt.subplots(2, 10)
    # images_and_labels = list(zip(data.images, data.target))
    # for ax, (image, label) in zip(xarr[0, :], images_and_labels[:xarr.shape[1]]):
    #     ax.set_axis_off()
    #     ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    #     ax.set_title(f"Training: {label}")

    index_images = []
    for i in ytest:
        if ytest[i] != knn.predict(xtest)[i]:
            index_images.append(i)
            if len(index_images) == num_test // 4:
                break

    for i in xrange(num_test - len(index_images)):
        index_images.append(i)

    print(knn.kneighbors(xtest)[0].shape)
    all_images = np.empty((num_test, k + 1, xtest.shape[1]))
    print(all_images.shape)
    for i in index_images:
        all_images[i][:k] = knn.kneighbors(xtest)[index_images[i]]
        all_images[i][k:] = xtest[index_images[i]]

    plt.title("k-Neighbors & Tests")
    plt.xlabel("")
    plt.ylabel("")
    plt.scatter(all_images[:][:][:, 0], all_images[:][:][:, 1])
    plt.show()


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
        """
        Build linear least weight vector W
        :param x: NxD matrix containing N attributes vectors for training
        :param y: NxK matrix containing N class vectors for training
        ------------------------------------------------------------------
        Derivation w*:
        L(w) = ||Xw-y||^2 = (Xw-y)^t(Xw-y)
        ∇_w L(w) = 2X^t(Xw-y) = 0
             ⇌ X^tXw = X^ty (X^tX is full ranked)
             ⇌ w* = (X^tX)^-1 X^ty
         ------------------------------------------------------------------
        Algorithm:
        1. Bias trick is applied
        2. wstar is calculated to minimize loss function
        3. Fields (bias, weights) are declared.
        The weights field is the vector that will be multiplied in the score function
        """

        num_samples, dim = np.shape(x)
        w0 = np.ones((num_samples, 1))
        data = (np.concatenate((w0, x), axis=1))
        self.wstar = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(data), data)), np.transpose(data)), y)
        self.bias = self.wstar[0]
        self.weight = [self.wstar[1], self.wstar[2]]

    def predict(self, xquery):
        # TODO implement prediction using linear score function
        """
        weight:     (dim x 1)
        bias:       (scalar)

        :param xquery: Input to be classified (num_samples x dim)
        :return: class prediction (+-1) values
        ------------------------------------------------------------------
        To calculate the score, the matrix of data (input) is multiplied with the vector of weights
        """
        num_samples, dim = np.shape(xquery)
        class_pred = np.zeros((num_samples, 1))

        y = np.matmul(xquery, self.weight)

        for i in xrange(0, num_samples):
            if y[i] < 0:
                class_pred[i] = -1
            else:
                class_pred[i] = +1

        return class_pred


def task3():
    # get data
    for outlier in [1, 2, 4, 8, 16]:
        # get data. xplot, yplot is same as xtrain, ytrain but without outlier
        xtrain, xtest, ytrain, ytest, xplot, yplot = make_data(outlier=outlier)

        # Plot XTrain
        plt.title("XTrain")
        plt.scatter(xtrain[:, 0], xtrain[:, 1])
        # plt.show()

        # Accuracy & Plot of LLS
        lls = LinearLeastSquares()
        lls.fit(xtrain, ytrain)
        prediction_lls = lls.predict(xtest)
        accuracy_lls = metrics.accuracy_score(prediction_lls, ytest)
        print("Accuracy Linear Least Squares: ", accuracy_lls)
        N = 100
        x = np.linspace(-1.0, 2.0, N)
        y = np.linspace(-1.0, 2.0, N)
        plt.title("LLS Decision Boundary")
        xx, yy = np.meshgrid(x, y)
        zz = lls.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.contourf(xx, yy, zz)
        plt.scatter(xtest[:, 0], xtest[:, 1], c=ytest, alpha=0.8)
        plt.show()

        # Accuracy & Plot of SVM
        svm = sklearn.svm.LinearSVC()
        svm.fit(xtrain, ytrain)
        prediction_svm = svm.predict(xtest)
        accuracy_svm = metrics.accuracy_score(prediction_svm, ytest)
        print("Accuracy Support Vector Machine: ", accuracy_svm)
        N = 100
        x = np.linspace(-1.0, 2.0, N)
        y = np.linspace(-1.0, 2.0, N)
        plt.title("SVM Decision Boundary")
        xx, yy = np.meshgrid(x, y)
        zz = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        plt.contourf(xx, yy, zz)
        plt.scatter(xtest[:, 0], xtest[:, 1], c=ytest, alpha=0.8)
        plt.show()


if __name__ == "__main__":
    task1()
    # task2()
    # task3()
