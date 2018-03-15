'''
Coding a K-Nearst Neigbours classifier from scratch (variable k)
'''

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from scipy.spatial import distance
from collections import Counter

# calculate distance between data points


def euc(a, b):
    return distance.euclidean(a, b)

# DIY KNN classifier


class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test, k=1):
        predictions = []
        for row in X_test:
            label = self.closest(row, k)
            predictions.append(label)
        return predictions

    def closest(self, row, k):
        best_dist = [(euc(row, self.X_train[0]), 0)] * k
        for i in range(1, len(self.X_train)):
            dist = euc(self.X_train[i], row)
            if dist < best_dist[k-1][0]:
                best_dist[k-1] = (dist, i)
                best_dist.sort()

        results = []
        for item in best_dist:
            results.append(self.y_train[item[1]])
        c = Counter(results).most_common()
        majority = c[0][0]

        return majority


# load iris dataset
iris = load_iris()

# split test/training data
X = iris.data
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# training
clf = ScrappyKNN()
clf.fit(X_train, y_train)

# classify
print(accuracy_score(y_test, clf.predict(X_test, 5)))
