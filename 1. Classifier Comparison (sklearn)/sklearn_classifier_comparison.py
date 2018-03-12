"""
Reads Iris flowers' pedal sizes and classifies them by species,
comparing 4 different classifier methods.
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import numpy as np


# load iris dataset
iris = load_iris()

# # example use of dataset
# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data[0])
# print(iris.target[0])
# for i in range(len(iris.data)):
#     print('Example {}: Features: {}, Label: {}'.format(
#         i+1, iris.data[i], iris.target_names[iris.target[i]]))

# training data
test_idx = [i*5 for i in range(30)]  # 10 out of each species 50 entries
training_data = np.delete(iris.data, test_idx, axis=0)
training_target = np.delete(iris.target, test_idx)

# testing data
test_data = iris.data[test_idx]
test_target = iris.target[test_idx]

# let the traning commence!
clf_tree = DecisionTreeClassifier()
clf_tree.fit(training_data, training_target)

clf_svc = SVC(kernel='linear')
clf_svc.fit(training_data, training_target)

clf_prctrn = Perceptron()
clf_prctrn.fit(training_data, training_target)

clf_knn = KNeighborsClassifier()
clf_knn.fit(training_data, training_target)

# prediction
pred_tree = clf_tree.predict(test_data)
pred_svc = clf_svc.predict(test_data)
pred_prcptrn = clf_prctrn.predict(test_data)
pred_knn = clf_knn.predict(test_data)

# accuracy
print('Decision Tree Accuracy: {}'.format(
    accuracy_score(test_target, pred_tree)))
print('SVC Linear Accuracy: {}'.format(accuracy_score(test_target, pred_svc)))
print('Perceptron Accuracy: {}'.format(
    accuracy_score(test_target, pred_prcptrn)))
print('KNN Accuracy: {}'.format(accuracy_score(test_target, pred_knn)))


# print("Prediction: {}".format(
#     ", ".join(iris.target_names[np.argmax(pred_svc, axis=1)])))
# print("Pred. prob: {}".format(np.max(pred_svc, axis=1)))
# print("Actual spc: {}".format(", ".join(iris.target_names[test_target])))
