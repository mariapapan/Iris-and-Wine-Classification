#Iris and wine classification

from sklearn.datasets import load_iris, load_wine

#comment the next three lines and uncomment the three followings for wine classification
iris = load_iris()
X = iris.data
y = iris.target

'''wine = load_wine()
X = wine.data
y = wine.target'''

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

from sklearn.metrics import accuracy_score, f1_score

print("\n----------Accuracies---------\n")

#----------Decision trees----------
from sklearn import tree 

clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(X_train, y_train)

predictions_tree = clf_tree.predict(X_test)

print("Decision Trees:", accuracy_score(y_test, predictions_tree))

#----------k-NN classifier----------
from sklearn.neighbors import KNeighborsClassifier

clf_kNN = KNeighborsClassifier()
clf_kNN.fit(X_train, y_train)

predictions_kNN = clf_kNN.predict(X_test)

print("k-NN classifier:", accuracy_score(y_test, predictions_kNN))

#-----------Naive bayes-------------
from sklearn.naive_bayes import GaussianNB

clf_gnb = GaussianNB()
clf_gnb.fit(X_train, y_train)

predictions_gnb = clf_gnb.predict(X_test)

print("Naive Bayes:", accuracy_score(y_test, predictions_gnb))

#----------Random forests------------
from sklearn.ensemble import RandomForestClassifier

clf_random_forests = RandomForestClassifier()
clf_random_forests.fit(X_train, y_train)

predictions_random_forests = clf_random_forests.predict(X_test)

print("Random forests:", accuracy_score(y_test, predictions_random_forests))




