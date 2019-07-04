"""Iris Flower Classification
Content of the dataset - 150 entries under 5 attributes,
Petal length, Petal width, Sepal length, Sepal width and Class"""

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

# Pre process the data
DATA = pd.read_csv("IRIS.csv")
X = DATA.iloc[:, :-1].values
Y = DATA.iloc[:, -1].values
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, Y, test_size=0.2)

SC = StandardScaler()
X_TRAIN = SC.fit_transform(X_TRAIN)
X_TEST = SC.transform(X_TEST)

# Classifiers
classifiers = []
XGB_CLASS = XGBClassifier()
SVM_CLASS = svm.SVC()
TREE_CLASS = tree.DecisionTreeClassifier()
FOREST_CLASS = RandomForestClassifier(n_estimators=100)
NAIVE_CLASS = GaussianNB()
K_MEANS_CLASS = KNeighborsClassifier()
REGRESSION_CLASS = LogisticRegression(solver="lbfgs", multi_class="auto")

classifiers.append(XGB_CLASS)
classifiers.append(SVM_CLASS)
classifiers.append(TREE_CLASS)
classifiers.append(FOREST_CLASS)
classifiers.append(NAIVE_CLASS)
classifiers.append(K_MEANS_CLASS)
classifiers.append(REGRESSION_CLASS)

accuracy_list = []
cm_list = []

for clf in classifiers:
    clf.fit(X_TRAIN, Y_TRAIN)
    y_pred = clf.predict(X_TEST)
    acc = accuracy_score(Y_TEST, y_pred)
    cm = confusion_matrix(Y_TEST, y_pred)
    accuracy_list.append([acc])
    cm_list.append([cm])

most_accurate_class = classifiers[accuracy_list.index(max(accuracy_list))]
accuarcy_percent = max(accuracy_list)
print("The most accurate classifier is the", most_accurate_class, " with and accuracy of",
      accuarcy_percent)
