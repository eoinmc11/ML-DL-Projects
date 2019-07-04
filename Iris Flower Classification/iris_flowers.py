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
data = pd.read_csv("IRIS.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Classifiers
classifiers = []
xgb_class = XGBClassifier()
svm_class = svm.SVC()
tree_class = tree.DecisionTreeClassifier()
forest_class = RandomForestClassifier(n_estimators=100)
naive_class = GaussianNB()
kmeans_class = KNeighborsClassifier()
regression_class = LogisticRegression(solver="lbfgs", multi_class="auto")

classifiers.append(xgb_class)
classifiers.append(svm_class)
classifiers.append(tree_class)
classifiers.append(forest_class)
classifiers.append(naive_class)
classifiers.append(kmeans_class)
classifiers.append(regression_class)

accuracy_list = []
cm_list = []

for clf in classifiers:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    accuracy_list.append([acc])
    cm_list.append([cm])

most_accurate_class = classifiers[accuracy_list.index(max(accuracy_list))]
accuarcy_percent = max(accuracy_list)
print("The most accurate classifier is the", most_accurate_class, " with and accuracy of", accuarcy_percent)
