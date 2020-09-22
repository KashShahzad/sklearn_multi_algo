from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

# split features and labels
X= iris.data
y = iris.target

classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

model = svm.SVC()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)

print('predictions: ', predictions)
print('accuracy: ', acc)