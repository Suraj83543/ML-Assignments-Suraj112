# Author: Suraj Dev Kant

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target)

model = SVC(kernel='rbf')
model.fit(X_train, y_train)

print("Accuracy:", model.score(X_test, y_test))