# Author: Suraj Dev Kant

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500)

model = RandomForestClassifier(n_estimators=100, oob_score=True)
model.fit(X, y)

print("OOB Score:", model.oob_score_)
print("Feature Importance:", model.feature_importances_)