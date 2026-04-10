# Author: Suraj Dev Kant

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

data = load_iris()
X = data.data

pipe = Pipeline([
    ('pca', PCA(n_components=2)),
    ('kmeans', KMeans(n_clusters=3))
])

labels = pipe.fit_predict(X)
print("Cluster Labels:", labels[:10])