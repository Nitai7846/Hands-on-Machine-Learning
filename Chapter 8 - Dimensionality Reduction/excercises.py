#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 19:30:45 2024

@author: nitaishah
"""

import numpy as np
import pandas as pd 
import sklearn 
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml 
import time


mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()
X, y = mnist["data"], mnist["target"]
X.shape
y.shape
y_int = np.array(y, dtype=int)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

rfc = RandomForestClassifier()
start_time = time.time()
rfc.fit(X_train, y_train)
end_time = time.time()

training_time = end_time - start_time
training_time

y_pred = rfc.predict(X_test)
accuracy_score(y_test, y_pred)

from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)
X_reduced.shape

rfc_pca = RandomForestClassifier()
start_time_pca = time.time()
rfc_pca.fit(X_reduced, y_train)
end_time_pca = time.time()

training_time_pca = end_time_pca - start_time_pca
training_time_pca

X_test_pca = pca.transform(X_test)
y_pred_pca = rfc_pca.predict(X_test_pca)
accuracy_score(y_test, y_pred_pca)

from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
start_time_log = time.time()
log_clf.fit(X_train, y_train)
end_time_log = time.time()

training_time_log = end_time_log - start_time_log
training_time_log

y_pred_log = log_clf.predict(X_test)
accuracy_score(y_test, y_pred_log)

log_pca = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
start_time_log_pca = time.time()
log_pca.fit(X_reduced, y_train)
end_time_log_pca = time.time()

training_time_log_pca = end_time_log_pca - start_time_log_pca
training_time_log_pca 

y_pred_log_pca = log_pca.predict(X_test_pca)
accuracy_score(y_test, y_pred_log_pca)

from sklearn.manifold import TSNE

tsne_model = TSNE(n_components=2)
X_tsne = tsne_model.fit_transform(X)
X_tsne

X_tsne[:, 1]

def vanilla_plot(X,y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="jet", s=100, alpha=0.75)
    plt.title('Scatter Plot with Classifications')
    plt.axis('off')
    plt.colorbar()
    plt.show()

vanilla_plot(X_tsne, y_int)

from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

def plot_digits(X, y, min_distance=0.05, images=None, figsize=(13,10)):
    X_normalized = MinMaxScaler().fit_transform(X)
    neighbors = np.array([[10.,10.]])
    plt.figure(figsize=figsize)
    cmap = plt.cm.get_cmap("jet")
    digits = np.unique(y)
    for digit in digits:
        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=[cmap(digit / 9)])
    plt.axis("off")
    ax = plt.gcf().gca()
    for index, image_coord in enumerate(X_normalized):
        closest_distance = np.linalg.norm(neighbors-image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(int(y[index])),
                         color=cmap(y[index]/9), fontdict={"weight": "bold", "size": 16})
            else:
                image = images[index].reshape(28,28)
                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"), image_coord)
                ax.add_artist(imagebox)
 
plot_digits(X_tsne, y_int)        

plot_digits(X_tsne, y_int, images=X, figsize=(35, 25))

from sklearn.decomposition import PCA
X_pca = PCA(n_components=2).fit_transform(X)
vanilla_plot(X_pca, y_int)
plot_digits(X_pca, y_int, images=X, figsize=(35,35))

from sklearn.manifold import LocallyLinearEmbedding
X_lle = LocallyLinearEmbedding(n_components=2).fit_transform(X)
vanilla_plot(X_lle, y_int)
plot_digits(X_lle, y_int, images=X, figsize=(35,35))

from sklearn.pipeline import Pipeline

pca_lle = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("lle", LocallyLinearEmbedding(n_components=2, random_state=42)),
])
t0 = time.time()
X_pca_lle_reduced = pca_lle.fit_transform(X)
t1 = time.time()
print("PCA+LLE took {:.1f}s.".format(t1 - t0))
plot_digits(X_pca_lle_reduced, y_int)
plt.show()

from sklearn.manifold import MDS

m = 2000
t0 = time.time()
X_mds_reduced = MDS(n_components=2, random_state=42).fit_transform(X[:m])
t1 = time.time()
print("MDS took {:.1f}s (on just 2,000 MNIST images instead of 10,000).".format(t1 - t0))
plot_digits(X_mds_reduced, y_int[:m])
plt.show()


pca_mds = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("mds", MDS(n_components=2, random_state=42)),
])
t0 = time.time()
X_pca_mds_reduced = pca_mds.fit_transform(X[:2000])
t1 = time.time()
print("PCA+MDS took {:.1f}s (on 2,000 MNIST images).".format(t1 - t0))
plot_digits(X_pca_mds_reduced, y_int[:2000])
plt.show()


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

t0 = time.time()
X_lda_reduced = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
t1 = time.time()
print("LDA took {:.1f}s.".format(t1 - t0))
plot_digits(X_lda_reduced, y_int, figsize=(12,12))
plt.show()

pca_tsne = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("tsne", TSNE(n_components=2, random_state=42)),
])
t0 = time.time()
X_pca_tsne_reduced = pca_tsne.fit_transform(X)
t1 = time.time()
print("PCA+t-SNE took {:.1f}s.".format(t1 - t0))
plot_digits(X_pca_tsne_reduced, y_int)
plt.show()


