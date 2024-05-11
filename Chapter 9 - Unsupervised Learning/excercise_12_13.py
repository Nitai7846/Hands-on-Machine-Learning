#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 11 19:11:02 2024

@author: nitaishah
"""

import numpy as np
import pandas as pd
import sklearn
import sklearn.datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

data = sklearn.datasets.fetch_olivetti_faces()
data

images = data.images
target = data.target
description = data.DESCR
flattened_images = data.data

X = flattened_images
y = target

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)

from sklearn.decomposition import PCA

pca = PCA(0.99)
X_train_pca = pca.fit_transform(X_train)
X_valid_pca = pca.transform(X_valid)
X_test_pca = pca.transform(X_test)

from sklearn.mixture import GaussianMixture

num_components = 3  
gmm = GaussianMixture(n_components=num_components, random_state=42)

gmm.fit(X_train_pca)

y_pred = gmm.fit_predict(X_train_pca)

n_gen_faces = 20
gen_faces_reduced, y_gen_faces = gmm.sample(n_samples=n_gen_faces)
gen_faces = pca.inverse_transform(gen_faces_reduced)

def plot_faces(data):
    for i in range(len(data)):
        new = data[i].reshape(64,64)
        plt.imshow(new, cmap='gray')
        plt.axis('off')
        plt.show()
        
        
plot_faces(gen_faces)

n_rotated = 4
rotated = np.transpose(X_train[:n_rotated].reshape(-1, 64, 64), axes=[0, 2, 1])
rotated = rotated.reshape(-1, 64*64)
y_rotated = y_train[:n_rotated]

n_flipped = 3
flipped = X_train[:n_flipped].reshape(-1, 64, 64)[:, ::-1]
flipped = flipped.reshape(-1, 64*64)
y_flipped = y_train[:n_flipped]

n_darkened = 3
darkened = X_train[:n_darkened].copy()
darkened[:, 1:-1] *= 0.3
y_darkened = y_train[:n_darkened]

X_bad_faces = np.r_[rotated, flipped, darkened]
y_bad = np.concatenate([y_rotated, y_flipped, y_darkened])

plot_faces(X_bad_faces)

X_bad_faces_pca = pca.transform(X_bad_faces)
gmm.score_samples(X_bad_faces_pca)

gmm.score_samples(X_train_pca[:10])

def reconstruction_errors(pca, X):
    X_pca = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_pca)
    mse = np.square(X_reconstructed - X).mean(axis=-1)
    return mse

reconstruction_errors(pca, X_train).mean()

reconstruction_errors(pca, X_bad_faces).mean()

X_bad_faces_reconstructed = pca.inverse_transform(X_bad_faces_pca)
plot_faces(X_bad_faces_reconstructed)



