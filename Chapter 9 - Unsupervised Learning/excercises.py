#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 15:37:51 2024

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

def display(num):
    first_image = data.images[num]
    plt.imshow(first_image, cmap='gray')  
    plt.axis('off')
    plt.show()

X = flattened_images
y = target

X_temp, X_holdout, y_temp, y_holdout = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)


print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_holdout.shape, y_holdout.shape)

from sklearn.cluster import KMeans

k_range = range(5, 150, 5)
kmeans_per_k = []
for k in k_range:
    print("k={}".format(k))
    kmeans = KMeans(n_clusters=k, random_state=42).fit(X_train)
    kmeans_per_k.append(kmeans)

from sklearn.metrics import silhouette_score

silhouette_scores = [silhouette_score(X_train, model.labels_)
                     for model in kmeans_per_k]
best_index = np.argmax(silhouette_scores)
best_k = k_range[best_index]
best_score = silhouette_scores[best_index]

plt.figure(figsize=(8, 3))
plt.plot(k_range, silhouette_scores, "bo-")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Silhouette score", fontsize=14)
plt.plot(best_k, best_score, "rs")
plt.show()
   
best_k 

best_kmeans = KMeans(n_clusters=110, random_state=42).fit(X_train)
len(best_kmeans.labels_)

best_kmeans.labels_

def plot_faces(data, labels):
    images_by_cluster = {}

    for image, cluster_label in zip(data, labels):

        if cluster_label in images_by_cluster:
           
            images_by_cluster[cluster_label].append(image)
        else:
            
            images_by_cluster[cluster_label] = [image]
            
    return images_by_cluster

images_by_cluster = plot_faces(X_train, best_kmeans.labels_)
plt.imshow(images_by_cluster[1][1].reshape(64,64), cmap='gray')

images_by_cluster.items()
    
for cluster_num, cluster_images in images_by_cluster.items():
    # Create a figure object for each cluster
    plt.figure(figsize=(15, 3))
    
    # Iterate over each image in the cluster and plot it
    for i, image in enumerate(cluster_images):
        # Add subplot for each image
        plt.subplot(1, len(cluster_images), i + 1)
        
        # Plot the image
        plt.imshow(image.reshape(64, 64), cmap='gray')
        plt.title(f'Cluster {cluster_num}')  # Set subplot title with cluster number
        plt.axis('off')  # Turn off axis labels
    
    # Adjust layout for the cluster's plots
    plt.tight_layout()
    plt.show()

#Q2

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)    
y_pred = log_reg.predict(X_val)
accuracy_score(y_val, y_pred)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('kmeans', KMeans()),
    ('log_reg', LogisticRegression())
])

param_grid = {
    'kmeans__n_clusters': range(5, 150)  # Range of cluster values
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_search.fit(X_train, y_train)

print("Best parameters found:", grid_search.best_params_)
y_pred_KNN = grid_search.predict(X_val)
accuracy_score(y_val, y_pred_KNN)

X_train_reduced = best_kmeans.transform(X_train)
X_val_reduced  = best_kmeans.transform(X_val)
X_test_reduced = best_kmeans.transform(X_holdout)

X_train_extended = np.c_[X_train, X_train_reduced]
X_valid_extended = np.c_[X_val, X_val_reduced]
X_test_extended = np.c_[X_holdout, X_test_reduced]


grid_search_new = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_search_new.fit(X_train_extended, y_train)
y_pred_KNN_extended = grid_search_new.predict(X_valid_extended)
accuracy_score(y_val, y_pred_KNN_extended)

