#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:54:58 2024

@author: nitaishah
"""

#%% Q1 USE KNN and Optimize KNN using Grid Search CV

from sklearn.datasets import fetch_openml 


mnist = fetch_openml('mnist_784', version=1, as_frame=False)

mnist.keys()

X, y = mnist["data"], mnist["target"]

X.shape

y.shape

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


y = y.astype(np.uint8)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

knn_class = KNeighborsClassifier()
knn_class.fit(X_train, y_train)

y_pred = knn_class.predict(X_test)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

conf_mx = confusion_matrix(y_test, y_pred)
plt.matshow(conf_mx, cmap=plt.cm.gray)
print(metrics.accuracy_score(y_test, y_pred)) ## returns 0.9688


from sklearn.model_selection import GridSearchCV

knn_grid = KNeighborsClassifier()

param_grid = {
    'n_neighbors': [3, 5, 7, 9],
    'weights': ['uniform', 'distance']
}

grid_search = GridSearchCV(knn_grid, param_grid, cv=5, scoring='accuracy')

grid_search.fit(X_train, y_train)

print("Best Hyperparameters:", grid_search.best_params_)

print("Best Cross-Validated Score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy) ##returns 0.9717


#%% Q2 Augmented Dataset and KNN

import scipy
from scipy.ndimage import shift

def convert_arr_to_img(arr):
    img = arr.reshape(28,28)
    return img

img = convert_arr_to_img(X_train[0])
plt.imshow(img)

def shift_img(img):
    img_right = shift(img, [0,1])
    img_left = shift(img, [0,-1])
    img_up = shift(img, [-1,0])
    img_down = shift(img, [1,0])
    img_arr = np.array([img.reshape(784,), img_right.reshape(784,), img_left.reshape(784,), img_up.reshape(784,), img_down.reshape(784,)])
    return img_arr

img_arr = shift_img(img)
img_arr[0].shape 



def augment_dataset(X_train,y_train):
    augmented_X_train = []
    augmented_y_train = []
    for i in range(0, len(X_train)):
        img = convert_arr_to_img(X_train[i])
        img_arr = shift_img(img)
        for j in range(0, len(img_arr)):
            augmented_X_train.append(img_arr[j])
            augmented_y_train.append(y_train[i])
        
    return augmented_X_train, augmented_y_train

augmented_X_train ,augmented_y_train = augment_dataset(X_train, y_train)
len(augmented_X_train)
len(augmented_y_train)

#confirm if the augmentation function works as intended
plt.imshow(augmented_X_train[450].reshape(28,28))
augmented_y_train[450]

knn_augment = KNeighborsClassifier()
knn_augment.fit(augmented_X_train, augmented_y_train)

y_pred_augment = knn_augment.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred_augment)) ##returns 0.9754



