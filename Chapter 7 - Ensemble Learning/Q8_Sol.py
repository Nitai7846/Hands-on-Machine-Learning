#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 00:12:04 2024

@author: nitaishah
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml 
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()
X, y = mnist["data"], mnist["target"]
len(X)


X_train, y_train = X[:50000], y[:50000]
X_val, y_val = X[50000:60000], y[50000:60000]
X_test, y_test = X[60000:], y[60000:]

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)
plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")

rnd_clf = RandomForestClassifier()
rnd_clf.fit(X_train, y_train)
y_pred_val = rnd_clf.predict(X_val)
accuracy_score(y_val, y_pred_val)

svm_clf = LinearSVC()
svm_clf.fit(X_train, y_train)
y_pred_val_svm = svm_clf.predict(X_val)
accuracy_score(y_val, y_pred_val_svm)

ext_clf = ExtraTreesClassifier()
ext_clf.fit(X_train, y_train)
y_pred_val_ext = ext_clf.predict(X_val)
accuracy_score(y_val, y_pred_val_ext)

from sklearn.ensemble import VotingClassifier

named_estimators = [
    ("random_forest_clf", rnd_clf),
    ("extra_trees_clf", ext_clf),
    ("svm_clf", svm_clf)
]

voting_clf = VotingClassifier(named_estimators)
voting_clf.fit(X_train, y_train)
voting_clf.score(X_val, y_val)

y_pred_test_rf = rnd_clf.predict(X_test)
accuracy_score(y_test, y_pred_test_rf)

y_pred_test_svm = svm_clf.predict(X_test)
accuracy_score(y_test, y_pred_test_svm)

y_pred_test_ext = ext_clf.predict(X_test)
accuracy_score(y_test, y_pred_test_ext)

y_pred_test_vot = voting_clf.predict(X_test)
accuracy_score(y_test, y_pred_test_vot)

voting_clf.set_params(svm_clf=None)
del voting_clf.estimators_[2]
voting_clf.score(X_val, y_val)

y_pred_test_vot = voting_clf.predict(X_test)
accuracy_score(y_test, y_pred_test_vot)



