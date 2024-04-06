#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 02:08:38 2024

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
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()
X, y = mnist["data"], mnist["target"]
len(X)


X_train, y_train = X[:50000], y[:50000]
X_val, y_val = X[50000:60000], y[50000:60000]
X_test, y_test = X[60000:], y[60000:]

rnd_clf = RandomForestClassifier()
rnd_clf.fit(X_train, y_train)
y_pred_val_rf = rnd_clf.predict(X_val)
y_pred_val_rf

svm_clf = LinearSVC()
svm_clf.fit(X_train, y_train)
y_pred_val_svm = svm_clf.predict(X_val)
y_pred_val_svm

ext_clf = ExtraTreesClassifier()
ext_clf.fit(X_train, y_train)
y_pred_val_ext = ext_clf.predict(X_val)
y_pred_val_ext

y_pred_val_rf = y_pred_val_rf.astype(float)
y_pred_val_svm = y_pred_val_svm.astype(float)
y_pred_val_ext = y_pred_val_ext.astype(float)
combined_array = np.column_stack((y_pred_val_rf, y_pred_val_svm, y_pred_val_ext))

y_val = y_val.astype(int)

blender_model = XGBClassifier()
blender_model.fit(combined_array, y_val)

y_test_rf = rnd_clf.predict(X_test)
y_test_svm = svm_clf.predict(X_test)
y_test_ext = ext_clf.predict(X_test)

y_test_rf = y_test_rf.astype(float)
y_test_svm = y_test_svm.astype(float)
y_test_ext = y_test_ext.astype(float)
combined_array_new = np.column_stack((y_test_rf, y_test_svm, y_test_ext))
combined_array_new

final_pred = blender_model.predict(combined_array_new)

y_test = y_test.astype(int)

accuracy_score(y_test,final_pred) ## 96.96 Blender Model
accuracy_score(y_test, y_test_rf) ## 96.84 Random Forest Model
accuracy_score(y_test, y_test_svm) ## 83.73 SVM Model
accuracy_score(y_test, y_test_ext) ## 97.12 Extra Trees Model


