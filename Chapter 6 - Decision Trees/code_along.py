#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 19:39:02 2024

@author: nitaishah
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X,y)

tree_clf.predict_proba([[5,1.5]])
tree_clf.predict([[5,1.5]])

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X,y)

from sklearn.tree import export_graphviz


export_graphviz(
        tree_reg,
        out_file="/Users/nitaishah/Desktop/Hands-on-ML/Chapter 6 - Decision Trees/ reg_tree.dot",
        feature_names=["x1"],
        rounded=True,
        filled=True
    )

