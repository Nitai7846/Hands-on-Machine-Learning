#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:24:04 2024

@author: nitaishah
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train_set = pd.read_csv("/Users/nitaishah/Downloads/titanic/train.csv")
train_set.head()

test_set =  pd.read_csv("/Users/nitaishah/Downloads/titanic/test.csv")
test_set.head()

train_set.describe()
train_set.info()

##First seprate label and features

X_train = train_set.drop('Survived', axis=1)
y_train = train_set['Survived']

##The Cabin Feature has too many null values, hence it should be dropped
##PassengerID, Name, Ticket is not important as well

X_train = X_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
X_train.info()

##Let us first deal with Numerical Values. We see that the Age column has some missing values. We can fix this via simple imputer

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')

X_train_num = X_train.drop(['Sex', 'Embarked'], axis=1)

imputer.fit(X_train_num)

X = imputer.transform(X_train_num)

X_train_num_transformed = pd.DataFrame(X, columns=X_train_num.columns, index=X_train_num.index)

X_train_num_transformed.info() ## Check to see if all values are handled

## Now we will deal with the text values
## The Embarked Feature has 2 values that are missing, we need to correct those first.                    

X_train_cat = X_train[['Sex', 'Embarked']]

def encode_missing_catgeorical(dataframe, column):
    most_frequent = dataframe[column].mode()[0]
    dataframe[column].fillna(most_frequent, inplace=True)
    return dataframe

X_train_cat = encode_missing_catgeorical(X_train_cat, 'Embarked')
X_train_cat.info()

from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder() 
X_train_cat_1hot = cat_encoder.fit_transform(X_train_cat)
X_train_cat_1hot

##Expedite the entire process using pipelines. First we shall develop a pipeline for numerical attributes.
##Numerical Attribute pipeline shall have Imputation and Scaling

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), 
                         ('scaler', StandardScaler()),])

X_train_num_final = num_pipeline.fit(X_train_num)

##Our encode_missing_categorical function needs to be converted to fit into a pipeline


from sklearn.base import BaseEstimator, TransformerMixin

class MostFrequentImputer(BaseEstimator, TransformerMixin):
    
    def __init__(self,columns=None):
        self.columns = columns
       
    def fit(self,dataframe,y=None):
        self.most_frequent_values = dataframe[self.columns].mode().iloc[0]
        return self
    
    def transform(self, dataframe):
        X_filled = dataframe.copy()
        X_filled[self.columns] = X_filled[self.columns].fillna(self.most_frequent_values)
        return X_filled
      
    
column_to_impute = 'Embarked'

# Creating a pipeline with MostFrequentImputer
cat_pipeline = Pipeline([
    ('most_frequent_imputer', MostFrequentImputer(columns=[column_to_impute])),
    ('one_hot_encoder', OneHotEncoder())
])

# Fitting and transforming the data


X_train_cat_transformed = cat_pipeline.fit_transform(X_train_cat)

from sklearn.compose import ColumnTransformer

num_attribs = list(X_train_num)
cat_attribs = list(X_train_cat)

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs)])

X_train_prepared = full_pipeline.fit_transform(X_train)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, cross_val_score

forest_clf = RandomForestClassifier()
forest_clf.fit(X_train_prepared, y_train)
y_train_pred = forest_clf.predict(X_train_prepared)

from sklearn.metrics import accuracy_score
accuracy_score(y_train, y_train_pred)

X_test = test_set
X_test = X_test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

X_test_prepared = full_pipeline.transform(X_test)
y_test = forest_clf.predict(X_test_prepared)
y_test = pd.DataFrame(y_test)
y_test.columns = ['Survived']

submission_df = pd.concat([test_set['PassengerId'], y_test['Survived']], axis=1)
submission_df.to_csv('/Users/nitaishah/Desktop/Hands-on-ML/Chapter3 - Classification/titanicsubmission.csv', index=False)
