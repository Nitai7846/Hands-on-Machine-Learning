#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 14:41:46 2024

@author: nitaishah
"""

import numpy as np
import pandas as pd


housing = pd.read_csv("/Users/nitaishah/Desktop/Hands-on-ML/datasets/housing/housing.csv")

housing["income_cat"] = pd.cut(housing["median_income"], bins=[0, 1.5, 3, 4.5,6,np.inf], labels=[1,2,3,4,5])
housing["income_cat"].hist()

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
    
    
strat_test_set["income_cat"].value_counts()/len(strat_test_set)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
    

housing = strat_train_set.copy()

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

from sklearn.impute import SimpleImputer 

imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)
imputer.fit(housing_num)

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)

housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]


from sklearn.preprocessing import OneHotEncoder
cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot

housing_cat_1hot.toarray()
cat_encoder.categories_

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        
    def fit(self, X, y=None):
        return self 
    
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix]/ X[:, households_ix]
        population_per_household = X[: , population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix]/X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
    

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),])

housing_num_tr = num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer 

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs), 
    ("cat", OneHotEncoder(), cat_attribs),])


housing_prepared = full_pipeline.fit_transform(housing)


#%% Q1 

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

param_grid = {
    'kernel': ['linear', 'rbf'],  # You can add other kernels as well
    'C': [1, 10, 100]       # Regularization parameter
       # Epsilon in the epsilon-SVR model
}

svr = SVR()



grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_

cvres = grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
#%% Q2 

from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(svr, param_distributions=param_grid, n_iter=6, scoring='neg_mean_squared_error', cv=5)

random_search.fit(housing_prepared, housing_labels)

random_search.best_params_

cvres_random = random_search.cv_results_

for mean_score, params in zip(cvres_random["mean_test_score"], cvres_random["params"]):
    print(np.sqrt(-mean_score), params)


#%% Q3

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from sklearn.base import BaseEstimator, TransformerMixin

forest_reg = RandomForestRegressor()

forest_reg.fit(housing_prepared, housing_labels)

forest_predictions = forest_reg.predict(housing_prepared)

forest_mse = mean_squared_error(housing_labels, forest_predictions)

forest_rmse = np.sqrt(forest_mse)

forest_rmse

feature_importances = forest_reg.feature_importances_

feature_importances

attributes = housing.columns

sorted(zip(feature_importances, attributes), reverse=True)

def return_top_indices(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    
    def fit(self, X, y=None):
        self.feature_indices_ = return_top_indices(self.feature_importances, self.k)
        return self
    
    def transform(self, X):
        return X[:, self.feature_indices_]
        
k = 5

top_k_feature_indices = return_top_indices(feature_importances, k)
top_k_feature_indices

preparation_and_feature_selection_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k))
])

housing_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(housing)

housing_prepared_top_k_features[0:3]

housing_prepared[0:3, top_k_feature_indices]

#%% Q4 

prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k)),
    ('random_forest', forest_reg)
])

prepare_select_and_predict_pipeline.fit(housing, housing_labels)
