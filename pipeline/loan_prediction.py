# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 09:08:03 2021

@author: alex arzt
"""

#libraries 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics



#import data
data = pd.read_csv("loan_train.csv")
data = data.drop("Loan_ID", axis=1)
data["Loan_Status"].replace(('Y', 'N'), (1, 0), inplace=True)

#check out data types
data.dtypes # -> we have numerical data (need scaling) and categorical data (need one hot encoding)



#-------------------------------- build pipeline ---------------------------
#numeric features: 1. imput 2. scale; categorical features: 1. imput 2. ohe

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])


# use column transformer to apply the preprocessing steps to the right features
numeric_features = data.select_dtypes(include=['int64', 'float64']).drop(['Loan_Status'], axis=1).columns
categorical_features = data.select_dtypes(include=['object']).columns


preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# pipeline that combines preprocessor with classifier
rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])


#-------------------- model selection on train set only ---------------------------------------
#split data
X = data.loc[:, data.columns != "Loan_Status"]
y = data["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

#grid search CV to find optimal hyperparameters
rf_grid = { 
    'classifier__n_estimators': [200, 500],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
    'classifier__max_depth' : [4,5,6,7,8],
    'classifier__criterion' :['gini', 'entropy']}

rf = GridSearchCV(rf, rf_grid, n_jobs= 4, verbose=0, refit = True, scoring = "accuracy")
rf.fit(X_train, y_train)
print(rf.best_score_)
               
#----------------------------- model evaluation -----------------------
preds = rf.predict(X_test)
print(metrics.accuracy_score(y_test, preds))








