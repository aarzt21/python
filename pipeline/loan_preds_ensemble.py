# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 12:50:42 2021

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
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import FunctionTransformer



#import data
data = pd.read_csv("loan_train.csv")
data = data.drop("Loan_ID", axis=1)
data["Loan_Status"].replace(('Y', 'N'), (1, 0), inplace=True)

test_data = pd.read_csv("loan_test.csv")
test_data = test_data.drop("Loan_ID", axis=1)


#----------------------------- EDA -----------------------

#check out data types
data.isnull().sum()

data.Gender = data.Gender.astype("category")
data.Married = data.Married.astype("category")
data.Dependents = data.Dependents.astype("category")
data.Education = data.Education.astype("category")
data.Self_Employed = data.Self_Employed.astype("category")
data.Credit_History = data.Credit_History.astype("category")
data.Property_Area = data.Property_Area.astype("category")
data.Loan_Status = data.Loan_Status.astype("category")

data.dtypes # -> we have numerical data (need scaling) and categorical data (need one hot encoding)


test_data.Gender = test_data.Gender.astype("category")
test_data.Married = test_data.Married.astype("category")
test_data.Dependents = test_data.Dependents.astype("category")
test_data.Education = test_data.Education.astype("category")
test_data.Self_Employed = test_data.Self_Employed.astype("category")
test_data.Credit_History = test_data.Credit_History.astype("category")
test_data.Property_Area = test_data.Property_Area.astype("category")

X_test = test_data




#-------------------------------- build pipeline ---------------------------
#numeric features: 1. imput 2. scale; categorical features: 1. imput 2. ohe

def log1(x):
    return np.log(x+1)

LT = FunctionTransformer(log1)

numeric_transformer = Pipeline(steps=[
   ('imputer', KNNImputer(n_neighbors=5)),
   ('LT', LT),
    ('scaler', StandardScaler())])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])



# use column transformer to apply the preprocessing steps to the right features
num_features = data.select_dtypes(include=['int64', 'float64']).columns
cat_features = data.select_dtypes(include=['category']).drop('Loan_Status', axis=1).columns


preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)])


# pipeline that combines preprocessor with classifier
rf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', RandomForestClassifier())])

lr = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier', LogisticRegression())])

gb = Pipeline(steps=[('preprocessor', preprocessor),
                      ('classifier',GradientBoostingClassifier())])



#-------------------- model selection on train set only ---------------------------------------
#split data
X = data.loc[:, data.columns != "Loan_Status"]
y = data["Loan_Status"]


#grid search CV to find optimal hyperparameters
rf_grid = { 
    'classifier__n_estimators': [200, 500],
    'classifier__max_features': ['auto', 'sqrt', 'log2'],
    'classifier__max_depth' : [4,5,6,7,8],
    'classifier__criterion' :['gini', 'entropy']}

rf = GridSearchCV(rf, rf_grid, n_jobs= 6, verbose=1, refit = True, scoring = "accuracy")
rf.fit(X, y)

gb_grid = {
    "classifier__learning_rate" : [0.05, 0.1, 0.2], 
    'classifier__n_estimators': [200, 500],
    "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
    "classifier__max_depth" : [2,3,4]
    }

gb = GridSearchCV(gb, gb_grid, n_jobs= 6, verbose=1, refit = True, scoring = "accuracy")
gb.fit(X, y)




#----------------------------- model evaluation -----------------------
preds_rf = rf.predict(X_test)
preds_gb = gb.predict(X_test)
preds_lr = lr.fit(X, y).predict(X_test)


#majority vote
preds_ens = np.stack([preds_rf, preds_gb, preds_lr], axis=1) 
preds_ens = pd.DataFrame(data=preds_ens, columns=["rf", "gb","lr"])
preds_ens['majority'] = preds_ens.mode(axis=1)[0]

test_data = pd.read_csv("loan_test.csv")
Loan_ID = test_data.Loan_ID

res = pd.DataFrame(data = {"Loan_ID": Loan_ID,
                           "Loan_Status": preds_ens.majority})

res.Loan_Status.replace((1, 0), ('Y', 'N'), inplace=True)

res.to_csv("res2.csv",index=False, header=True)

