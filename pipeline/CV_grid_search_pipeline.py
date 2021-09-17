# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 18:56:43 2021

@author: alex arzt

This is a practice script of mine that 
demonstrates that you can cross validate and grid search an
entire pipeline. Concretely, each data preprocessing is done
on the training data during EACH iteration of the CV. This is 
the right way to do it - instead of preprocessing the entire
data set and then doing the CV. 

"""

import pandas as pd
df = pd.read_csv('http://bit.ly/kaggletrain')
X = df[["Sex", "Name"]]
y = df["Survived"]



# import and prepare all the necessary stuff for the preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.compose import make_column_transformer

ohe = OneHotEncoder()
vect = CountVectorizer()
ct = make_column_transformer((ohe, ['Sex']), (vect, 'Name'))

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='liblinear', random_state=1)


#create pipeline object; steps in pipeline: 1. one-hot-encode "sex" and vectorize "name" 2. fit model

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(ct, clf) 

#now: cross-validate the entire pipeline - NOT JUST THE MODEL!
from sklearn.model_selection import cross_val_score
print(cross_val_score(pipe, X, y, cv=5, scoring='accuracy').mean())

# now lets do a grid search CV to find optimal hyperparameters
params = {}
params['columntransformer__countvectorizer__min_df'] = [1, 2]
params['logisticregression__C'] = [0.1, 1, 10]
params['logisticregression__penalty'] = ['l1', 'l2']

# try all possible combinations of those parameter values
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(pipe, params, cv=5, scoring='accuracy')
grid.fit(X, y)

# what was the best score found during the search?
print(grid.best_score_)

# which combination of parameters produced the best score?
print(grid.best_params_)






