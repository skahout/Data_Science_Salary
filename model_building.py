# -*- coding: utf-8 -*-
"""
Created on Mon May  3 20:25:00 2021

@author: Sheryar Kahout
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('eda_data.csv')

# choose relevant columns
df.columns

df_model = df[['avg_salary', 'Rating', 'Size', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'comp_count', 'hourly', 'employer_provided', 'job_state', 'same_state', 'age','python_yn', 'spark_yn', 'aws_yn', 'excel_yn', 'job_simp', 'seniority', 'des_len']]


# get dummy variables
df_dum = pd.get_dummies(df_model)

# train test split
from sklearn.model_selection import train_test_split
X = df_dum.drop('avg_salary', axis = 1)
y = df_dum.avg_salary.values #.values creates array instead of a series

train = X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# mulitple linear regression
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score, GridSearchCV

lm = LinearRegression()
lm.fit(X_train,y_train)

cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=4) #mean abs error shows how far on avg we are off on our general prediction.
np.mean(cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=4))

#Around 19.7k Dollars off
#Matrix is sparse and limited data so for MLR it's difficult to get good values from this 

# lasso regression
#This would normalize those values so should be better for our models
lm_l = Lasso(alpha=.03)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=4))
# Gives 21.2k Dollars off so a little bit worse.
#Increasing Alpha values on lasso regression

alpha = []
error = []

for i in range(1,100):
    alpha.append(i/100)
    lml = Lasso(alpha=(i/100))
    error.append(np.mean(cross_val_score(lml, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=4)))

plt.plot(alpha,error)

err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err, columns = ['aplha','error'])
df_err[df_err.error == max(df_err.error)]
# It seems the best error is given when alpha = .03


# random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf,X_train,y_train, scoring = 'neg_mean_absolute_error', cv = 4))
# error of 14.4k dollars. Much better


# tune models using GridsearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}
gs = GridSearchCV(rf, parameters, scoring = 'neg_mean_absolute_error', cv = 4)

#Takes a lot of time
#gs.fit(X_train, y_train)

gs.best_score_
gs.best_estimator_

# test ensembles
tpred_lm = lm.predict(X_test)
tpred_lml = lm_l.predict(X_test)
tpred_rf = gs.best_estimator_.predict(X_test)

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm) #18.85310800201176
mean_absolute_error(y_test,tpred_lml) #19.032844153948414
mean_absolute_error(y_test,tpred_rf) #12.129961649089166

mean_absolute_error(y_test,(tpred_lm+tpred_rf)/2) #14.326232442246416


















