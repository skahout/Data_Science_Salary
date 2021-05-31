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


from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)

cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=4) #mean abs error shows how far on avg we are off on our general prediction.
np.mean(cross_val_score(lm, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=4))
#Around 19.7k Dollars off
#Matrix is sparse and limited data so for MLR it's difficult to get good values from this 

type(X_train)

# Decision Trees Regression
from sklearn.tree import DecisionTreeRegressor
lmt = DecisionTreeRegressor()
lmt.fit(X_train,y_train)

np.mean(cross_val_score(lmt,X_train,y_train, scoring = 'neg_mean_absolute_error', cv = 4))
# $15.5k $ off


# lasso regression
from sklearn.linear_model import Lasso
#This would normalize those values so should be better for our models
lm_l = Lasso(alpha=.03)
lm_l.fit(X_train,y_train)
np.mean(cross_val_score(lm_l, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=4))
# Gives 19.2k Dollars off so a little bit worse.
#Increasing Alpha values on lasso regression

#Below code is to generate the best alpha value for Alpha
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

# Support Vector Regression
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
regr.fit(X_train,y_train)
 
np.mean(cross_val_score(regr, X_train, y_train, scoring = 'neg_mean_absolute_error', cv=4))
# Gives 28k Dollars 

# random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()

np.mean(cross_val_score(rf,X_train,y_train, scoring = 'neg_mean_absolute_error', cv = 4))
# error of 14.4k dollars. Much better

from sklearn.model_selection import GridSearchCV
# tune models using GridsearchCV
parameters_rf = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')} #parameters for Random Forest
parameters_lmt = {'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')} #parameters for Decision Tree
gs_rf = GridSearchCV(rf, parameters_rf, scoring = 'neg_mean_absolute_error', cv = 4)
gs_lmt = GridSearchCV(lmt, parameters_lmt, scoring = 'neg_mean_absolute_error', cv = 4)


#Takes a lot of time
gs_rf.fit(X_train, y_train)
gs_lmt.fit(X_train, y_train)
 
gs_rf.best_score_
gs_rf.best_estimator_

gs_lmt.best_score_
gs_lmt.best_estimator_

# test ensembles
tpred_lm = lm.predict(X_test) #linear Regression
tpred_lmt = lmt.predict(X_test) # Decision Tree
tpred_lmt1 = gs_lmt.best_estimator_.predict(X_test) #Decision Tree Tuned
tpred_lml = lm_l.predict(X_test) # lasso Regression
tpred_SVR = regr.predict(X_test) # SVR Regression
tpred_rf = gs.best_estimator_.predict(X_test) # Random Forest Tuned

from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test,tpred_lm) #18.8k
mean_absolute_error(y_test,tpred_lmt) #9.9k
mean_absolute_error(y_test, tpred_lmt1) #11.2k
mean_absolute_error(y_test,tpred_lml) #19k
mean_absolute_error(y_test,tpred_rf) #12k
mean_absolute_error(y_test,tpred_SVR) #29k

# Ensemble of Decision Tree and its opimization gives the best result although unsure if this is the correct approach
mean_absolute_error(y_test,(tpred_lmt+tpred_lmt1)/2) #8.8k

# Building Artificial Neural Network
import tensorflow as tf
# train test split
from sklearn.model_selection import train_test_split
X = df_dum.drop('avg_salary', axis = 1)
y = df_dum.avg_salary.values #.values creates array instead of a series

train = X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#buidling ann
ann = tf.keras.models.Sequential()

#adding first hidden layer
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))

#adding second hidden layer
ann.add(tf.keras.layers.Dense(units=10, activation='relu'))

#adding output layer
ann.add(tf.keras.layers.Dense(units=1))

ann.compile(optimizer = 'adam', loss = 'mean_absolute_error')

ann.fit(X_train, y_train, batch_size = 32, epochs = 1500)
# Less than $1.5k off of the actual avg_salary which is substantially better than the values produced by previous models

y_pred = ann.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

np.mean(y_pred)
np.mean(y_test)
