# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 14:57:11 2016

@author: rsoni106
"""

import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from keras.regularizers import l2, activity_l2
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score

""" Y goes as a horizontal array and not a single column """

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

final_data = pd.read_csv("C:/Users/rsoni106/Documents/Work/Methodology Work/Seizure Detection/DataPreparation/Final_files/final_data_patient_t.csv")

y = final_data[["event"]]

del final_data["event"]

final_data = StandardScaler().fit_transform(final_data)

X_train, X_test, y_train, y_test = train_test_split(final_data, y, test_size=0.25, random_state=1234)

rf_model = RandomForestClassifier(criterion = "entropy", max_depth = 2, min_samples_leaf = 1, n_jobs = -1, 
                                  n_estimators = 50, oob_score = False,random_state=1234)

param_grid = { 
    'n_estimators': [300,350,400],
    'max_depth': [8,10],
    'min_samples_leaf': [5,10],
    'max_features': ['log2']
}
                                  
cv_rf_model = GridSearchCV(estimator = rf_model, param_grid = param_grid, cv = 5, scoring = 'roc_auc', verbose = 3)

cv_rf_model.fit(full_data, np.ravel(y))

print (cv_rf_model.best_params_)

print(cv_rf_model.best_score_)

print(cv_rf_model.grid_scores_)

################### Best parameters for RF, need to tune around this only #################

rf_model_1 = RandomForestClassifier(criterion = "entropy", max_depth = 10, min_samples_leaf = 5, n_jobs = -1, 
                                  n_estimators = 300, oob_score = True, verbose = 2, max_features = 'log2',random_state=1234)
                                  
rf_model_1.fit(train_data, np.ravel(train_y))

y_rf = rf_model_1.predict_proba(test_data)

y_rf = y_rf[:,1]

roc_auc_score(test_y, y_rf)

log_loss(test_y,y_rf)

concordance(pd.DataFrame(y_rf),test_y)


####################### GBM #############

gbm_model =  GradientBoostingClassifier(learning_rate = 0.01, n_estimators = 50, max_depth = 2, min_samples_leaf = 1,
                                        subsample = 0.4,random_state=1234)
                                            
param_grid = { 
    'n_estimators': [100,125,150],
    'max_depth': [8],
    'min_samples_leaf': [5],
    'subsample': [0.75,0.8],
    'max_features': [0.7,0.5,'log2'],
    'learning_rate': [0.05,0.07]
}

cv_gbm_model = GridSearchCV(estimator = gbm_model, param_grid = param_grid, cv = 5, scoring = 'roc_auc', verbose = 10)

cv_gbm_model.fit(full_data, np.ravel(y))

print (cv_gbm_model.best_params_)

################### Best parameters for GBM, need to tune around this only #################

gbm_model_1 =  GradientBoostingClassifier(learning_rate = 0.07, n_estimators = 150, max_depth = 8, min_samples_leaf = 5,
                                        subsample = 0.75, max_features = 'log2', verbose = 3,random_state=1234)
                                  
gbm_model_1.fit(train_data, np.ravel(train_y))

y_gbm1 = gbm_model_1.predict_proba(test_data)

y_gbm1 = y_gbm1[:,1]

roc_auc_score(test_y, y_gbm1)

log_loss(test_y,y_gbm1)

concordance(pd.DataFrame(y_gbm1),test_y)

############################################################################

sgd_model = SGDClassifier(alpha = 0.1, n_iter = 50, loss = "log", penalty = 'l1', l1_ratio = 0.75, n_jobs = -1,
                          random_state=1234)

""" Use higher alpha for tuning the parameters """
                                            
param_grid = { 
    'n_iter': [400,450,500,550,600,300],
    'alpha': [0.0001,0.0002,0.0003],
    'penalty': ['l2'],
    'loss': ["log"],
    'power_t': [0.5,0.4,0.6]
}
"""1 mil/10K - Total number of minimum steps needed for an SGD classifier"""

cv_sgd_model = GridSearchCV(estimator = sgd_model, param_grid = param_grid, cv = 5, verbose = 2, scoring = 'roc_auc')

cv_sgd_model.fit(full_data, np.ravel(y))

print (cv_sgd_model.best_params_)

################### Best parameters for SGD, need to tune around this only #################

sgd_model_1 = SGDClassifier(alpha = 0.0002, n_iter = 500, loss = "log", penalty = 'l2', n_jobs = -1,power_t=0.5, verbose=3,
                            random_state=1234)
                                  
sgd_model_1.fit(train_data, np.ravel(train_y))

y_sgd = sgd_model_1.predict_proba(test_data)

y_sgd = y_sgd[:,1]

roc_auc_score(test_y, y_sgd)

log_loss(y_test,y_sgd)
concordance(pd.DataFrame(y_sgd),test_y)

########################## Extra Trees #######################

extf_model = ExtraTreesClassifier(n_estimators=10, random_state=1234)

param_grid = { 
    'n_estimators': [400,450,300,350],
    'max_depth': [8,10],
    'min_samples_leaf': [5,10],
    'max_features': [0.75,'sqrt','log2']
}
                                  
cv_extf_model = GridSearchCV(estimator = extf_model, param_grid = param_grid, cv = 5, scoring = 'roc_auc', verbose = 3)

cv_extf_model.fit(full_data, np.ravel(y))

print (cv_extf_model.best_params_)

#################### Tune around this only ##################

extf_model_1 = ExtraTreesClassifier(n_estimators=400, max_depth=10,min_samples_leaf=10, random_state=1234,max_features=0.75,
                                    verbose=3)
                                  
extf_model_1.fit(train_data, np.ravel(train_y))

y_ext = extf_model_1.predict_proba(test_data)

y_ext = y_ext[:,1]

roc_auc_score(test_y, y_ext)

log_loss(test_y,y_ext)
concordance(pd.DataFrame(y_ext),test_y)
###################### KNN #################################

knn_model = KNeighborsClassifier(n_neighbors=2, n_jobs=-1)

param_grid = { 
    'algorithm': ['ball_tree','kd_tree'],
    'n_neighbors': [25,30,15],
    'leaf_size': [25,20,15],
    'p': [2,3],
    'weights': ['distance','uniform'],
    'metric':['minkowski','euclidean']
}

param_grid = { 
    'algorithm': ['ball_tree'],
    'n_neighbors': [45,50,55],
    'leaf_size': [25,20,30],
    'p': [2],
    'weights': ['distance'],
    'metric':['minkowski']
}

cv_knn_model = GridSearchCV(estimator = knn_model, param_grid = param_grid, cv = 5, scoring = 'roc_auc', verbose = 3)

cv_knn_model.fit(final_data, np.ravel(y))

print (cv_knn_model.best_params_)

#############

knn_model_1 = KNeighborsClassifier(n_neighbors=30, algorithm = "ball_tree", leaf_size = 25, n_jobs=-1,
weights = 'distance',p=2,metric='minkowski')
                                  
knn_model_1.fit(train_data, np.ravel(train_y))

y_knn = knn_model_1.predict_proba(test_data)

y_knn = y_knn[:,1]

roc_auc_score(test_y, y_knn)

log_loss(y_test,y_knn)
concordance(pd.DataFrame(y_knn),test_y)


gbm_model_1 =  GradientBoostingClassifier(learning_rate = 0.01, n_estimators = 400, max_depth = 10, min_samples_leaf = 10,
                                        subsample = 0.7, max_features = 'log2', verbose = 3,random_state=1234)
                                  
gbm_model_1.fit(train_data, np.ravel(train_y))

y_gbm1 = gbm_model_1.predict_proba(test_data)

y_gbm1 = y_gbm1[:,1]

roc_auc_score(test_y, y_gbm1)

log_loss(test_y,y_gbm1)

temp=gbm_model_1.predict(test_data)
accuracy_score(np.array(test_y),temp)

precision_score(np.array(test_y),temp)
recall_score(np.array(test_y),temp)
f1_score(np.array(test_y),temp)
confusion_matrix(np.array(test_y),temp)

fimp = pd.DataFrame(gbm_model_1.feature_importances_)

fimp.to_csv("C:/Users/rsoni106/Documents/Work/Methodology Work/Seizure Detection/DataPreparation/Final_files/imp.csv")