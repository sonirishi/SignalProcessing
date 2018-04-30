# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 12:45:25 2016

@author: rsoni106
"""

import pandas as pd
import numpy as np
from sklearn import cross_validation as cv
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV

"""" final_data is the 75% dataframe created from signal data, test_data is the 25% split"""""

full_data = pd.read_csv("C:/Users/rsoni106/Documents/Work/Methodology Work/Seizure Detection/DataPreparation/Final_files/final_data_patient_t1.csv")

y = full_data[["event"]]

del full_data["event"]

train_data, test_data, train_y, test_y = train_test_split(full_data, y, test_size=0.25, random_state=1234)
train_y = train_y.reset_index(); test_y = test_y.reset_index()
del train_y["index"]; del test_y["index"]

scaler = StandardScaler()

train_data = scaler.fit_transform(train_data)

test_data = scaler.transform(test_data)

kfold = cv.StratifiedKFold(np.ravel(train_y),5)

new_data = pd.DataFrame(np.zeros((train_data.shape[0],1)))

test_data_n = pd.DataFrame(np.zeros((test_data.shape[0],1)))

train_fin_data = pd.DataFrame(np.zeros((train_data.shape[0],1)))

test_fin_data = pd.DataFrame(np.zeros((test_data.shape[0],1)))

""" RF parameters input """
   
def random_forest(train_data,var_count,y,validate, test_data):
    rf_model = RandomForestClassifier(criterion = "entropy", max_depth = 10, min_samples_leaf = 5, n_jobs = -1, 
                                  n_estimators = 350, oob_score = True, max_features = 'log2',random_state=1234)
    rf_model.fit(train_data,np.ravel(y))
    valid_pred = rf_model.predict_proba(validate)
    test_pred = rf_model.predict_proba(test_data)
    return valid_pred, test_pred
    
""" GBM parameters input """
    
def gbm_model(train_data,var_count,y,validate, test_data):
    gbm_model =  GradientBoostingClassifier(learning_rate = 0.05, n_estimators = 150, max_depth = 8, min_samples_leaf = 5,
                                        subsample = 0.75, max_features = 'log2',random_state=1234)
    gbm_model.fit(train_data,np.ravel(y))
    valid_pred = gbm_model.predict_proba(validate)
    test_pred = gbm_model.predict_proba(test_data)
    return valid_pred, test_pred
    
""" SGD parameters input """
    
def sgd_model(train_data,var_count,y,validate, test_data):    
    sgd_model = SGDClassifier(alpha = 0.0003, n_iter = 450, loss = "log", penalty = 'l2', 
                              n_jobs = -1,power_t=0.5,random_state=1234)
    sgd_model.fit(train_data,np.ravel(y))
    valid_pred = sgd_model.predict_proba(validate)
    test_pred = sgd_model.predict_proba(test_data)
    return valid_pred, test_pred
    
def extra_forest(train_data,var_count,y,validate, test_data):
    extf_model = ExtraTreesClassifier(n_estimators=350, max_depth=10,min_samples_leaf=10, random_state=1234,max_features=0.75)
    extf_model.fit(train_data,np.ravel(y))
    valid_pred = extf_model.predict_proba(validate)
    test_pred = extf_model.predict_proba(test_data)
    return valid_pred, test_pred    

def knn_model(train_data,var_count,y,validate, test_data):
    knn_model = KNeighborsClassifier(n_neighbors=175,algorithm = "ball_tree",leaf_size = 20,
                                   n_jobs=-1,weights='distance',metric='manhattan')
    knn_model.fit(train_data,np.ravel(y))
    valid_pred = knn_model.predict_proba(validate)
    test_pred = knn_model.predict_proba(test_data)
    return valid_pred, test_pred
    
model_list = [random_forest,gbm_model,sgd_model,extra_forest,knn_model]    

""" Create the predictor variable - todo """

for i in range(len(model_list)):
    for j,(train_index, test_index) in enumerate(kfold):
        train_data1 = np.array(train_data[train_index])
        y1 = np.array(train_y.loc[train_index])
        validate = np.array(train_data[test_index])
        var_count = train_data1.shape[1]
        pred1, pred2 = model_list[i](train_data1,var_count,y1,validate,test_data)
        temp = pd.DataFrame(pred1)[[1]]
        """ Indexing is extremely important in python """
        temp.index = test_index
        new_data.loc[test_index] = temp
        test_data_n[[j]] = pd.DataFrame(pred2)[[1]]
        print("iteration complete")
    test_predict_mean = np.mean(test_data_n,axis=1)
    train_fin_data = pd.concat((train_fin_data,new_data),axis=1)
    test_fin_data = pd.concat((test_fin_data,test_predict_mean),axis = 1)
        
train_fin_data.columns = ["temp","rf","gbm","sgd","exf","knn"]
test_fin_data.columns = ["temp","rf","gbm","sgd","exf","knn"]

del train_fin_data["temp"]
del test_fin_data["temp"]

########################## Concordance metrics ###########################

def concordance(model_vec,test_y):
    probab = pd.concat((model_vec,test_y),axis=1)
    vec0 = probab[probab.loc[:,"event"]== 0]
    vec1 = probab[probab.loc[:,"event"]== 1]
    vec0 = vec0.assign(temp = 1)
    vec1 = vec1.assign(temp = 1)
    final = pd.merge(vec0, vec1, on='temp', how='outer')
    temp = pd.DataFrame(np.zeros((final.shape[0],1)))
    temp.columns=["ind"]
    final=final.rename(columns = {'0_x':'probab0','0_y':'probab1'})
    for i in range(final.shape[0]):
        if(final.iloc[i]["probab1"] > final.iloc[i]["probab0"]):
            temp.iloc[i]["ind"] = 1
        elif(final.iloc[i]["probab1"] < final.iloc[i]["probab0"]): 
            temp.iloc[i]["ind"] = -1
        else:
            temp.iloc[i]["ind"] = 0
    concordance = sum(temp.loc[:,"ind"]==1)
    discordance = sum(temp.loc[:,"ind"]==-1)
    tied_pair = sum(temp.loc[:,"ind"]==0)
    
    totalpairs = temp.shape[0]
    
    gamma = (concordance - discordance)/(concordance + discordance + tied_pair)
    perconcordance = concordance/totalpairs
    per_tied = tied_pair/totalpairs
    N = test_y.shape[0]
    
    taua = 2*(concordance-discordance)/(N*(N-1))
    
    return perconcordance, gamma, per_tied, taua

##################################################################

logit = LogisticRegression(random_state=1234,fit_intercept=False)

param_grid = {
    'penalty': ['l2','l1'],
    'C': [0.01,0.012,0.02,0.025,0.03],
    'max_iter': [50,75,100,125,55]
}

cv_logit_model = GridSearchCV(estimator = logit, param_grid = param_grid, cv = 5, scoring = 'roc_auc', verbose = 3)

cv_logit_model.fit(train_fin_data, np.ravel(train_y))

print (cv_logit_model.best_params_)

""" Final ensemble model """

final_rf = RandomForestClassifier(criterion = "entropy", max_depth = 3, min_samples_leaf = 1, n_jobs = -1, 
                                  n_estimators = 20, max_features = 0.7, oob_score = False,random_state=1234)

final_rf.fit(train_fin_data, np.ravel(train_y))

y_final_rf = final_rf.predict_proba(test_fin_data)

y_final_rf = y_final_rf[:,1]

roc_auc_score(test_y, y_final_rf)

log_loss(test_y,y_final_rf)

concordance(pd.DataFrame(y_final_rf),test_y)

#################################################################

test_y = pd.DataFrame(test_y)

y_final_rf = pd.DataFrame(y_final_rf)

final_data = pd.concat((test_y,y_final_rf),axis=1)

final_data.to_csv("C:/Users/rsoni106/Documents/Work/Methodology Work/Seizure Detection/DataPreparation/Final_files/final_data.csv")


test_y = pd.DataFrame(test_y)

y_gbm1 = pd.DataFrame(y_gbm1)

gbm_data = pd.concat((test_y,y_gbm1),axis=1)

gbm_data.to_csv("C:/Users/rsoni106/Documents/Work/Methodology Work/Seizure Detection/DataPreparation/Final_files/gbm_data.csv")

################ Stacking using MCMC #############################

import pymc3 as pm

data = dict(x1=train_fin_data["rf"], x2=train_fin_data["gbm"], x3=train_fin_data["sgd"],
            x4=train_fin_data["exf"],x5=train_fin_data["knn"],y=np.ravel(train_y))

with pm.Model() as model:
    pm.glm.glm('y ~ x1 + x2 + x3 + x4 + x5',data,family=pm.glm.families.Binomial())
    step = pm.NUTS()
    trace = pm.sample(100,step,progressbar=True)
    
predicted_test = np.median(trace.x1)*test_fin_data["rf"] + np.median(trace.x2)*test_fin_data["gbm"] + np.median(trace.x3)*test_fin_data["sgd"] + test_fin_data["exf"]*np.median(trace.x4) + test_fin_data["knn"]*np.median(trace.x5) + np.median(trace.Intercept)

roc_auc_score(test_y, predicted_test)

########using mean of the trace#####################

predicted_test1 = np.mean(trace.x1)*test_fin_data["rf"] + np.mean(trace.x2)*test_fin_data["gbm"] + np.mean(trace.x3)*test_fin_data["sgd"] + test_fin_data["exf"]*np.mean(trace.x4) + test_fin_data["knn"]*np.mean(trace.x5) + np.mean(trace.Intercept)

roc_auc_score(test_y, predicted_test1)

################## Metropolis #################

data_1 = dict(x1=train_fin_data["rf"], x2=train_fin_data["gbm"], x3=train_fin_data["sgd"],
            x4=train_fin_data["exf"],x5=train_fin_data["knn"],y=np.ravel(train_y))

with pm.Model() as model:
    pm.glm.glm('y ~ 0 + x1 + x2 + x3 + x4 + x5',data_1,family=pm.glm.families.Binomial())
    start = pm.find_MAP()
    step = pm.Metropolis()
    trace_m = pm.sample(10,step,start=start,progressbar=True)
    
with pm.Model() as model:
    pm.glm.glm('y ~ 0 + x1 + x2 + x3 + x4 + x5',data_1,family=pm.glm.families.Binomial())
    start = pm.find_MAP()
    step = pm.Slice()
    trace_m = pm.sample(100,step,start=start,progressbar=True)    
    
predicted_test2 = np.median(trace_m.x1)*test_fin_data["rf"] + np.median(trace_m.x2)*test_fin_data["gbm"] + np.median(trace_m.x3)*test_fin_data["sgd"] + test_fin_data["exf"]*np.median(trace_m.x4) + test_fin_data["knn"]*np.median(trace_m.x5)

roc_auc_score(test_y, predicted_test2)

logit = LogisticRegression(random_state=1234,fit_intercept=False,max_iter=50,penalty='l1',C=0.01)

logit.fit(train_fin_data,train_y)

predicted_test3 = logit.predict_proba(test_fin_data)

predicted_test3 = predicted_test3[:,1]

roc_auc_score(test_y, predicted_test3)

################# Export 5 fold the output to csv #####################

train_fin_data.to_csv("C:/Users/rsoni106/Documents/Work/Methodology Work/Seizure Detection/DataPreparation/Final_files/train_oof.csv")

test_fin_data.to_csv("C:/Users/rsoni106/Documents/Work/Methodology Work/Seizure Detection/DataPreparation/Final_files/test_oof.csv")

train_y.to_csv("C:/Users/rsoni106/Documents/Work/Methodology Work/Seizure Detection/DataPreparation/Final_files/train_y.csv")

test_y.to_csv("C:/Users/rsoni106/Documents/Work/Methodology Work/Seizure Detection/DataPreparation/Final_files/test_y.csv")

#################### Build a 5 fold MCMC ensemble #####################

newfold = cv.StratifiedKFold(np.ravel(train_y),5)

predicted_test = np.zeros((test_fin_data.shape[0],5))

for i, (train_index,test_index) in enumerate(newfold):
    train_ens = train_fin_data.iloc[train_index]
    train_y_ens = train_y.iloc[train_index]
    data_1 = dict(x1=train_ens["rf"], x2=train_ens["gbm"], x3=train_ens["sgd"],
            x4=train_ens["exf"],x5=train_ens["knn"],y=np.ravel(train_y_ens))
    with pm.Model() as model:
        pm.glm.glm('y ~ 0 + x1 + x2 + x3 + x4 + x5',data_1,family=pm.glm.families.Binomial())
        start = pm.find_MAP()
        step = pm.Metropolis()
        trace_m = pm.sample(2000,step,start=start,progressbar=True)
    predicted_test[:,i] = np.median(trace_m.x1)*test_fin_data["rf"] + np.median(trace_m.x2)*test_fin_data["gbm"] + np.median(trace_m.x3)*test_fin_data["sgd"] + test_fin_data["exf"]*np.median(trace_m.x4) + test_fin_data["knn"]*np.median(trace_m.x5)
    
predicted_test_fin = predicted_test.mean(axis=1)

roc_auc_score(test_y, predicted_test2)

################ Factorization Machines ####################

import pywFM as fm

fm_logit = fm.FM(task="classification")

fm_logit.run(train_data,np.ravel(train_y),test_data,np.ravel(test_y))

train_fin_data.drop(train_fin_data[[0]],axis=1,inplace=True)
train_y.drop(train_y[[0]],axis=1,inplace=True)