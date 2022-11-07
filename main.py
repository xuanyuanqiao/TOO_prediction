import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, KFold,StratifiedKFold,GridSearchCV,RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.preprocessing import MinMaxScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_curve, auc, top_k_accuracy_score
from sklearn.utils import class_weight
import xgboost as xgb
from itertools import cycle
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('always')

from trains import train

genelist = pd.read_csv("data/fullGeneList.txt",names=["Genes"])
genelist = genelist.Genes.tolist() 

def get_key(target_dict, val):
    for key,value in target_dict.items():
        if val ==value:
            return key

### TCGA feature matrix
rmd = pd.read_csv("data/tcga/rmd_all_tcga.csv",index_col=0)
rmd_df = rmd.T
CNV_mtx = pd.read_csv("data/tcga/CNV_all_tcga.csv",index_col=0)
somatic_mtx = pd.read_csv("data/tcga/somatic_all_tcga.csv",index_col=0)
labels = pd.read_csv("data/tcga/labels_all_tcga.csv",index_col=0)

CNV_mtx = CNV_mtx.div(CNV_mtx.max().max(),axis=0) ## Normalize CNV to [-1,1]

somatic_mtx = somatic_mtx.add_suffix('_mut')
CNV_mtx = CNV_mtx.add_suffix('_CNV')
feat_mtx = pd.concat([somatic_mtx,CNV_mtx,rmd_df], axis=1)

### Generate Labels
cancers = []
for i in range(0,len(labels)):
    study = labels.iloc[[i]]["studyId"].item()
    tmp = str(study)
    cancer = tmp.split('_')[0]
    cancers.append(cancer)

labels["cancer"] = cancers

CancerType = []

for i in range(0,len(feat_mtx)):
    sample = feat_mtx.index[i]
    study = labels[labels.sampleId==sample]['cancer'].item()
    CancerType.append(study)
    
types = pd.Series(CancerType)
targets = types.value_counts()

targets = list(targets.index)
classes = [i for i in range(len(targets))]

target_dict = {}
for key in targets:
    for value in classes:
        target_dict[key]=value
        classes.remove(value)
        break

mtx = feat_mtx.copy()
mtx['Target'] = CancerType
mtx = mtx.replace(target_dict)

## Hyperparameter tuning
params = {
    'objective':['multi:softprob'],
    'n_estimators': [100, 200, 500],
    'learning_rate': [0.01,0.05,0.1],
    'booster': ['gbtree', 'gblinear'],
    'gamma': [0, 0.5, 1],
    'reg_alpha': [0, 0.5, 1],
    'reg_lambda': [0.5, 1, 5],
    'base_score': [0.2, 0.5, 1],
    'max_depth': [50,100,200],
    'min_child_weight': [1,5,10],
    'subsample': [0.8],
    'colsample_bytree' : [0.3, 0.5, 0.7]
}

best_model = RandomizedSearchCV(XGBClassifier(), params, n_jobs=1, n_iter = 3, cv=5, scoring='accuracy')
best_model.fit(np.array(feat_mtx), np.array(mtx.Target))

print('Best score:', best_model.best_score_)
print('Best score:', best_model.best_params_)

## feature selection RFE
x = np.array(feat_mtx)
y = np.array(mtx.Target)
estimator = xgb.XGBClassifier(colsample_bytree=0.3,gamma=0, learning_rate=0.1,
              max_depth=100, min_child_weight=5, n_estimators=500,subsample = 0.8,objective='multi:softprob')

selector = RFE(estimator, n_features_to_select=500, step=50)
selector = selector.fit(np.array(feat_mtx), np.array(mtx.Target))
selector.ranking_

feature_bool = selector.support_
feat_mtx_select = feat_mtx.loc[:,feature_bool]

## Weighted Samples
classes_weights = class_weight.compute_sample_weight(
    class_weight='balanced',
    y=mtx.Target
)
feat_mtx_select["Weight"] = classes_weights

## train ML model
x = np.array(feat_mtx_select)
y = np.array(mtx.Target)
X_trainval, X_valtest, y_trainval, y_valtest = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
xgb_fit = xgb.XGBClassifier(colsample_bytree=0.3,gamma=0, learning_rate=0.1,
              max_depth=100, min_child_weight=5, n_estimators=500,subsample = 0.8,objective='multi:softprob')

train(xgb_fit,X_trainval,y_trainval,targets, num_class = 32, k_fold=5,seed=42,output="tcga/")
## eval on hold-out test set
X_test = X_valtest[:,:-1]
preds_valtest = xgb_fit.predict(X_test)
df_result = classification_report(y_valtest,preds_valtest,output_dict=True)
df_result  = pd.DataFrame(df_result).transpose()
print('{Precision: }'.format(sum(df_result.iloc[:32,0])/32))
print('{Recall: }'.format(sum(df_result.iloc[:32,1])/32))
print('{F1-score: }'.format(sum(df_result.iloc[:32,2])/32))

