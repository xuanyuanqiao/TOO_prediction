import pandas as pd
from pandas.io import spss
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

import sklearn
from sklearn import metrics
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold,StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from xgboost import XGBClassifier


genelist = pd.read_csv("fullGeneList.txt",sep='\t',header=None,names=["genes"])
samples = pd.read_csv("data_clinical_sample.txt",sep='\t', skiprows=[0,1,2,3])
somatic = pd.read_csv("data_mutations.txt", sep='\t', skiprows=[0,1])
CNV = pd.read_csv("data_cna.txt",sep='\t')

select_histo = ['Biliary-AdenoCA','ColoRect-AdenoCA', 'Eso-AdenoCA', 'Liver-HCC', 'Panc-AdenoCA', 'Panc-Endocrine', 'Stomach-AdenoCA']
samples = samples[samples["HISTOLOGY_ABBREVIATION"].isin(select_histo)]

select_col = ['SAMPLE_ID', 'PATIENT_ID', 'ORGAN_SYSTEM', 'HISTOLOGY_ABBREVIATION', 
       'PURITY', 'PLOIDY', 'PURITY_CONFUGURATION', 'WGD', 'SAMPLE_TYPE',
       'CANCER_TYPE', 'CANCER_TYPE_DETAILED', 'ICGC_SAMPLE_ID', 'SEQUENCING_TYPE', 
        'SAMPLE_CLASS', 'ANCESTRY_PRIMARY','TMB_NONSYNONYMOUS']
samples = samples[select_col]
samples = samples[samples["SAMPLE_CLASS"]=="Tumor"]
samples = samples[samples["PURITY"]>0.2]

labels = samples[["SAMPLE_ID","HISTOLOGY_ABBREVIATION"]]
sample_used = labels.SAMPLE_ID.tolist()

### generate somatic point mutation matrix
mut_cols = ["Hugo_Symbol","Entrez_Gene_Id","Variant_Type","Tumor_Sample_Barcode","t_alt_count"]
somatic = somatic[mut_cols]
somatic = somatic[somatic.t_alt_count>5]
somatic = somatic[somatic["Hugo_Symbol"].isin(genelist)]
somatic = somatic[somatic["Tumor_Sample_Barcode"].isin(sample_used)]

### generate copy number alteration matrix
CNV = CNV[CNV["Hugo_Symbol"].isin(genelist)]
sample_used.append("Hugo_Symbol")
CNV = CNV[sample_used]
CNV = CNV[-CNV.isnull().any(axis=1)] ## remove genes containing NaN value 
CNV = CNV.set_index("Hugo_Symbol")
CNV_t = CNV.T
CNV_t.isnull().values.any() ### check NaN in CNV matrix

### filter somatic point mutation matrix
genes_used = CNV_t.columns.tolist()
samples_used = CNV_t.index.tolist()
somatic = somatic[somatic["Tumor_Sample_Barcode"].isin(samples_used)]
somatic = somatic[somatic["Hugo_Symbol"].isin(genes_used)]
somatic_mtx = somatic[["Hugo_Symbol","Tumor_Sample_Barcode"]]
somatic_mtx["status"]=1
somatic_mtx = somatic_mtx.pivot_table(index=["Tumor_Sample_Barcode"],columns="Hugo_Symbol",values="status",fill_value=0)
somatic_mtx.isnull().values.any()

### filter CNV matrix
samples_used = somatic_mtx.index.tolist()
CNV_mtx = CNV_t.loc[samples_used]

### combine CNV and PM
somatic_mtx = somatic_mtx.add_suffix('_mut')
CNV_mtx = CNV_mtx.add_suffix('_CNV')
feat_mtx = pd.concat([somatic_mtx,CNV_mtx], axis=1)

labels = labels[labels["SAMPLE_ID"].isin(samples_used)]

CancerType = []
for i in range(0,len(feat_mtx)):
    sample = feat_mtx.index[i]
    study = labels[labels.SAMPLE_ID==sample]['HISTOLOGY_ABBREVIATION'].item()
    #label = study.split(' ')[0]
    #CancerType.append(label)
    CancerType.append(study)
    
types = pd.Series(CancerType)
targets = types.value_counts()
print(targets)
Cancers_df = targets.to_frame(name='Samples')
types = pd.Series(CancerType)
targets = types.value_counts()
targets = list(targets.index)
classes = [i for i in range(32)]

target_dict = {}
for key in targets:
    for value in classes:
        target_dict[key]=value
        classes.remove(value)
        break


mtx = feat_mtx.copy()
mtx['Target'] = CancerType
mtx = mtx.replace(target_dict) ## mtx["Target"] = mtx["Target"].map(target_dict)

## L1-based feature selection
x = np.array(feat_mtx)
y = np.array(mtx.Target)
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x, y)
model = SelectFromModel(lsvc, prefit=True)
x = model.transform(x)

def plot_cm(y_true, y_pred, classes, cmap=plt.cm.Blues, output):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm/cm.sum(axis=0)
    print(cm)
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm_percent, interpolation='nearest', cmap=cmap)
    #ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
          yticks=np.arange(cm.shape[0]),
          # ... and label them with the respective list entries
          xticklabels=classes, yticklabels=classes)
    ax.tick_params(labelsize=12)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j]),
                  ha="center", va="center",
                  color="white")
    
    plt.xlabel('True label',fontsize=12)
    plt.ylabel('Predicted label',fontsize=12)
    fig.tight_layout()
    savePath = os.path.join(output,"CM.jpg")
    plt.savefig(savePath, dpi=200)
    return ax

## train single ML model
def train(models,x,y,ensemble=False,output=False,k_fold=5,seed=0):
            ### 5 Fold Cross Validation 
    kf = StratifiedKFold(n_splits = k_fold, random_state = seed, shuffle = True)
    kf.get_n_splits(x, y)

    ModelNames = []
    for i in range(len(models)):
        ModelNames.append(str(models[i]).split('(')[0])

    if ensemble:
        foldcount = 0

        test_all = []
        pred_all = []

        cv_test = []
        cv_pred = []
        acc_list = []

        for train_index, test_index in kf.split(x, y):
            foldcount += 1
            print("K FOLD validation setp => {}".format(foldcount))

            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            clf2 = VotingClassifier(estimators = [(ModelNames[0],models[0]), (ModelNames[1],models[1]), (ModelNames[2],models[2])], voting='soft')
            clf2.fit(x_train, y_train)
            predictions = clf2.predict(x_test)
            
            report = classification_report(y_test, predictions,output_dict=True)
            df = pd.DataFrame(report).transpose()
            acc = df.loc['accuracy'][0]
            acc_list.append(acc)
            
            test_all = test_all + y_test.tolist()
            pred_all = pred_all + predictions.tolist()
            
            cv_test.append(y_test.tolist())
            cv_pred.append(predictions.tolist())
        
        print("\nAverage accuracy: \n",sum(acc_list)/5)
        plot_cm(test_all, pred_all, classes = targets,cmap="coolwarm",output)
      
    else:

        ### Store results
        fscores_list = [[] for _ in range(length)]
        precision_list = [[] for _ in range(length)]
        recall_list = [[] for _ in range(length)]
        acc_list = [[] for _ in range(length)]
        report_list = [[] for _ in range(length)]

        length = len(models)

        for i in range(length):
            model = models[i]
            model_name = str(models[i]).split('(')[0]
            print('-' * 30)
            foldcount = 0

            for train_index, test_index in kf.split(x, y):
                foldcount += 1
                print("K FOLD validation setp => {}".format(foldcount))

                x_train, x_test = x[train_index], x[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                model.fit(x_train,y_train)
                predictions = model.predict(x_test)
                
                report = classification_report(y_test, predictions,output_dict=True)
                df = pd.DataFrame(report).transpose()
                
                f_score = df.loc['macro avg','f1-score']
                precision = df.loc['macro avg','precision']
                recall= df.loc['macro avg','recall']
                acc = df.loc['accuracy'][0]
                
                report_list[i].append(report)
                fscores_list[i].append(f_score)
                precision_list[i].append(precision)
                recall_list[i].append(recall)
                acc_list[i].append(acc)

        avg_fscore = [sum(sub_list) / len(sub_list) for sub_list in fscores_list]
        avg_precision = [sum(sub_list) / len(sub_list) for sub_list in precision_list]
        avg_recall = [sum(sub_list) / len(sub_list) for sub_list in recall_list]
        avg_acc = [sum(sub_list) / len(sub_list) for sub_list in acc_list]
        result = pd.DataFrame(list(zip(avg_fscore,avg_precision,avg_recall,avg_acc)), 
                columns=['fscore','precision','recall','accuracy'], index=ModelNames)
        save_path = os.path.join(output,"single_ML_out.csv")
        result.to_csv(save_path)

models = [XGBClassifier(max_depth=100,learning_rate=0.1),
          LogisticRegression(penalty='l1',class_weight='balanced',random_state=42,solver='liblinear'),
          RandomForestClassifier(class_weight='balanced',random_state=42)]

train(models,x,y,ensemble=True,k_fold=5,seed=42,output="pcawg/")
