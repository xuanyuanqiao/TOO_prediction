import pandas as pd
import numpy as np
import os
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report,top_k_accuracy_score
from plots import plot_acc_per_class, plot_cm

color_palette = ["#42c0c7","#de4352","#5cc356","#a651cb","#a2bb31","#6264de","#52902c","#e26ccd","#459654","#e14383","#5ac397","#b03d8d","#b2aa48","#4875d7","#d49d35","#855db7","#e0742b","#7999e3","#c04027","#50a4d3","#985c29","#5a67a6","#74721f","#c28cd1","#9bb66d","#b14457","#347e5b","#df83a2","#6a733a","#954e6d","#cf9f68","#df7d65"]

def train(model,x,y,targets, num_class=32, output="",k_fold=5,seed=0):
    ### 5 Fold Cross Validation 
    kf = StratifiedKFold(n_splits = k_fold, random_state = seed, shuffle = True)
    kf.get_n_splits(x, y)

    ### Store results
    test_all = []
    pred_all = []
    probs_all = [[] for i in range(num_class)]

    cv_test = []
    cv_pred = []

    fscores_list = []
    precision_list = []
    recall_list = []
    acc_list = []
    top3_acc_list = []
    report_list = []

    print('-' * 30)
    foldcount = 0

    for train_index, test_index in kf.split(x, y):
        foldcount += 1
        print("K FOLD validation setp => {}".format(foldcount))

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        train_weight = x_train[:,-1]
        x_train = x_train[:,:-1]
        x_test = x_test[:,:-1]
        probs = model.fit(x_train, y_train,sample_weight=train_weight).predict_proba(x_test)
        predictions = model.predict(x_test)
        for i in range(num_class):
            probs_all[i] = probs_all[i] + probs[:,i].tolist()
        
        test_all = test_all + y_test.tolist()
        pred_all = pred_all + predictions.tolist()
        report = classification_report(y_test, predictions,output_dict=True)
        df = pd.DataFrame(report).transpose()
        
        f_score = df.loc['macro avg','f1-score']
        precision = df.loc['macro avg','precision']
        recall= df.loc['macro avg','recall']
        acc = df.loc['accuracy'][0]
        
        report_list.append(report)
        fscores_list.append(f_score)
        precision_list.append(precision)
        recall_list.append(recall)
        acc_list.append(acc)
        top3_acc = top_k_accuracy_score(y_test,probs, k=3)
        top3_acc_list.append(top3_acc)
    
    probs_df = pd.DataFrame(np.array(probs_all).T,columns = targets)
    pred_df = pd.DataFrame(np.array([test_all,pred_all]).T,columns=["y_true","y_pred"])
    result_df = pd.concat([probs_df,pred_df],axis=1)
    save_path = os.path.join(output,"CV_test_out.csv")
    result_df.to_csv(save_path)

    performance = pd.DataFrame(list(zip(fscores_list,precision_list,recall_list,acc_list,top3_acc_list)), 
        columns=['fscore','precision','recall','accuracy','top3_accuracy'])
    performance.to_csv(output,"CV_performance,csv")

    plot_acc_per_class(cv_test,cv_pred,save_path=os.path.join(output,"TCGA_CV_test_acc_per_class.jpg"),color = color_palette)
    plot_cm(test_all, pred_all, classes= targets,cmap="coolwarm", save_path=os.path.join(output,"TCGA_CV_test_CM.jpg"))


