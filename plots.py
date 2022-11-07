import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import classification_report,confusion_matrix

### define functions
def plot_acc_per_class(cv_test,cv_pred,targets, save_path="",figsize=(8,3),color = color_palette):
    pred_per_class = []
    for i in range(5):
        fold_test = cv_test[i]
        fold_pred = cv_pred[i]
        matrix = confusion_matrix(fold_test, fold_pred)
        acc_list = matrix.diagonal()/matrix.sum(axis=0)
        pred_per_class.append(acc_list.tolist())

    df_class_all = pd.DataFrame(np.array(pred_per_class).tolist(),columns=targets)
    df_class_all = df_class_all*100
    sns.set_palette(color_palette)
    plt.figure(figsize=figsize)
    ax = sns.boxplot(data=df_class_all)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
    plt.yticks(fontsize=12)
    plt.ylabel('Accuracy/%',size=12)
    plt.savefig(save_path, dpi=400,bbox_inches='tight')
    plt.show()

def plot_cm(y_true, y_pred, classes, cmap=plt.cm.Blues, save_path="", figure_size = (5,5)):
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = np.transpose( np.transpose(cm) / cm.astype(np.float).sum(axis=1) )
    print(cm)
    fig, ax = plt.subplots(figsize=figure_size)
    im = ax.imshow(cm_percent, interpolation='nearest', cmap=cmap)

    ax.set(xticks=np.arange(cm.shape[1]),
          yticks=np.arange(cm.shape[0]),
          xticklabels=classes, yticklabels=classes)
    ax.tick_params(labelsize=12)

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right",
            rotation_mode="anchor")
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j]),
                  ha="center", va="center",
                  color="white")
    
    plt.xlabel('Predicted label',fontsize=12)
    plt.ylabel('True label',fontsize=12)
    fig.tight_layout()
    plt.savefig(save_path, dpi=400)
    return ax