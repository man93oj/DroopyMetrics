from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
# true values and predicted probabilites for roc calculation
roc_y_true = []
roc_y_score_mrd = []
roc_y_score_iris = []

# rootdir = os.path.join(os.path.join("data", "output"), "test")
# rootdir = os.path.join(os.path.join("data", "output"), "train")
rootdir = os.path.join(os.path.join("data", "output"), "val")
# rootdir = os.path.join(os.path.join("data", "output"), "test_cropped")
filepath = os.path.join(rootdir, "val_DroopyEyeMetricsResults.xlsx")
df = pd.read_excel(filepath)

for i in range(len(df)):
    roc_y_score_mrd.append(1 - (df.loc[i, "MRD1"] * 0.1))
    roc_y_score_iris.append(1 - df.loc[i, "Iris_Ratio"])
    roc_y_true.append(df.loc[i,"Ground_Truth"])

fpr_iris, tpr_iris, iris_thresholds = roc_curve(roc_y_true, roc_y_score_iris)
fpr_mrd, tpr_mrd, mrd_thresholds = roc_curve(roc_y_true, roc_y_score_mrd)

roc_auc_iris = auc(fpr_iris, tpr_iris)
roc_auc_mrd = auc(fpr_mrd, tpr_mrd)

iris_optimal_idx = np.argmax(tpr_iris - fpr_iris)
iris_optimal_threshold = iris_thresholds[iris_optimal_idx]

mrd_optimal_idx = np.argmax(tpr_mrd - fpr_mrd)
mrd_optimal_threshold = mrd_thresholds[mrd_optimal_idx]

print((1-mrd_optimal_threshold)*10, (1-iris_optimal_threshold))
# Plot ROC curve
plt.plot(fpr_iris, tpr_iris,
         label='IRIS ratio ROC curve (area = %0.3f)' % roc_auc_iris)
plt.plot(fpr_mrd, tpr_mrd, label='MRD ROC curve (area = %0.3f)' % roc_auc_mrd)
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.legend(loc="lower right")
# plt.title("Area under ROC = {}".format(roc_auc))

plt.savefig(os.path.join(rootdir, r'Ptosis_metrics.png'), bbox_inches='tight')