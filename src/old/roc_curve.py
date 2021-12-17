from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# true values and predicted probabilites for roc calculation
roc_y_true = []
roc_y_score_mrd = []
roc_y_score_iris = []


df = pd.read_excel(
    r'data/output/DroopyEyeMetricsResults.xlsx')

for i in range(len(df)):
    left_eye_pred_prob_mrd = 1 - (df.loc[i, "Left_Eye_Distance"] * 0.1)
    right_eye_pred_prob_mrd = 1 - (df.loc[i, "Right_Eye_Distance"] * 0.1)
    left_eye_pred_prob_iris = 1 - df.loc[i, "Left_Iris_Ratio"]
    right_eye_pred_prob_iris = 1 - df.loc[i, "Right_Iris_Ratio"]

    print(df.loc[i, "Ground_Truth_Prediction"], left_eye_pred_prob_iris, right_eye_pred_prob_iris)

    if df.loc[i, "Ground_Truth_Prediction"] == 'both droopy':
        roc_y_true.append(1)
        roc_y_score_iris.append(left_eye_pred_prob_iris)
        roc_y_score_mrd.append(left_eye_pred_prob_mrd)
        roc_y_true.append(1)
        roc_y_score_iris.append(right_eye_pred_prob_iris)
        roc_y_score_mrd.append(right_eye_pred_prob_mrd)

    elif df.loc[i, "Ground_Truth_Prediction"] == 'left droopy':
        roc_y_true.append(1)
        roc_y_score_iris.append(left_eye_pred_prob_iris)
        roc_y_score_mrd.append(left_eye_pred_prob_mrd)
        roc_y_true.append(0)
        roc_y_score_iris.append(right_eye_pred_prob_iris)
        roc_y_score_mrd.append(right_eye_pred_prob_mrd)

    elif df.loc[i, "Ground_Truth_Prediction"] == 'right droopy':
        roc_y_true.append(0)
        roc_y_score_iris.append(left_eye_pred_prob_iris)
        roc_y_score_mrd.append(left_eye_pred_prob_mrd)
        roc_y_true.append(1)
        roc_y_score_iris.append(right_eye_pred_prob_iris)
        roc_y_score_mrd.append(right_eye_pred_prob_mrd)

    elif df.loc[i, "Ground_Truth_Prediction"] == 'not droopy':
        roc_y_true.append(0)
        roc_y_score_iris.append(left_eye_pred_prob_iris)
        roc_y_score_mrd.append(left_eye_pred_prob_mrd)
        roc_y_true.append(0)
        roc_y_score_iris.append(right_eye_pred_prob_iris)
        roc_y_score_mrd.append(right_eye_pred_prob_mrd)

print(len(roc_y_true))

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
plt.plot(fpr_iris, tpr_iris, label='IRIS ratio ROC curve (area = %0.3f)' % roc_auc_iris)
plt.plot(fpr_mrd, tpr_mrd, label= 'MRD ROC curve (area = %0.3f)' % roc_auc_mrd)
plt.xlabel('False Positive Rate or (1 - Specifity)')
plt.ylabel('True Positive Rate or (Sensitivity)')
plt.legend(loc="lower right")
# plt.title("Area under ROC = {}".format(roc_auc))

plt.savefig('Ptosis_metrics_Mar19.png', bbox_inches='tight')
