import pandas as pd
from sklearn.metrics import roc_curve, auc


df_iDNN = pd.read_csv('E:/QD065LPSc/Ksuc/models/integrated_DNN/iDNN_y_label&score.csv')
label = df_iDNN['label']
score = df_iDNN['score']
iDNN_fpr, iDNN_tpr, iDNN_threshold = roc_curve(label, score)
iDNN_auc = auc(iDNN_fpr, iDNN_tpr)
Sn = list(iDNN_tpr)
iDNN_fpr = list(iDNN_fpr)
for i, fpr in enumerate(iDNN_fpr):
    iDNN_fpr[i] = 1 - fpr
Sp = list(iDNN_fpr)
threshold = list(iDNN_threshold)

df_Sn = pd.DataFrame(Sn)
df_Sp = pd.DataFrame(Sp)
df_threshold = pd.DataFrame(threshold)

df = pd.concat([df_Sn, df_Sp, df_threshold], axis=1)
df.to_csv('E:/QD065LPSc/Ksuc/Sn&Sp.csv', index=False)