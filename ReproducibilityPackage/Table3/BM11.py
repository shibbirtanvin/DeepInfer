from tensorflow import keras
import numpy as np
import os
import pandas as pd
import csv
from collections import Counter
pd.options.mode.chained_assignment = None  # default='warn'
#load model
model_path = 'Models/BM11.h5'
model = keras.models.load_model(model_path)
model_name = (os.path.basename(model_path)).split('.')[0]
# #load dataset unseen test dataset
d_test = 'Data/BM_unseen.csv'
file = d_test
with open(file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=',')
    fields =reader.fieldnames
    print(reader.fieldnames)
    features = fields[:-1]
    features = fields
    print(features)
df = pd.read_csv(file, sep=',')
print(df.values)

X = df.iloc[:,:-1]
y = df.iloc[:, :1]
WP_df = df

#Bank Customer Actual and predictionvalue
actual =pd.read_csv('Other/bankcustomer_Actual_df.csv', sep=',')
predictionvalue =pd.read_csv('Other/bankcustomer_predictionsBM11.csv', sep=',')

print(actual)
X= df
y =actual
ActualOutcome = actual
print(ActualOutcome)

Implication = pd.read_csv('Other/BM11_Implication.csv', sep=',')
WP_df["DeepInfer_Implication"] = Implication
WP_df['DeepInfer_Implication'] = WP_df['DeepInfer_Implication'].replace({'Uncertain':'Wrong'})
print(WP_df["DeepInfer_Implication"])

print(ActualOutcome)
WP_df['Actual_Outcome'] = ActualOutcome
print(WP_df)

WP_df['Predicted_Outcome'] = predictionvalue
print(WP_df)

WP_df["GroundTruth"] = WP_df.apply(lambda x: "Correct" if (x['Actual_Outcome'] == x['Predicted_Outcome']) else "Wrong", axis=1)
#
print(WP_df)
WP_df["TruePositive"] = WP_df.apply(lambda x: "TP" if (x['GroundTruth'] == 'Correct' and x['DeepInfer_Implication'] == 'Correct') else "Not", axis=1)
WP_df["FalsePositive"] = WP_df.apply(lambda x: "FP" if (x['DeepInfer_Implication'] == 'Correct' and x['GroundTruth'] == 'Wrong') else "Not", axis=1)
WP_df["TrueNegative"] = WP_df.apply(lambda x: "TN" if (x['GroundTruth'] == 'Wrong' and x['DeepInfer_Implication'] == 'Wrong') else "Not", axis=1)
WP_df["FalseNegative"] = WP_df.apply(lambda x: "FN" if (x['GroundTruth'] == 'Correct' and x['DeepInfer_Implication'] == 'Wrong') else "Not", axis=1)

print(WP_df)

WP_df["ActualFalsePositive"] = WP_df.apply(lambda x: "TPAct" if (x['Actual_Outcome'] == x['Predicted_Outcome']) else "FPAct", axis=1)

Total_GT_Correct = WP_df["GroundTruth"].str.contains('Correct', regex=False).sum().astype(int)
Total_GT_Wrong = WP_df["GroundTruth"].str.contains('Wrong', regex=False).sum().astype(int)
Total_DeepInfer_Implication_Correct = WP_df["DeepInfer_Implication"].str.contains('Correct', regex=False).sum().astype(int)
Total_DeepInfer_Implication_Wrong = WP_df["DeepInfer_Implication"].str.contains('Wrong', regex=False).sum().astype(int)
Total_DeepInfer_Implication_Uncertain = WP_df["DeepInfer_Implication"].str.contains('Uncertain', regex=False).sum().astype(int)

##
FP_count =WP_df["FalsePositive"].str.contains('FP', regex=False).sum().astype(int)
TP_count =WP_df["TruePositive"].str.contains('TP', regex=False).sum().astype(int)
FN_count =WP_df["FalseNegative"].str.contains('FN', regex=False).sum().astype(int)
TN_count =WP_df["TrueNegative"].str.contains('TN', regex=False).sum().astype(int)
ActFP_count =WP_df["ActualFalsePositive"].str.contains('FPAct', regex=False).sum().astype(int)
ActTP_count =WP_df["ActualFalsePositive"].str.contains('TPAct', regex=False).sum().astype(int)
print("Actual FP:", ActFP_count)
print("Actual TP:", ActTP_count)
print("DeepInfer FP:",FP_count)
print("DeepInfer TP:",TP_count)
print("DeepInfer FN:",FN_count)
print("DeepInfer TN:",TN_count)
precision = TP_count/(TP_count+FP_count)
recall = TP_count/(TP_count+FN_count)
acc = (TP_count+TN_count)/(TP_count+TN_count+FP_count+FN_count)
tpr = TP_count/(TP_count+FN_count)
fpr = FP_count/(FP_count+TN_count)
f1 = (2* ((recall*precision)/(recall+precision)))
print("Precision:",(str(round(precision, 2))))
print("Recall:",(str(round(recall, 2))))
print("Accuracy:",(str(round(acc, 2))))
print("TPR:",(str(round(tpr, 2))))
print("FPR:",(str(round(fpr, 2))))
print("F-1 Score:",(str(round(f1, 2))))

#column_list=["Actual FP", "Actual TP", "DeepInfer FP", "DeepInfer TP", "DeepInfer FN", "DeepInfer TN", "Precision", "Recall", "Accuracy", "TPR", "FPR", "F-1 Score"]
# df_new = pd.DataFrame({'Actual FP': ActFP_count, 'Actual TP': ActTP_count, 'DeepInfer FP': FP_count , 'DeepInfer TP': TP_count , 'DeepInfer FN': FN_count , 'DeepInfer TN': TN_count ,
#                         'Precision': (str(round(precision, 2))), 'Recall': (str(round(recall, 2))) ,'Accuracy': (str(round(acc, 2))),
#                         'TPR': (str(round(tpr, 2))), 'FPR': (str(round(fpr, 2))) ,'F-1 Score': (str(round(f1, 2)))
#                        }, index=[0])
# #
# df_new.to_csv('table3.csv', columns = column_list, index=False)

with open('table3.csv', 'a') as p:
    theWriter = csv.writer(p)
    theWriter.writerow([str(model_name)] + [str(ActFP_count)] + [str(ActTP_count)] + [str(FP_count)] + [str(TP_count)] + [str(FN_count)] + [str(TN_count)] +
                       [str(round(precision, 2))] + [str(round(recall, 2))] + [str(round(acc, 2))] +  [str(round(tpr, 2))] +
                       [str(round(fpr, 2))] + [str(round(f1, 2))]
                       )





