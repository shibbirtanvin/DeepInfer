from tensorflow import keras
import numpy as np
import os
import pandas as pd
import csv
from collections import Counter
pd.options.mode.chained_assignment = None  # default='warn'
#load model
model_path = 'Models/BM3.h5'
model = keras.models.load_model(model_path)
model_name = (os.path.basename(model_path)).split('.')[0]
n = [0.75, 0.90, 0.95, 0.99]
cond = ['>=','<=','>','<','==','!=']
l = len(model.layers)
Q=[]

print(l)

# Getting Weights and Biases for each layer and Compute Fiction variable \gamma
w = np.array([])
Weight = [] #Storing Weights
Bias = [] #Storing Biases
Gamma = [] #Storing Weights
N = [] #Activation function list
Gamma_tr =[]
X =[]

for i in range(0,l):
    print(i)
    w = model.layers[i].get_weights()[0]
    Weight.append(w)
    b = model.layers[i].get_weights()[1]
    Bias.append(b)
    a = model.layers[i].get_config()['activation']
    N.append(a)
    w_tr = np.transpose(w)
    #print(f'Array:\n{w}')
    #print(f'Transposed Array:\n{w_tr}')
    A = np.matmul(w,w_tr)
    A_inv = np.linalg.inv(A)
    B = np.matmul(w_tr,A_inv)
    Gamma.append(B)
    B_tr = np.transpose(B)
    Gamma_tr.append(B_tr)

print(len(Weight))
print(len(Bias))
print(len(N))
print(len(Gamma))
for i in range(len(Weight)):
    print("W_",i+1,":", Weight[i])

for i in range(len(Bias)):
    print("B_",i+1,":", Bias[i])

for i in range(len(N)):
    print("Activation function of layer_a", i + 1, ":", N[i])

for i in range(len(Gamma)):
    print("Gamma_", i + 1, ":", Gamma[i])

for i in range(l):
    for j in range(len(n)):
        for k in range(len(cond)):
            M = np.matmul((Gamma_tr[i]*n[j]), -(Bias[i]))
            print("X_", i + 1, "postcondition:",cond[k], n[j])
            print(M)

def beta(N,Q,l,i):
    """This is a recursive function
    to find the wp of N"""
    p =0
    N0 = N[0]
    if l == 1:
        if (N0 == 'linear'):
            M = np.matmul((Gamma_tr[i] * Q), -(Bias[i]))
            print("X_", i + 1, "postcondition:", cond[0], Q)
            print(M)

        if (N0 == 'relu'):
            try:
                M1 = np.matmul((Gamma_tr[i] * Q), -(Bias[i]))
            except ValueError:
                pass
            M2 = np.matmul(Gamma_tr[i], -(Bias[i]))
            print("X_", i + 1, "postcondition:", cond[0], Q)
            print(M2)
            M = M2

        if (N0 == 'sigmoid'):
            M = np.matmul((Gamma_tr[i] * np.log(Q / (1 - Q))), -(Bias[i]))
            print("X_", i + 1, "postcondition:", cond[0], Q)
            print(M)

        if (N0 == 'tanh'):
            n_tanh = abs((Q - 1) / (Q + 1))
            M = np.matmul((Gamma_tr[i] * (0.5 * np.log(n_tanh))), -(Bias[i]))
            print("X_", i + 1, "postcondition:", cond[0], Q)
            print(M)

        return M
    else:
        N0 = [N[0]]
        N1 = N[1:]
        l = len(N1)
        wp2 = beta(N1,Q,l,i)
        i = i - 1
        Q =wp2
        while (i >=0):
            wp1 = beta(N0, Q, 1, i)
            i = i - 1
            Q =wp1
        return Q

Q=n[2]
print(Q)
i=l-1
wp = beta(N,Q,l,i)
print("wp: ")
print(wp)

#load dataset unseen test dataset
d_test = 'Data/BM_unseen.csv'

file = d_test

with open(file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=',')
    names =reader.fieldnames
    print(reader.fieldnames)
    features = np.array([])
    for i in names:
        features = np.append(features, i)

    print('Numpy Array: ', features)
    print('feature length: ', features.size)
    print('WP size ', wp.size)
if(wp.size == features.size):
    print("True")
    WPdictionary = dict(zip(features, wp))
    print(WPdictionary)
    for key, value in WPdictionary.items():
        print(key)
else:
    #print(WP_values)
    feature_counter = 1
    feature_count = np.array([])
    for i in wp:
        print("feature_counter",feature_counter, '>=', "{0:.2f}".format(i))
        feature_count = np.append(feature_count,feature_counter)
        feature_counter = feature_counter + 1
    WPdictionary = dict(zip(feature_count, wp))
    print(WPdictionary)

for key, value in WPdictionary.items():
    print(key, '>=', value)


WPdict = WPdictionary

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
rslt_df = df

print(df)
print(rslt_df)

print(rslt_df.size)
print(rslt_df)

#Bank Customer Actual and predictionvalue
actual =pd.read_csv('Other/bankcustomer_Actual_df.csv', sep=',')
predictionvalue =pd.read_csv('Other/bankcustomer_predictionsBM3.csv', sep=',')

print(actual)
X= df
y =actual
ActualOutcome = actual
print(ActualOutcome)


rslt_df = df
print(rslt_df.size)
print(rslt_df)


for key, value in WPdict.items():
    print(key, '->', value)
    rslt_df[key] = rslt_df[key].map(lambda x: "Y" if x >= value else "N")
    print(rslt_df)

WP_df = rslt_df

WP_violation_count =WP_df.apply(lambda x: x.str.contains("N")).sum()

WP_satisfied_count =WP_df.apply(lambda x: x.str.contains("Y")).sum()

WP_df['total_Sat'] = 0

print(WP_df)
for key, value in WPdict.items():
    print(key, '->', value)
    WP_df['total_Sat'] = WP_df['total_Sat']+ WP_df[key].str.contains('Y', regex=False).astype(int)
    print(WP_df)


WP_df['total_Viol'] = 0
for key, value in WPdict.items():
    print(key, '->', value)
    WP_df['total_Viol'] = WP_df['total_Viol']+ WP_df[key].str.contains('N', regex=False).astype(int)
    print(WP_df)


violationMean= WP_df['total_Viol'].mean()

print(violationMean)

satisfiedMean = WP_df['total_Sat'].mean()

print(satisfiedMean)

print("WP_violation_count")
print(WP_violation_count)
print("WP_satisfied_count")
print(WP_satisfied_count)

important_features = []
unimportant_featues = []
print(WP_violation_count.mean())
print(WP_satisfied_count.mean())

for i, v in WP_violation_count.items():
    #print('index: ', i, 'value: ', v)
    if (v <= WP_violation_count.mean()):
    #if (v <= violationMean):
        print('index: ', i, 'value: ', v)
        important_features.append(str(i))
    else:
        unimportant_featues.append(str(i))

print("More_Important_features: ")
print(important_features)
print("Less_Important_features: ")
print(unimportant_featues)

WP_df["vCount_MoreImpFeat"] = 0
WP_df["vCount_LessImpFeat"] = 0
for i in  important_features:
    WP_df["vCount_MoreImpFeat"] = WP_df["vCount_MoreImpFeat"]+ WP_df[i].str.contains('N', regex=False).astype(int)
for i in  unimportant_featues:
    WP_df["vCount_LessImpFeat"] = WP_df["vCount_LessImpFeat"]+ WP_df[i].str.contains('N', regex=False).astype(int)

print(WP_df)
WP_df["DeepInfer_Implication"] = WP_df.apply(lambda x: "Correct" if (x['vCount_MoreImpFeat'] == 0) else "Uncertain", axis=1)
WP_df["DeepInfer_Implication"] = WP_df.apply(lambda x: "Wrong" if (x['vCount_LessImpFeat'] > violationMean and x['vCount_LessImpFeat'] != x['vCount_MoreImpFeat']) else "Correct", axis=1)
WP_df["DeepInfer_Implication"] = WP_df.apply(lambda x: "Uncertain" if (x['vCount_LessImpFeat'] == x['vCount_MoreImpFeat'] and x['vCount_MoreImpFeat'] != 0) else x["DeepInfer_Implication"], axis=1)

#Appending ActualOutcome Column
print(ActualOutcome)
WP_df['Actual_Outcome'] = ActualOutcome
print(WP_df)

#predictionvalue
WP_df['Predicted_Outcome'] = predictionvalue
print(WP_df)


WP_df["GroundTruth"] = WP_df.apply(lambda x: "Correct" if (x['Actual_Outcome'] == x['Predicted_Outcome']) else "Wrong", axis=1)

Total_GT_Correct = WP_df["GroundTruth"].str.contains('Correct', regex=False).sum().astype(int)
Total_GT_Wrong = WP_df["GroundTruth"].str.contains('Wrong', regex=False).sum().astype(int)
Total_DeepInfer_Implication_Correct = WP_df["DeepInfer_Implication"].str.contains('Correct', regex=False).sum().astype(int)
Total_DeepInfer_Implication_Wrong = WP_df["DeepInfer_Implication"].str.contains('Wrong', regex=False).sum().astype(int)
Total_DeepInfer_Implication_Uncertain = WP_df["DeepInfer_Implication"].str.contains('Uncertain', regex=False).sum().astype(int)

print("Total_GroundTruth_Correct:",Total_GT_Correct)
print("Total_GroundTruth_Incorrect:",Total_GT_Wrong)
print("Total Violation:", WP_df['total_Viol'].sum())
print("Total Satisfied:", WP_df['total_Sat'].sum())
print("Total_DeepInfer_Implication_Correct:",Total_DeepInfer_Implication_Correct)
print("Total_DeepInfer_Implication_Incorrect:",Total_DeepInfer_Implication_Wrong)
print("Total Uncertain:",Total_DeepInfer_Implication_Uncertain)

#column_list=["Ground Truth #Correct", "Ground Truth #Incorrect", "DeepInfer #Violation", "DeepInfer #Satisfaction", "DeepInfer #Correct", "DeepInfer #Incorrect", "DeepInfer #Uncertain"]

# df_new = pd.DataFrame({'Model': model_name, 'Ground Truth #Correct': Total_GT_Correct, 'Ground Truth #Incorrect': Total_GT_Wrong ,
#                        'DeepInfer #Violation': WP_df['total_Viol'].sum(),
#                        'DeepInfer #Satisfaction': WP_df['total_Sat'].sum(),
#                         'DeepInfer #Correct': Total_DeepInfer_Implication_Correct, 'DeepInfer #Incorrect': Total_DeepInfer_Implication_Wrong ,'DeepInfer #Uncertain': Total_DeepInfer_Implication_Uncertain}, index=[0])
# #
# df_new.to_csv('table2.csv', index=False)

with open('table2.csv', 'a') as p:
    theWriter = csv.writer(p)
    theWriter.writerow([str(model_name)] + [str(Total_GT_Correct)] + [str(Total_GT_Wrong)] +
                       [str(WP_df['total_Viol'].sum())] + [str(WP_df['total_Sat'].sum())] + [str(Total_DeepInfer_Implication_Correct)]
                       + [str(Total_DeepInfer_Implication_Wrong)] + [Total_DeepInfer_Implication_Uncertain])