import csv

from numpy import loadtxt

import numpy as np

import pandas as pd
from collections import Counter
pd.options.mode.chained_assignment = None  # default='warn'
import inferDataPrecondition

import time
time_startpre= time.time()

def unseenPrediction(model, d_test):
    #model = keras.models.load_model('MNISTExample/mnist_model.h5')
    model = model
    d_test = d_test
    #inferDataPrecondition.inferDataPrecondition(model, d_test)
    WPdict = inferDataPrecondition.inferDataPrecondition(model, d_test)

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

    #Single Prediction
    print(df)
    # df = df.iloc[1]
    # print(df.iloc[1])

    # rslt_df = df.sample(n = 1)

    print(rslt_df)

    print(rslt_df.size)
    #drop last column
    # rslt_df = rslt_df.iloc[:,:-1]
    # actual = df[df.columns[-1]]

    print(rslt_df)


    ActualOutcome = actual


    print(rslt_df)

    for key, value in WPdict.items():
        print(key, '->', value)
        rslt_df[key] = rslt_df[key].map(lambda x: "Y" if x >= value else "N")
         #print(key, '->', value)
        print(rslt_df)
        #print(rslt_df.key.str.count("Y").sum())


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

    #Check important features
    print("WP_violation_count")
    print(WP_violation_count)
    print("WP_satisfied_count")
    print(WP_satisfied_count)

    #important feature detection
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
    #WP_df["DeepInfer_Implication"] = WP_df.apply(lambda x: "Wrong" if (x['total_Viol'] <= violationMean) else "Correct", axis=1)
    print("More_Important_features: ")
    print(important_features)
    print("Less_Important_features: ")
    print(unimportant_featues)
    # important_features = ['Glucose', 'Age' ,'BMI', 'BloodPressure']
    # unimportant_featues = ['Pregnancies','SkinThickness','Insulin','DiabetesPedigreeFunction']
    WP_df["vCount_MoreImpFeat"] = 0
    WP_df["vCount_LessImpFeat"] = 0
    for i in  important_features:
        WP_df["vCount_MoreImpFeat"] = WP_df["vCount_MoreImpFeat"]+ WP_df[i].str.contains('N', regex=False).astype(int)
    for i in  unimportant_featues:
        WP_df["vCount_LessImpFeat"] = WP_df["vCount_LessImpFeat"]+ WP_df[i].str.contains('N', regex=False).astype(int)

    print(WP_df)
    WP_df["DeepInfer_Implication"] = WP_df.apply(lambda x: "Correct" if (x['vCount_MoreImpFeat'] == 0) else "Uncertain", axis=1)
    WP_df["DeepInfer_Implication"] = WP_df.apply(lambda x: "Wrong" if (x['vCount_LessImpFeat'] > violationMean or x['vCount_MoreImpFeat'] < violationMean  or x['vCount_LessImpFeat'] != x['vCount_MoreImpFeat']) else "Correct", axis=1)
    WP_df["DeepInfer_Implication"] = WP_df.apply(lambda x: "Uncertain" if (x['vCount_LessImpFeat'] == x['vCount_MoreImpFeat'] or x['vCount_LessImpFeat'] == violationMean and x['vCount_MoreImpFeat'] != 0) else x["DeepInfer_Implication"], axis=1)


    #Appending ActualOutcome Column
    print(ActualOutcome)
    WP_df['Actual_Outcome'] = ActualOutcome
    print(WP_df)

    #For PIMA, DIABETES
    predictionvalue = (model.predict(X) > 0.5).astype(int)


    #predictionvalue
    WP_df['Predicted_Outcome'] = predictionvalue
    print(WP_df)


    #WP_df["GroundTruth"] = WP_df[key].map(lambda x: "Y" if x >= value else "N")

    WP_df["GroundTruth"] = WP_df.apply(lambda x: "Correct" if (x['Actual_Outcome'] == x['Predicted_Outcome']) else "Wrong", axis=1)

    print(WP_df)
    #WP_df["FalsePositive"] = WP_df.apply(lambda x: "TP" if (x['GroundTruth'] == x['DeepInfer_Implication']) else "FP", axis=1)
    WP_df["TruePositive"] = WP_df.apply(lambda x: "TP" if (x['GroundTruth'] == 'Correct' and x['DeepInfer_Implication'] == 'Correct') else "Not", axis=1)
    #
    #WP_df["TruePositive"] = WP_df.apply(lambda x: "TP" if (x['GroundTruth'] == x['DeepInfer_Implication']) else "FP", axis=1)
    WP_df["FalsePositive"] = WP_df.apply(lambda x: "FP" if (x['DeepInfer_Implication'] == 'Correct' and x['GroundTruth'] == 'Wrong') else "Not", axis=1)

    WP_df["TrueNegative"] = WP_df.apply(lambda x: "TN" if (x['GroundTruth'] == 'Wrong' and x['DeepInfer_Implication'] == 'Wrong') else "Not", axis=1)

    WP_df["FalseNegative"] = WP_df.apply(lambda x: "FN" if (x['GroundTruth'] == 'Correct' and x['DeepInfer_Implication'] == 'Wrong') else "Not", axis=1)

    print(WP_df)

    WP_df["ActualFalsePositive"] = WP_df.apply(lambda x: "TPAct" if (x['Actual_Outcome'] == x['Predicted_Outcome']) else "FPAct", axis=1)

    FP_count =WP_df["FalsePositive"].str.contains('FP', regex=False).sum().astype(int)
    TP_count =WP_df["TruePositive"].str.contains('TP', regex=False).sum().astype(int)
    FN_count =WP_df["FalseNegative"].str.contains('FN', regex=False).sum().astype(int)
    TN_count =WP_df["TrueNegative"].str.contains('TN', regex=False).sum().astype(int)
    Total_GT_Correct = WP_df["GroundTruth"].str.contains('Correct', regex=False).sum().astype(int)
    Total_GT_Wrong = WP_df["GroundTruth"].str.contains('Wrong', regex=False).sum().astype(int)
    Total_DeepInfer_Implication_Correct = WP_df["DeepInfer_Implication"].str.contains('Correct', regex=False).sum().astype(int)
    Total_DeepInfer_Implication_Wrong = WP_df["DeepInfer_Implication"].str.contains('Wrong', regex=False).sum().astype(int)
    Total_DeepInfer_Implication_Uncertain = WP_df["DeepInfer_Implication"].str.contains('Uncertain', regex=False).sum().astype(int)
    ActFP_count =WP_df["ActualFalsePositive"].str.contains('FPAct', regex=False).sum().astype(int)
    ActTP_count =WP_df["ActualFalsePositive"].str.contains('TPAct', regex=False).sum().astype(int)

    print(WP_df)


    elapsed_timePerm = time.time() - time_startpre

    print(elapsed_timePerm)





