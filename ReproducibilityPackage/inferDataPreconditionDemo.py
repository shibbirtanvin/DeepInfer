from tensorflow import keras
import numpy as np
import pandas as pd
import csv
#load model
model = keras.models.load_model('PD1.h5')
# model = keras.models.load_model('Models/PD2.h5')
# model = keras.models.load_model('Models/PD3.h5')
# model = keras.models.load_model('Models/PD4.h5')
n = [0.75, 0.90, 0.95, 0.99]
cond = ['>=','<=','>','<','==','!=']
l = len(model.layers)
Q=[] #postconditions

#print(l)

# Getting Weights and Biases for each layer and Compute Fiction variable \gamma
w = np.array([])
Weight = [] #Storing Weights
Bias = [] #Storing Biases
Gamma = [] #Storing Weights
N = [] #Activation function list
Gamma_tr =[]
X =[]
# if 'dense' in layer.name or 'Dense' in str(model.layers[i]):
#    print("For dense based network")

for i in range(0,l):
    #print(i)
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

# print(len(Weight))
# print(len(Bias))
# print(len(N))
# print(len(Gamma))
# for i in range(len(Weight)):
#     print("W_",i+1,":", Weight[i])
#
# for i in range(len(Bias)):
#     print("B_",i+1,":", Bias[i])
#
# for i in range(len(N)):
#     print("Activation function of layer_a", i + 1, ":", N[i])
#
# for i in range(len(Gamma)):
#     print("Gamma_", i + 1, ":", Gamma[i])

for i in range(l):
    for j in range(len(n)):
        for k in range(len(cond)):
            M = np.matmul((Gamma_tr[i]*n[j]), -(Bias[i]))
            #print("X_", i + 1, "postcondition:",cond[k], n[j])
            #print(M)

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
            #print("X_", i + 1, "postcondition:", cond[0], Q)
            #print(M1)
            #print(M2)
            M = M2

        if (N0 == 'sigmoid'):
            M = np.matmul((Gamma_tr[i] * np.log(Q / (1 - Q))), -(Bias[i]))
            #print("X_", i + 1, "postcondition:", cond[0], Q)
            #print(M)

        if (N0 == 'tanh'):
            n_tanh = abs((Q - 1) / (Q + 1))
            M = np.matmul((Gamma_tr[i] * (0.5 * np.log(n_tanh))), -(Bias[i]))
            #print("X_", i + 1, "postcondition:", cond[0], Q)
            #print(M)

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
#print(Q)
i=l-1
wp = beta(N,Q,l,i)
#print("wp: ")
#print(wp)

#load dataset pima unseen test dataset
d_test = 'PD_unseen.csv'

file = d_test

with open(file, mode='r', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter=',')
    names =reader.fieldnames
    #print(reader.fieldnames)
    names = names[:-1]
    # df.iloc[: , 1:]
    #names = names[:,1:]
    #del names[0]
    features = np.array([])
    # Add / Append an element at the end of a numpy array
    for i in names:
        features = np.append(features, i)

    #print('Numpy Array: ', features)
    #print('feature length: ', features.size)
    #print('WP size ', wp.size)
if(wp.size == features.size):
    #print("True")
    WPdictionary = dict(zip(features, wp))
    #print(WPdictionary)
    #for key, value in WPdictionary.items():
        #print(key)
        #print(key, '>', "{0:.2f}".format(value))
else:
    feature_counter = 1
    feature_count = np.array([])
    for i in wp:
        feature_count = np.append(feature_count,feature_counter)
        feature_counter = feature_counter + 1
    WPdictionary = dict(zip(feature_count, wp))
    #print(WPdictionary)

# for key, value in WPdictionary.items():
#     print(key, '>=', value)
