
# first neural network with keras make predictions
import numpy
from numpy import loadtxt, math
from keras.models import Sequential
from keras.layers import Dense
import math
from tensorflow import keras
import pandas as pd
import numpy as np
# load the dataset
from scipy.stats import nbinom

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import regularizers
from keras import optimizers
from keras.layers import Dense, Dropout, Activation
import inferDataPrecondition


import time
time_startpre= time.time()


df = loadtxt('PIMADataset/pima-indians-diabetes.csv', delimiter=',')
X = df[:,0:8]
y = df[:,8]
print(X.shape)
print(y.shape)


# X = df_new.drop('y', axis=1)
# Y = df_new['y']

#train-test split
x_train, x_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=10)




model = Sequential()
model.add(Dense(12, input_dim=8, activation='linear'))
model.add(Dense(8, activation='linear'))
#model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10, verbose=0, validation_split=0.2)
# make class predictions with the model
# access validation accuracy for each epoch


acc = model.history.history['accuracy']
print(acc) # [0.9573, 0.9696, 0.9754, 0.9762, 0.9784]


score = model.evaluate(X, y, verbose=0)


np.savetxt("PD_xtrain.csv", x_train, delimiter=",")
np.savetxt("PD_ytrain.csv", y_train, delimiter=",")
np.savetxt("PD_xval.csv", x_test, delimiter=",")
np.savetxt("PD_yval.csv", y_test, delimiter=",")

model = keras.models.load_model('pima_model4.h5')
d_test = 'PIMADataset/pimatest_20.csv'
inferDataPrecondition.inferDataPrecondition(model, d_test)

elapsed_timePerm = time.time() - time_startpre

print(elapsed_timePerm)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


#model.save('pima.h5')
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(100):
    #X[i]
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))
