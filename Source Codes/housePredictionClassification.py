import pandas as pd
import numpy as np
df = pd.read_csv('housepricedata.csv')

features = list(df.columns.values)
features.remove('BedroomAbvGr')
print(features)

dataset = df.values
dataset
X_ = dataset[:,0:10]
Y = dataset[:,10]
#preprocessing

print(df)
features = list(df.columns.values)
print(features)

import xgboost
#
#train an XGBoost model
X, y = X_,Y
model = xgboost.XGBRegressor().fit(X, y)

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
X_scale

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_scale, Y, test_size=0.2)

#model
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

model = Sequential([
    Dense(12, activation='relu', input_shape=(10,)),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid'),
])

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['acc'])


hist = model.fit(X_train, Y_train,
          batch_size=32, epochs=100,
          validation_data=(X_test, Y_test))

#model.save('HP1.h5')

score = model.evaluate(X_test, Y_test, verbose=0)

print("Test loss:", score[0])
print("Test accuracy:", score[1])

predictions = (model.predict(X) > 0.5).astype(int)


import matplotlib.pyplot as plt

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
#plt.show()

plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')