import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import shap
from tensorflow import keras

def normalize(df):
    result = df.copy()
    max_value = df.max()
    min_value = df.min()
    result = (df - min_value) / (max_value - min_value)
    return result

from pandas.api.types import is_string_dtype

data = pd.read_csv('german_credit_data.csv',index_col=0,sep=',')
labels = data.columns

print(labels)

# lets go through column 2 column
for col in labels:
    if is_string_dtype(data[col]):
        if col == 'Risk':
            # we want 'Risk' to be a binary variable
            data[col] = pd.factorize(data[col])[0]
            continue
        # the other categorical columns should be one-hot encoded
        data = pd.concat([data, pd.get_dummies(data[col], prefix=col)], axis=1)

        data.drop(col, axis=1, inplace=True)
    else:
        data[col] = normalize(data[col])

print(data)
data = data.drop(labels=['Saving accounts_rich'], axis=1)
print(data)

data_train = data.iloc[:800]
data_valid = data.iloc[800:]
x_train = data_train.iloc[:,:-1]
y_train = data_train.iloc[:,-1]
x_val = data_valid.iloc[:,:-1]
y_val = data_valid.iloc[:,-1]

from keras.models import Sequential
from keras import regularizers
from keras import optimizers
from keras.layers import Dense, Dropout, Activation

sgd = optimizers.SGD(lr=0.03, decay=0, momentum=0.9, nesterov=False)

model = Sequential()
model.add(Dense(units=50, activation='relu', input_dim=22, kernel_initializer='glorot_normal', bias_initializer='zeros'))#, kernel_regularizer=regularizers.l2(0.01)))

model.add(Dense(units=1, activation='sigmoid', kernel_initializer='glorot_normal', bias_initializer='zeros'))

model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train.values, y_train.values, validation_data=(x_val.values, y_val.values), epochs=30, batch_size=128)

#model.save('GC1.h5')
# evaluate the model
loss, acc = model.evaluate(x_val, y_val, verbose=0)
print('Test Loss: %.3f' % loss)
print('Test Accuracy: %.3f' % acc)
