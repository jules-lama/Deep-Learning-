# -*- coding: utf-8 -*-
"""
@author: LAMA GRAIH JULES
"""

#handling the imports
import pickle
import numpy as np
from collections import Counter
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

file = "cifar-10-python/cifar-10-batches-py/"

#function made by toronto edu. It helps "unpickle" the data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#image labels
def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            
batch = unpickle(file + "data_batch_1")
label_names = load_label_names()

data = []
for key in batch.keys():
    if str(key) == "b'data'":
        print(key)
        data = batch[key]

labels = []
for key in batch.keys():
    if str(key) == "b'labels'":
        print(key)
        labels = batch[key]
    
imgs_train = []    
for i in range(len(data)):
    imgs_train.append(np.reshape(data[i], (3,32,32)).transpose(1,2,0))

imgs_train = np.asarray(imgs_train)

test = unpickle(file + 'test_batch')
data_test = []
for key in test.keys():
    if str(key) == "b'data'":
        print(key)
        data_test = test[key]

labels_test = []
for key in test.keys():
    if str(key) == "b'labels'":
        print(key)
        labels_test = test[key]
        
imgs_test = []
for i in range(len(data_test)):
    imgs_test.append(np.reshape(data_test[i], (3,32,32)).transpose(1,2,0))
imgs_test = np.asarray(imgs_test)

#building the model
encoded_train = keras.utils.to_categorical(labels)
encoded_test = keras.utils.to_categorical(labels_test)


model = keras.Sequential()
model.add(keras.layers.Conv2D(input_shape = (32,32,3), kernel_size = (3,3), filters = 32, activation = 'relu'
                              ,padding = 'valid'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Conv2D(kernel_size = (3,3), filters = 64, activation = 'relu'
                              ,padding = 'valid'))
model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(imgs_train, np.asarray(encoded_train),
          batch_size= 64,
          epochs=100,
          validation_data=(imgs_test, np.asarray(encoded_test)))





