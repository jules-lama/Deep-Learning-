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
import load_data
import matplotlib.pyplot as plt
import os

#importing data
train_data, train_labels = load_data.get_data_and_labels()
test_data, test_labels = load_data.get_data_and_labels_test()

train_data = [np.reshape(dat, (3,32,32)).transpose(1,2,0) for dat in train_data]
test_data = [np.reshape(dat, (3,32,32)).transpose(1,2,0) for dat in test_data]

#bulding a simple model
model = keras.Sequential()
model.add(keras.layers.Conv2D(input_shape = (32,32,3), kernel_size = (3,3), filters = 32))
model.add(keras.layers.MaxPooling2D(pool_size = (2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1040, activation = 'relu', use_bias = True))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(1040, activation = 'relu', use_bias = True, 
                             kernel_regularizer = keras.regularizers.l2(0.1)))
model.add(keras.layers.Dense(1040, activation = 'relu', use_bias = True, 
                             kernel_regularizer = keras.regularizers.l2(0.1)))
model.add(keras.layers.Dropout(0.1))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(lr = 0.1, loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.summary()

#one hot encoding 
labels_data_train = keras.utils.to_categorical(train_labels)
labels_data_test = keras.utils.to_categorical(test_labels)

#tensorboard
tensorboard_callback = keras.callbacks.TensorBoard(log_dir = os.path.join('cnn', 'shallow'), 
                                                   histogram_freq=0,  
                                                   write_graph=True, 
                                                   write_images=True)

history = model.fit(np.asarray(train_data), np.asarray(labels_data_train),
          batch_size= 128,
          epochs = 30,
          validation_data=(np.asarray(test_data), np.asarray(labels_data_test)),
           callbacks=[tensorboard_callback])

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()