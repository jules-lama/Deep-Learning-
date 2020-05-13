# -*- coding: utf-8 -*-
"""
@author: Rabah Oumessaoud
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

tensorboard_callback = keras.callbacks.TensorBoard(log_dir= os.path.join('linear', 'simple'),
                                                   histogram_freq=0,  
                                                   write_graph=True,
                                                   write_images=True)

#bulding a simple model
model = keras.Sequential()
model.add(keras.layers.Dense(3072, activation = 'linear'))
model.add(keras.layers.Dense(1))
model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

#one hot encoding 
labels_data_train = keras.utils.to_categorical(train_labels)
labels_data_test = keras.utils.to_categorical(test_labels)

history = model.fit(np.array(train_data), np.asarray(labels_data_train),
          batch_size= 128,
          epochs=10,
          validation_data=(np.array(test_data), np.asarray(labels_data_test)),
          callbacks = [tensorboard_callback])

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