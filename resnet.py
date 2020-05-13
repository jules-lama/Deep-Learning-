# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 20:47:45 2020

@author: RabahPGM
"""
#handling the imports
import numpy as np
from collections import Counter
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import load_data
import matplotlib.pyplot as plt
import datetime
import os

#importing data
train_data, train_labels = load_data.get_data_and_labels()
test_data, test_labels = load_data.get_data_and_labels_test()

train_data = [np.reshape(dat, (3,32,32)).transpose(1,2,0) for dat in train_data]
test_data = [np.reshape(dat, (3,32,32)).transpose(1,2,0) for dat in test_data]

path = os.path.join('resnet', 'v1', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = keras.callbacks.TensorBoard(log_dir= path,
                                                   histogram_freq=0,  
                                                   write_graph=True,
                                                   write_images=True)

def res_net_block(input_data, filters, conv_size):
  x = keras.layers.Conv2D(filters, conv_size, activation='relu', padding='same', 
                          kernel_regularizer = keras.regularizers.l2(0.001))(input_data)
  x = keras.layers.Dropout(0.4)(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Conv2D(filters, conv_size, activation='relu', padding='same',
                          kernel_regularizer = keras.regularizers.l2(0.001))(x)
  x = keras.layers.Dropout(0.4)(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.Add()([x, input_data])
  x = keras.layers.Activation('relu')(x)
  return x

inputs = keras.Input(shape=(32, 32, 3))

x = keras.layers.Conv2D(32, 3, activation='relu')(inputs)
x = keras.layers.Dropout(0.4)(x)
x = keras.layers.Conv2D(64, 3, activation='relu')(x)
x = keras.layers.Dropout(0.4)(x)
x = keras.layers.MaxPooling2D(pool_size = (2,2))(x)


for i in range(3):
    x = res_net_block(x, 64, 3)

x = keras.layers.Conv2D(128, 3, activation='relu')(x)
x = keras.layers.Dropout(0.4)(x)
x = keras.layers.MaxPooling2D(pool_size = (2,2))(x)

for i in range(3):
    x = res_net_block(x, 128, 3)

x = keras.layers.Conv2D(256, 3, activation='relu')(x)
x = keras.layers.Dropout(0.4)(x)
x = keras.layers.MaxPooling2D(pool_size = (2,2))(x)

for i in range(3):
    x = res_net_block(x, 256, 3)

x = keras.layers.Flatten()(x)
x = keras.layers.Dense(1024, activation='relu',
                       kernel_regularizer = keras.regularizers.l2(0.001))(x)
x = keras.layers.Dropout(0.4)(x)
outputs = keras.layers.Dense(10, activation='softmax')(x)

model = keras.Model(inputs, outputs)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


model.summary()

#one hot encoding 
labels_data_train = keras.utils.to_categorical(train_labels)
labels_data_test = keras.utils.to_categorical(test_labels)


history = model.fit(np.asarray(train_data), np.asarray(labels_data_train),
          batch_size= 128,
          epochs = 100,
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