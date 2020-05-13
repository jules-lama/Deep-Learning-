# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 14:40:40 2019

@author: Rabah Oumessaoud
"""

#handling the imports
import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

file = "cifar-10-python/cifar-10-batches-py/"

#function made by toronto edu. It helps "unpickle" the data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#image labels
def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

#the dataset is composed by 5 main data batches, a test batch, and the metadata
#here, I will analyse one of the data bachtes
batch = unpickle(file + "data_batch_1")
label_names = load_label_names()

data = []
for key in batch.keys():
    if str(key) == "b'data'":
        print(key)
        data = batch[key]
        
lable_index = []
for key in batch.keys():
    if str(key) == "b'labels'":
        print(key)
        lable_index = batch[key]


#data visualization
#let's check the data shape

print("Shape of image data" + str(data.shape))

#Funny, the images are kind of flattened in the vector... the guy who did that must be a psycho
#Guess what? we'll have to reshape each line to get the images
imgs = [np.reshape(data[i], (3,32,32)).transpose(1,2,0) for i in range(10)]

fig = plt.figure(figsize = (32,32))
columns = 2
rows = 5

ax = []
for i in range(columns*rows):
    img = imgs[i]
    ax.append( fig.add_subplot(rows, columns, i+1) )
    ax[-1].set_title(label_names[lable_index[i]])  
    plt.imshow(img, alpha=0.25)
