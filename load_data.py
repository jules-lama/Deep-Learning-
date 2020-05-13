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

file = "cifar-10-python/cifar-10-batches-py/"

#function made by toronto edu. It helps "unpickle" the data
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

#image labels
def load_label_names():
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            


def get_data_and_labels():
    data = []
    labels = []
    for i in range(1,6):
        batch = unpickle(file + "data_batch_" + str(i))
        for key in batch.keys():
            if str(key) == "b'data'":
                data.extend((batch[key]))
            if str(key) == "b'labels'":
                labels.extend((batch[key]))
    return(data, labels)

def get_data_and_labels_test():
    data = []
    labels = []
    batch = unpickle(file + "test_batch")
    for key in batch.keys():
        if str(key) == "b'data'":
            data.extend(batch[key])
        if str(key) == "b'labels'":
            labels.extend(batch[key])
    return (data, labels)

d = get_data_and_labels()
d_test = get_data_and_labels_test()