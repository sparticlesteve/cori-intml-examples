"""
This module contains model and training code for the MNIST classifier.
"""

# System
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

# Data libraries
import numpy as np
from keras.datasets import mnist

# Deep learning
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras import backend as K
import tensorflow as tf


# Data config
n_classes = 10
img_rows, img_cols = 28, 28

# Force format on import
K.set_image_data_format('channels_last')

def load_data():
    # Read the data
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Reformat and scale the data
    input_shape = (img_rows, img_cols, 1)
    x_train = x_train.reshape(x_train.shape[0], *input_shape).astype(np.float32) / 255
    x_test = x_test.reshape(x_test.shape[0], *input_shape).astype(np.float32) / 255
    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_test = keras.utils.to_categorical(y_test, n_classes)
    return x_train, y_train, x_test, y_test

def build_model(h1=4, h2=8, h3=32, dropout=0.5,
                optimizer='Adadelta'):
    """Construct our Keras model"""
    input_shape = (img_rows, img_cols, 1)
    model = Sequential()
    model.add(Conv2D(h1, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(h2, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(h3, activation='relu'))
    model.add(Dropout(dropout))
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(optimizer=optimizer, loss=categorical_crossentropy,
                  metrics=['accuracy'])
    return model
