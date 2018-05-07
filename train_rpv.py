"""
Distributed batch training of ATLAS RPV CNN Classifier
"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import socket

import keras
import horovod.keras as hvd

from rpv import load_file, build_model, train_model

print('Distributed RPV classifier training')

# Initialize horovod
hvd.init()
print('MPI rank %i, local rank %i, host %s' %
      (hvd.rank(), hvd.local_rank(), socket.gethostname()))

# Data config
n_train = 32000 #412416
n_valid = 16000 #137471
n_test = 16000 #137471
input_dir = '/global/cscratch1/sd/sfarrell/atlas-rpv-images'

# Load the data files
train_file = os.path.join(input_dir, 'train.h5')
valid_file = os.path.join(input_dir, 'val.h5')
test_file = os.path.join(input_dir, 'test.h5')
train_input, train_labels, train_weights = load_file(train_file, n_train)
valid_input, valid_labels, valid_weights = load_file(valid_file, n_valid)
test_input, test_labels, test_weights = load_file(test_file, n_test)
print('train shape:', train_input.shape)
print('valid shape:', valid_input.shape)
print('test shape: ', test_input.shape)

# Model config
h1, h2, h3, h4, h5 = 64, 128, 256, 256, 512
optimizer = keras.optimizers.Adam
#lr = 0.1 * hvd.size()
lr = 0.001 * hvd.size()

# Training config
batch_size = 128
n_epochs = 8
use_weights = False

# Build the model
model = build_model(train_input.shape[1:],
                    h1=h1, h2=h2, h3=h3, h4=h4, h5=h5,
                    optimizer=optimizer, lr=lr)
if hvd.rank() == 0:
    model.summary()

# Train the model
print('Begin training')
history = train_model(model, train_input=train_input, train_labels=train_labels,
                      valid_input=valid_input, valid_labels=valid_labels,
                      batch_size=batch_size, n_epochs=n_epochs)

# Evaluate the final
if hvd.rank() == 0:
    score = model.evaluate(test_input, test_labels, verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
