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

from rpv import load_dataset, build_model, train_model

print('Distributed RPV classifier training')

# Initialize horovod
hvd.init()
print('MPI rank %i, local rank %i, host %s' %
      (hvd.rank(), hvd.local_rank(), socket.gethostname()))

# Data config
n_train = 64000 #412416
n_valid = 32000 #137471
n_test = 32000 #137471
#input_dir = '/data0/users/sfarrell/atlas-rpv-images'
input_dir = '/global/cscratch1/sd/sfarrell/atlas-rpv-images'

# Load the data files
train_data, valid_data, test_data = load_dataset(input_dir, n_train, n_valid, n_test)
train_input, train_labels, train_weights = train_data
valid_input, valid_labels, valid_weights = valid_data
test_input, test_labels, test_weights = test_data
print('train shape:', train_input.shape, 'Mean label:', train_labels.mean())
print('valid shape:', valid_input.shape, 'Mean label:', valid_labels.mean())
print('test shape: ', test_input.shape, 'Mean label:', test_labels.mean())

# Model config
conv_sizes = [16, 32, 64]
fc_sizes = [128]
optimizer = 'Adam'
lr = 0.001 * hvd.size()
dropout = 0.2

# Training config
batch_size = 128
n_epochs = 4

# Build the model
model = build_model(train_input.shape[1:],
                    conv_sizes=conv_sizes, fc_sizes=fc_sizes,
                    dropout=dropout, optimizer=optimizer, lr=lr,
                    use_horovod=True)
if hvd.rank() == 0:
    model.summary()

# Train the model
print('Begin training')
history = train_model(model, train_input=train_input, train_labels=train_labels,
                      valid_input=valid_input, valid_labels=valid_labels,
                      batch_size=batch_size, n_epochs=n_epochs,
                      verbose=2, use_horovod=True)

# Evaluate the final
if hvd.rank() == 0:
    score = model.evaluate(test_input, test_labels, verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
