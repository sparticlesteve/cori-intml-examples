"""
Distributed batch training of ATLAS RPV CNN Classifier
"""

# System
import socket
import argparse

# Externals
import horovod.keras as hvd

# Locals
from rpv import load_dataset, build_model, train_model

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', default='/global/cscratch1/sd/sfarrell/atlas-rpv-images')
parser.add_argument('--n-train', type=int, default=64000)
parser.add_argument('--n-valid', type=int, default=32000)
parser.add_argument('--n-test', type=int, default=0)
parser.add_argument('--h1', type=int, default=16)
parser.add_argument('--h2', type=int, default=32)
parser.add_argument('--h3', type=int, default=64)
parser.add_argument('--h4', type=int, default=128)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr-scaling', choices=['linear'])
parser.add_argument('--optimizer', default='Adam')
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--n-epochs', type=int, default=4)
parser.add_argument('--fom', choices=['best', 'last'])
args = parser.parse_args()

print('Distributed RPV classifier training')

# Initialize horovod
hvd.init()
print('MPI rank %i, local rank %i, host %s' %
      (hvd.rank(), hvd.local_rank(), socket.gethostname()))

# Load the data files
train_data, valid_data, test_data = load_dataset(
    args.input_dir, args.n_train, args.n_valid, args.n_test)
train_input, train_labels, train_weights = train_data
valid_input, valid_labels, valid_weights = valid_data
test_input, test_labels, test_weights = test_data
print('train shape:', train_input.shape, 'Mean label:', train_labels.mean())
print('valid shape:', valid_input.shape, 'Mean label:', valid_labels.mean())
if args.n_test > 0:
    print('test shape: ', test_input.shape, 'Mean label:', test_labels.mean())

# Model config
conv_sizes = [args.h1, args.h2, args.h3]
fc_sizes = [args.h4]
if args.lr_scaling == 'linear':
    lr = args.lr * hvd.size()
else:
    lr = args.lr

# Build the model
model = build_model(train_input.shape[1:],
                    conv_sizes=conv_sizes, fc_sizes=fc_sizes,
                    dropout=args.dropout, optimizer=args.optimizer, lr=lr,
                    use_horovod=True)
if hvd.rank() == 0:
    model.summary()

# Train the model
print('Begin training')
history = train_model(model, train_input=train_input, train_labels=train_labels,
                      valid_input=valid_input, valid_labels=valid_labels,
                      batch_size=args.batch_size, n_epochs=args.n_epochs,
                      verbose=2, use_horovod=True)

# Print figure of merit for HPO
if args.fom == 'best':
    print('FoM:', min(history.history['val_loss']))
elif args.fom == 'last':
    print('FoM:', history.history['val_loss'][-1])

# Optionally evaluate on the test set
if hvd.rank() == 0 and args.n_test > 0:
    score = model.evaluate(test_input, test_labels, verbose=2)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])