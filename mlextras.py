import os

import keras
import tensorflow as tf
from mnist import build_model
from ipyparallel.datapub import publish_data

class IPyParallelLogger(keras.callbacks.Callback):
    def __init__(self):
        super(IPyParallelLogger, self).__init__()
        self.history = {}

    def on_train_begin(self, logs):
        self.history = {
            "acc": [],
            "loss": [],
            "val_acc": [],
            "val_loss": [],
            "epoch": []
        }
        publish_data({"status": "Begin Training", "history": self.history})

    def on_train_end(self, logs):
        publish_data({"status": "Ended Training", "history": self.history})

    def on_epoch_begin(self, epoch, logs):
        publish_data({"status": "Begin Epoch", "epoch": epoch, "history": self.history})

    def on_epoch_end(self, epoch, logs):
        for k in logs:
            self.history[k].append(logs[k])
        self.history["epoch"].append(epoch)
        publish_data({"status": "Ended Epoch", "epoch": epoch, "history": self.history})

def configure_session():
    """Make a TF session configuration with appropriate thread settings"""
    n_inter_threads = int(os.environ.get('NUM_INTER_THREADS', 2))
    n_intra_threads = int(os.environ.get('NUM_INTRA_THREADS', 32))
    config = tf.ConfigProto(
        inter_op_parallelism_threads=n_inter_threads,
        intra_op_parallelism_threads=n_intra_threads
    )
    return tf.Session(config=config)


def build_and_train(x_train, y_train, valid_frac, batch_size, n_epochs,
                    h1, h2, h3, dropout, optimizer, verbose=0, nthreads=1,
                    checkpoint_file=None, callbacks=[]):
    """Run training for one set of hyper-parameters."""

    # Build the model
    model = build_model(h1=h1, h2=h2, h3=h3,
                        dropout=dropout, optimizer=optimizer)

    ipp_logger = IPyParallelLogger()
    callbacks.append(ipp_logger)

    if checkpoint_file is not None:
        callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_file))

    # Train the model
    history = model.fit(x_train, y_train,
                        validation_split=valid_frac,
                        batch_size=batch_size, epochs=n_epochs,
                        verbose=verbose, callbacks=callbacks)
    return history.history
