import os

from ipyparallel.datapub import publish_data
import mnist
import keras
import tensorflow as tf

class IPyParallelLogger(keras.callbacks.Callback):
    """Keras callback that publishes data to IPyParallel engines"""
    def __init__(self):
        super(IPyParallelLogger, self).__init__()
        self.history = {}

    def on_train_begin(self, logs={}):
        self.history = {
            "loss": [],
            "acc": [],
            "val_loss": [],
            "val_acc": [],
            "logs": [],
        }
        if logs:
            self.history["logs"].append(logs)
        publish_data({"logs": logs, "status": "Begin Training", "history": self.history})

    def on_train_end(self, logs={}):
        if logs:
            self.history["logs"].append(logs)
        publish_data({"logs": logs, "status": "Ended Training", "history": self.history})
        
    def on_epoch_begin(self, epoch, logs={}):
        if logs:
            self.history["logs"].append(logs)
        publish_data({"logs": logs, "epoch": epoch, "status": "Begin Epoch", "history": self.history})

    def on_epoch_end(self, epoch, logs={}):
        if logs:
            self.history["logs"].append(logs)
            self.history["loss"].append(logs["loss"])
            self.history["val_loss"].append(logs["val_loss"])
            self.history["acc"].append(logs["acc"])
            self.history["val_acc"].append(logs["val_acc"])
        publish_data({"logs": logs, "epoch": epoch, "status": "Ended Epoch", "history": self.history})


def configure_session():
    """Make a TF session configuration with appropriate thread settings"""
    n_inter_threads = int(os.environ.get('NUM_INTER_THREADS', 2))
    n_intra_threads = int(os.environ.get('NUM_INTRA_THREADS', 32))
    config = tf.ConfigProto(
        inter_op_parallelism_threads=n_inter_threads,
        intra_op_parallelism_threads=n_intra_threads
    )
    return tf.Session(config=config)
