from ipyparallel.datapub import publish_data
import keras
import numpy as np

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []
        
    def on_epoch_end(self, epoch, logs={}):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        publish_data({
            'loss': self.loss, 
            'acc': self.acc, 
        })
        
def build_and_train(x_train, y_train, valid_frac, batch_size, n_epochs, 
                    h1, h2, h3, dropout, optimizer, verbose=0, nthreads=1,
                    checkpoint_file=None, callbacks=[]):
    """Run training for one set of hyper-parameters.
    TODO: add support for model checkpointing."""
    from mnist import build_model
    import keras
    # Build the model
    model = build_model(h1=h1, h2=h2, h3=h3,
                        dropout=dropout, optimizer=optimizer)
    
    if checkpoint_file is not None:
        callbacks.append(keras.callbacks.ModelCheckpoint(checkpoint_file))
        
    loss_history = LossHistory()
    callbacks.append(loss_history)
    
    # Train the model
    history = model.fit(x_train, y_train,
                        validation_split=valid_frac,
                        batch_size=batch_size, epochs=n_epochs,
                        verbose=verbose, callbacks=callbacks)
    return history.history