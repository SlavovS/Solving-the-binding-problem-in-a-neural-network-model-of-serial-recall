import keras.backend as K
from keras.callbacks import Callback, warnings
from keras.losses import categorical_crossentropy

from params import Params

# This acc function takes into account the output during recall only
def total_accuracy(y_true, y_pred):
    return K.min(
                K.cast(
                    K.equal(
                            K.argmax(y_true, axis = 2), 
                            K.argmax(y_pred, axis = 2)
                    ), dtype="float32"
                )
            )

                    
# Stopping training after specific validation accuracy is reached
class EarlyStoppingByAccuracy(Callback):
    def __init__(self, stop_crit, measure, verbose=0):
        super(Callback, self).__init__()
        self.monitor = measure
        self.value = stop_crit
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            return warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
            

def dual_train_loss(y_true, y_pred):
    return categorical_crossentropy(
                y_true[:, :,:Params.n_dual_train_items], 
                y_pred[:, :,:Params.n_dual_train_items]
            )
