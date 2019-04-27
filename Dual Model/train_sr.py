from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import Callback, warnings, ModelCheckpoint
from keras.callbacks import CSVLogger

import numpy as np

from params import Params
import models

tests = []

def make_trial(list_len, train_pos = None, train_letter = None):
    global tests
    
    assert(
            ((train_pos is None) and (train_letter is None))
            or
            (not(train_pos is None) and not(train_letter is None))
            )
    
    trial_input = np.zeros(shape=(list_len * 2, Params.n_tokens))
    trial_input_recall_cue = np.zeros(shape=(list_len * 2, 1))

    trial_output = np.zeros(shape=(list_len * 2, Params.n_tokens))
    while True:
        letters = np.random.permutation(Params.n_tokens)
        do_break = True        
        
        if train_pos is None:
            #  training
            for i in range(len(tests)):
                if letters[i] == tests[i]:
                    do_break = False
                    break
        else:
            #test
            if letters[train_pos] == train_letter:
                do_break = False

        if do_break:
            break
            
        
    for i in range(list_len):
        #encoding
        trial_input[i, letters[i]] = 1
        trial_output[i, letters[i]] = 1
        #recall
        #recall cue
        trial_input_recall_cue[i + list_len, 0] = 1
        #output letter
        trial_output[i + list_len, letters[i]] = 1
    
    #recall cue Represented as a separate input for the Dual model to work
    ##trial_input_val[list_len * 2, len(letters_26)] = 1
#        trial_input_recall_cue[0, Params.n_tokens] = 1        
        
    return (
            [trial_input.reshape(1, list_len * 2, Params.n_tokens),
             trial_input_recall_cue.reshape(1, list_len*2, 1)],
            trial_output.reshape(1, list_len *2, Params.n_tokens)
            )


def examples_generator():
    while(True):
        for list_len in range(1, Params.max_list_len + 1):
            yield make_trial(list_len)
  
def examples_generator_val():
    while(True):
        yield make_trial(Params.test_list_len)



# This acc function takes into account the output during recall only
def my_accuracy(y_true, y_pred):
#    print(y_pred)
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
    def __init__(self, monitor='val_my_accuracy', verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = Params.test_list_acc
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            return warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
            
            
            
sr_model = models.create_SR_model()
sr_model.compile(
        loss="categorical_crossentropy", 
        optimizer= Adam(lr = 0.001), 
        metrics=['accuracy', my_accuracy]
    )    
es =  EarlyStoppingByAccuracy(
        monitor='val_my_accuracy', 
        value=Params.test_list_acc, 
        verbose=1
    )
mc = ModelCheckpoint(
        'best_model_sr_checkpoint.h5', 
        monitor='val_my_accuracy', 
        mode='max', 
        verbose=1,
        save_best_only=True
    )

csv_logger = CSVLogger(
        'status_log.csv', 
        append=True, 
        separator=','
    )

history = sr_model.fit_generator(        
        examples_generator(),
        nb_epoch=10000,
        steps_per_epoch = 1000,
        verbose=1, 
        validation_data = examples_generator_val(),
        nb_val_samples = 300,
        
        callbacks = [es, mc, csv_logger]
    )


sr_model.save_weights("sr.weights.h5")            
