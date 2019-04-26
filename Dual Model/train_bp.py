from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import Callback, warnings, ModelCheckpoint

import numpy as np

from params import Params
import models

token_perm = []
def gen_token():
    global token_perm
    
    if len(token_perm) == 0:
        token_perm = np.random.permutation(Params.n_tokens)
        
    token = token_perm[0]
    token_perm = token_perm[1:]
    
    return token

# Training
def make_trial(list_len):
    trial = {}

    items = np.random.permutation(Params.n_items)
    
    #list_len = Params.n_tokens
    trial_input_token = np.zeros(shape=(list_len, Params.n_tokens))
    trial_input_item = np.zeros(shape=(list_len, Params.n_items))
    trial_input_recall_cue = np.zeros(shape=(list_len, 1))
    trial_output_item = np.zeros(shape=(list_len, Params.n_items))
    #letters = np.random.permutation(Params.n_tokens)
    
    token_i = np.random.randint(0, Params.n_tokens) # Index of the token 
    item_i = np.random.randint(0, Params.n_items)
    
    trial_input_item[0][item_i] = 1
    trial_output_item[0][item_i] = 1
    trial_input_token[0][token_i] = 1
    trial_input_recall_cue[0][0] = 0
    
    trial[token_i] = item_i
    for i in range(1, list_len):
        if np.random.rand() < 0.5:
            # Recall
            rand_token = np.random.choice(list(trial.keys()))
            trial_output_item[i, trial[rand_token]] = 1
            trial_input_token[i, rand_token] = 1
            trial_input_recall_cue[i, 0] = 1 # This is the recall cue
                                                    # for this specific token
            
            
        else:
            # Encoding
            
            token_i = gen_token()
            trial_input_item[i][items[i]] = 1
            trial_output_item[i][item_i] = 1
            trial_input_token[i][token_i] = 1
            #recal cue set to 0 for encoding
            trial_input_recall_cue[i, 0] = 0
            trial[token_i] = items[i]
        
    return (    
                [
                    trial_input_item.reshape(1, list_len, Params.n_items), 
                    trial_input_token.reshape(1, list_len, Params.n_tokens),
                    trial_input_recall_cue.reshape(1, list_len, 1)
                ],
                trial_output_item.reshape(1, list_len, Params.n_items)
            )

def examples_generator():
    while(True):
        yield make_trial(Params.max_list_len * 2)


######################################
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
    def __init__(self, monitor='val_acc', value=0.98, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            return warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current >= self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True

bp_model = models.create_BP_model()
bp_model.compile(
        loss="categorical_crossentropy", 
        optimizer= Adam(lr = 0.01), 
        metrics=['accuracy', my_accuracy]
    )

es = EarlyStoppingByAccuracy(
        monitor='val_my_accuracy', 
        value=Params.bp_acc, 
        verbose=1
    )
mc = ModelCheckpoint(
        'best_model_bp_intermediate.h5', 
        monitor='val_my_accuracy', 
        mode='max', 
        verbose=1,
        save_best_only=True
    )


history = bp_model.fit_generator(        
        examples_generator(),
        nb_epoch=100,
        steps_per_epoch = 5000,
        verbose=1, 
        validation_data = examples_generator(),
        nb_val_samples = 300,        
        callbacks = [es]
    )
bp_model.save_weights("bp.weights.h5")
