from keras.optimizers import Adam
import numpy as np

from params import Params
from misc import EarlyStoppingByAccuracy, total_accuracy
from trials import make_bp_trial

import models

def examples_generator():
    while(True):
        yield make_bp_trial(Params.max_list_len * 2)


bp_model = models.create_BP_model()
bp_model.compile(
        loss="categorical_crossentropy", 
        optimizer= Adam(lr = 0.01), 
        metrics=['accuracy', total_accuracy]
    )

es = EarlyStoppingByAccuracy(
        stop_crit=Params.bp_stop_crit,
        measure='total_accuracy',         
        verbose=1
    )

history = bp_model.fit_generator(        
        examples_generator(),
        epochs=100,
        steps_per_epoch=5000,
        verbose=1, 
        validation_data=examples_generator(),
        validation_steps=1000,        
        callbacks=[]
    )
bp_model.save_weights("bp.weights.h5")
