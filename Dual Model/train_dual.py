import numpy as np
from keras.optimizers import Adam
import models
from params import Params
from trials import make_dual_trial
from misc import total_accuracy, EarlyStoppingByAccuracy
from misc import dual_train_loss

def examples_generator():
    while(True):        
        for list_len in np.random.permutation(Params.max_list_len):

            yield make_dual_trial(list_len + 1)

def val_examples_generator():
    while(True):
        yield make_dual_trial(Params.test_list_len)



bp_model = models.create_BP_model()
sr_model = models.create_token_SR_model()

bp_model.load_weights("bp.weights.h5")

dual_model = models.create_dual_model(bp_model, sr_model)

dual_model.compile(
        loss=dual_train_loss, 
        optimizer= Adam(lr = 0.001), 
        metrics=['accuracy', total_accuracy]
    )


es = EarlyStoppingByAccuracy(
        stop_crit=Params.sr_stop_crit,
        measure='val_total_accuracy',         
        verbose=1
    )

history = dual_model.fit_generator(        
        examples_generator(),
        epochs=1000,
        steps_per_epoch=5000,
        verbose=1, 
        validation_data=val_examples_generator(),
        validation_steps=1000,        
        callbacks=[es]
    )

dual_model.save_weights("dual.weights.h5")