from keras.optimizers import Adam

from params import Params
from trials import make_token_sr_trial, make_item_sr_trial

from misc import total_accuracy, EarlyStoppingByAccuracy

import models

def examples_generator(trial_func):
    while(True):
        for list_len in range(1, Params.max_list_len + 1):
            yield trial_func(list_len)

def examples_generator_val(trial_func):
    while(True):
        yield trial_func(Params.test_list_len)
#  
#print("Training item SR model...")                  
#sr_item_model = models.create_item_SR_model()
#sr_item_model.compile(
#        loss="categorical_crossentropy", 
#        optimizer= Adam(lr = 0.001), 
#        metrics=['accuracy', total_accuracy]
#    )    
#es =  EarlyStoppingByAccuracy(
#        stop_crit = Params.sr_stop_crit,
#        measure='val_total_accuracy', 
#        verbose=1
#    )
#
#sr_item_model.fit_generator(        
#        examples_generator(make_item_sr_trial),
#        epochs=1000,
#        steps_per_epoch=1000,
#        verbose=1, 
#        validation_data=examples_generator_val(make_item_sr_trial),
#        validation_steps=300,       
#        callbacks=[es]
#    )
#
#
#sr_item_model.save_weights("item-sr.weights.h5")            

print("Training token SR model...")                  
sr_token_model = models.create_token_SR_model()
sr_token_model.compile(
        loss="categorical_crossentropy", 
        optimizer= Adam(lr = 0.01), 
        metrics=['accuracy', total_accuracy]
    )    
es =  EarlyStoppingByAccuracy(
        stop_crit = Params.sr_stop_crit,
        measure='val_total_accuracy', 
        verbose=1
    )

sr_token_model.fit_generator(        
        examples_generator(make_token_sr_trial),
        epochs=1000,
        steps_per_epoch=1000,
        verbose=1, 
        validation_data=examples_generator_val(make_token_sr_trial),
        validation_steps=300,       
        callbacks=[es]
    )


sr_token_model.save_weights("token-sr.weights.h5")            