import numpy as np

import models
from params import Params

def make_input_trial(list_len):
    trial_input_token = np.zeros(shape=(list_len * 2, Params.n_tokens))
    trial_input_recall_cue = np.zeros(shape=(list_len * 2, 1))
    trial_input_item = np.zeros(shape=(list_len * 2, Params.n_items))
    
    tokens = np.random.permutation(Params.n_tokens)
    items = np.random.permutation(Params.n_items)
    
    for i in range(list_len):
        #encoding
        #token
        trial_input_token[i, tokens[i]] = 1
        #recall cue
        trial_input_recall_cue[i + list_len, 0] = 1
        #item
        trial_input_item[i, items[i]] = 1        
    
    return [
            trial_input_token.reshape(1, list_len * 2, Params.n_tokens),
            trial_input_item.reshape(1, list_len * 2, Params.n_items),
            trial_input_recall_cue.reshape(1, list_len * 2, 1)
            ]
    
bp_model = models.create_BP_model()
sr_model = models.create_SR_model()

bp_model.load_weights("bp.weights.h5")
sr_model.load_weights("sr.weights.h5")

dual_model = models.create_DUAL_model(bp_model, sr_model)

Params.test_list_len = 6

sr_position = np.zeros(shape=Params.test_list_len)
position = np.zeros(shape=Params.test_list_len)
bp_position = np.zeros(shape=Params.test_list_len)



for _ in range(1, 100):    
    trial = make_input_trial(Params.test_list_len)
    
    sr_trial = [trial[0], trial[2]]
    bp_trial = [
            trial[1], 
            np.concatenate(
                    (
                        trial[0][0][:Params.test_list_len], 
                        trial[0][0][:Params.test_list_len])
                    ).reshape(
                            1, 
                            Params.test_list_len * 2, 
                            Params.n_tokens
                        ), 
            trial[2]
        ]

    sr_prediction = sr_model.predict(sr_trial)
    sr_target = sr_trial[0][0]
    
    
    bp_prediction = bp_model.predict(bp_trial)
    bp_target = bp_trial[0][0][:Params.test_list_len]
    
    dual_trial = [trial[1], trial[0], trial[2]]
    
    dual_prediction = dual_model.predict(dual_trial)
    dual_target = bp_target
    
  
    for pos in range(0, Params.test_list_len):
        
        #compute ACC of the SR model
        if (
                np.argmax(sr_prediction[0][pos + Params.test_list_len,:]) 
                == np.argmax(sr_target[pos])
            ):
            sr_position[pos] += 1

        #compute ACC of the BP model
        if (
                np.argmax(bp_prediction[0][pos + Params.test_list_len, :]) 
                == np.argmax(bp_target[pos])
            ):
            bp_position[pos] += 1         

        if (
                np.argmax(dual_prediction[0][pos + Params.test_list_len]) 
                == np.argmax(dual_target[pos])
            ):
            position[pos] += 1 
            
            
print(sr_position)            
print(bp_position)
print(position)
