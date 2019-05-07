# Simulation 3

import numpy as np

import models
from keras.models import load_model
from params import Params
from trials import make_dual_trial

bp_model = models.create_BP_model()
sr_model = models.create_token_SR_model()

bp_model.load_weights("bp.weights.h5")
sr_model.load_weights("token-sr.weights.h5")

dual_model = models.create_dual_model(bp_model, sr_model)
#dual_model.load_weights("dual.weights.h5")

for tls in range(1, 10):
    Params.test_list_len = tls
    Params.n_dual_train_items = Params.n_items
    
    sr_position = np.zeros(shape=Params.test_list_len)
    position = np.zeros(shape=Params.test_list_len)
    bp_position = np.zeros(shape=Params.test_list_len)
    
    output_tokens = {}
    
    for _ in range(1000):    
        
        dual_trial = make_dual_trial(Params.test_list_len)[0]
        dual_prediction = dual_model.predict(dual_trial)
        dual_target = dual_trial[0][0][:Params.test_list_len]
    
        sr_trial = [dual_trial[1], dual_trial[2]]
        sr_prediction = sr_model.predict(sr_trial)
        sr_target = sr_trial[0][0]
        
        bp_trial = [
                dual_trial[0], 
                np.concatenate(
                        (
                            dual_trial[1][0][:Params.test_list_len], 
                            dual_trial[1][0][:Params.test_list_len])
                        ).reshape(
                                1, 
                                Params.test_list_len * 2, 
                                Params.n_tokens
                            ), 
                dual_trial[2]
            ]
        bp_prediction = bp_model.predict(bp_trial)
        bp_target = bp_trial[0][0][:Params.test_list_len]
        
    
      
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
    
            #ACC of dual model
            if (
                    np.argmax(dual_prediction[0][pos + Params.test_list_len]) 
                    == np.argmax(dual_target[pos])
                ):
                position[pos] += 1 
    
            output_token = np.argmax(sr_prediction[0][pos + Params.test_list_len])
            
            if output_token not in output_tokens:
                output_tokens[output_token] = 0
                
            output_tokens[output_token] += 1
    
    #        else:
    #            print(np.argmax(sr_target, axis = -1))
    #            print(np.argmax(sr_prediction[0], axis = -1))
    #            print(np.max(sr_prediction[0], axis = -1))
    #            exit(0)
                
                
                
#    print(sr_position / 1000)            
    #print(bp_position / 1000)
    print(position / 1000)
#    
    print(output_tokens)
