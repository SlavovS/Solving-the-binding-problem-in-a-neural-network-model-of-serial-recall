import numpy as np

import models
from params import Params

token_perm = []
def gen_token():
    global token_perm
    
    if len(token_perm) == 0:
        token_perm = np.random.permutation(Params.n_tokens)
        
    token = token_perm[0]
    token_perm = token_perm[1:]
    
    return token

def make_trial(list_len):
    trial = {}
            
    #list_len = Params.n_tokens
    trial_input_token = np.zeros(shape=(list_len, Params.n_tokens))
    trial_input_item = np.zeros(shape=(list_len, Params.n_items))
    trial_input_recall_cue = np.zeros(shape=(list_len, 1))
    trial_output_item = np.zeros(shape=(list_len, Params.n_items))
    #letters = np.random.permutation(Params.n_tokens)
    
    token_i = np.random.randint(0, Params.n_tokens) # Index of the token 
    item_i = np.random.randint(0, Params.n_tokens)
    
    trial_input_item[0][item_i] = 1
    trial_output_item[0][item_i] = 1
    trial_input_token[0][token_i] = 1
    trial_input_recall_cue[0][0] = 0
    
    trial[token_i] = item_i
    for i in range(1, list_len):
        if i >= list_len / 2:
            # Recall
            rand_token = np.random.choice(list(trial.keys()))
            trial_output_item[i, trial[rand_token]] = 1
            trial_input_token[i, rand_token] = 1
            trial_input_recall_cue[i, 0] = 1 # This is the recall cue
                                                    # for this specific token
            
            
        else:
            # Encoding
            
            token_i = gen_token()
            item_i = np.random.randint(0, Params.n_tokens)
            trial_input_item[i][item_i] = 1
            trial_output_item[i][item_i] = 1
            trial_input_token[i][token_i] = 1
            #recal cue set to 0 for encoding
            trial_input_recall_cue[i, 0] = 0
#            trial_input_token[i, Params.n_tokens] = 0 # Encoding cue
            trial[token_i] = item_i
        
    return (    
                [
                    trial_input_item.reshape(1, list_len, Params.n_items), 
                    trial_input_token.reshape(1, list_len, Params.n_tokens),
                    trial_input_recall_cue.reshape(1, list_len, 1)
                ],
                trial_output_item.reshape(1, list_len, Params.n_items)
            )
    
bp_model = models.create_BP_model()
bp_model.load_weights("bp.weights.h5")

results = np.zeros(shape=Params.test_list_len * 2)

for i in range(100):
    trial = make_trial(Params.test_list_len * 2)    
#    print(trial[0][0])
#    print(trial[0][1])
#    print(trial[0][2])
#    exit(0)
    target = trial[1][0]
    prediction = bp_model.predict(trial[0])[0]
    for l in range(Params.test_list_len * 2):
       results[l] += np.argmax(prediction[l]) == np.argmax(target[l])

print(results)
