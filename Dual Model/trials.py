import numpy as np

from params import Params

def generate_sr_items(test = None):
    while True:
        items = np.random.permutation(Params.n_items)
        do_break = True        
        
        if test is None:
            #  training
            for i in range(len(Params.tests)):
                if items[i] == Params.tests[i]:
                    do_break = False
                    break
        else:
            #test
            if items[test[0]] != test[1]:
                do_break = False
            else:
                for i in range(len(Params.tests)):
                    if (items[i] == Params.tests[i]) and (i != test[0]):
                        do_break = False
                    
        if do_break:
            break
        
    return items

def make_item_sr_trial(list_len, test = None):
    trial_input = np.zeros(shape=(list_len * 2, Params.n_items))
    trial_input_recall_cue = np.zeros(shape=(list_len * 2, 1))
    trial_output = np.zeros(shape=(list_len * 2, Params.n_items))
    
    items = generate_sr_items(test)            
        
    for i in range(list_len):
        #encoding
        trial_input[i, items[i]] = 1
        trial_output[i, items[i]] = 1
        #recall
        #recall cue
        trial_input_recall_cue[i + list_len, 0] = 1
        #output letter
        trial_output[i + list_len, items[i]] = 1
            
    return (
            [trial_input.reshape(1, list_len * 2, Params.n_items),
             trial_input_recall_cue.reshape(1, list_len*2, 1)],
            trial_output.reshape(1, list_len *2, Params.n_items)
            )
    
def make_token_sr_trial(list_len):
    
    trial_input = np.zeros(shape=(list_len * 2, Params.n_tokens))
    trial_input_recall_cue = np.zeros(shape=(list_len * 2, 1))
    trial_output = np.zeros(shape=(list_len * 2, Params.n_tokens))    
    tokens = np.random.permutation(Params.n_tokens)
        
    for i in range(list_len):
        #encoding
        trial_input[i, tokens[i]] = 1
        trial_output[i, tokens[i]] = 1
        #recall
        #recall cue
        trial_input_recall_cue[i + list_len, 0] = 1
        #output letter
        trial_output[i + list_len, tokens[i]] = 1
            
    return (
            [trial_input.reshape(1, list_len * 2, Params.n_tokens),
             trial_input_recall_cue.reshape(1, list_len*2, 1)],
            trial_output.reshape(1, list_len *2, Params.n_tokens)
            )

token_perm = []

def gen_token():
    global token_perm
    
    if len(token_perm) == 0:
        token_perm = np.random.permutation(Params.n_tokens)
        
    token = token_perm[0]
    token_perm = token_perm[1:]
    
    return token

item_perm = []

def gen_item(n_items = None):
    global item_perm
    
    if n_items is None:
        n_items = Params.n_dual_train_items
        
    if len(item_perm) == 0:
        item_perm = np.random.permutation(n_items)
        
    item = item_perm[0]
    item_perm = item_perm[1:]
    
    return item

# Training
def make_bp_trial(list_len):    
    trial = {}
    
    #list_len = Params.n_tokens
    trial_input_token = np.zeros(shape=(list_len, Params.n_tokens))
    trial_input_item = np.zeros(shape=(list_len, Params.n_items))
    trial_input_recall_cue = np.zeros(shape=(list_len, 1))
    trial_output_item = np.zeros(shape=(list_len, Params.n_items))
    #letters = np.random.permutation(Params.n_tokens)

    items = np.random.permutation(Params.n_items)    
    token_i = gen_token() # Index of the token 
    item_i = items[0]
    items = items[1:]
    
    trial_input_item[0][item_i] = 1
    trial_output_item[0][item_i] = 1
    trial_input_token[0][token_i] = 1
    trial_input_recall_cue[0][0] = 0
    
    trial[token_i] = item_i
    for i in range(1, list_len):
        if len(items) < 1:
            items = np.random.permutation(Params.n_items)    
            
        if np.random.rand() > 0.5:
            # Recall
            rand_token = np.random.choice(list(trial.keys()))
            trial_output_item[i, trial[rand_token]] = 1
                        
            trial_input_token[i, rand_token] = 1
                    
            trial_input_recall_cue[i, 0] = 1 # This is the recall cue
                                                    # for this specific token
            
            
        else:
            # Encoding
            
            token_i = gen_token()
            item_i = items[0]
            items = items[1:]        
            
            trial_input_item[i][item_i] = 1
            trial_output_item[i][item_i] = 1
            trial_input_token[i][token_i] = 1
            #recal cue set to 0 for encoding
            trial_input_recall_cue[i, 0] = 0
            trial[token_i] = item_i
        
    return (    
                [
                    trial_input_item.reshape(1, list_len, Params.n_items), 
                    trial_input_token.reshape(1, list_len, Params.n_tokens),
                    trial_input_recall_cue.reshape(1, list_len, 1)
                ],
                trial_output_item.reshape(1, list_len, Params.n_items)
            )
    
def make_dual_trial(list_len):

    
    trial_input_token = np.zeros(shape=(list_len * 2, Params.n_tokens))
    trial_input_token_mask = np.zeros(shape=(list_len * 2, Params.n_tokens))
    trial_input_recall_cue = np.zeros(shape=(list_len * 2, 1))
    trial_input_item = np.zeros(shape=(list_len * 2, Params.n_items))
    trial_output_item = np.zeros(shape=(list_len * 2, Params.n_items))
    
    tokens = np.random.permutation(Params.n_tokens)
    items = []
    
    for i in range(list_len):
        
        if len(items) < 1:
            items = np.random.permutation(Params.n_dual_train_items)
            
        item_i = items[0]
        items = items[1:]
        
        #encoding
        #token
        trial_input_token[i, tokens[i]] = 1
        trial_input_token_mask[i+ list_len, :] = 1
#        trial_input_token[i + list_len, :] = 1
        #recall cue
        trial_input_recall_cue[i + list_len, 0] = 1
        #input item
        trial_input_item[i, item_i] = 1        
        #output item
        trial_output_item[i, item_i] = 1        
        trial_output_item[i + list_len, item_i] = 1        
    
    return (
                [
                    trial_input_item.reshape(1, list_len * 2, Params.n_items),                        
                    trial_input_token.reshape(1, list_len * 2, Params.n_tokens),                    
                    trial_input_recall_cue.reshape(1, list_len * 2, 1),
#                    trial_input_token_mask.reshape(1, list_len * 2, Params.n_tokens),                    
                ], [
                    trial_output_item.reshape(1, list_len * 2, Params.n_items),
                ]
            )
    
