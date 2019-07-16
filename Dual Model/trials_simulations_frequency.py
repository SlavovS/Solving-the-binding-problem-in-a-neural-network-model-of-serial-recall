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
    
#def gen_items(use_bigrams):
#    items = np.random.permutation(Params.n_items)
#    
#    if not use_bigrams:
#        return items
#    
#    for bigram in Params.bigrams:
#        for i in range(len(items) - 1):                    
#            if items[i] == bigram[0]:                
#                if items[i + 1] != bigram[1]:                    
#                    if np.random.rand() < 0.75:
#                       return gen_items(use_bigrams)
#
#        if items[len(items) - 1] == bigram[0]:
#            if items[0] != bigram[1]:
#                return gen_items(use_bigrams)
#                
#    return items
def gen_items(fixed_order = None, skip_items = 0, use_items = 0, use_freq = None):
    if fixed_order is None:
        if use_freq is not None:
            probs = np.ones(shape=Params.n_items)
            if use_freq == "high":
                probs[Params.n_items // 2:] = 0
            elif use_freq == "low":
                probs[:Params.n_items // 2] = 0
            else:
                probs[:Params.n_items // 2] = 2
                
            probs = probs / np.sum(probs)
            
            items = np.random.choice(
                        np.arange(Params.n_items),
                        size=Params.max_list_len,
                        p=probs,
                        replace=False
                    )
                        
            return items
        
        while True:
            if use_items > 0:
                items = np.concatenate((
                            np.arange(use_items),
                            np.random.permutation(Params.n_items - use_items) + use_items
                        ))
                
            items = np.random.permutation(Params.n_items)
            stop_loop = True
            for i in range(skip_items):
                if items[i] == i:
                    stop_loop = False
                    break
                                
            if stop_loop:
                break
            
        for i in range(skip_items):
            assert(items[i] != i)
            
        return items
    
    if fixed_order:
        rnd_offset = np.random.randint(0, Params.n_items)
        items = np.arange(Params.n_items)
        return np.concatenate((items[rnd_offset:], items[:rnd_offset]))
    else:
        dont_stop = True
        items = None
        while(dont_stop):
            dont_stop = False
            items = np.random.permutation(Params.n_items)
            for i in range(len(items) - 1):
                if items[i + 1] == items[i] + 1:
                    dont_stop = True
                    break
            if items[len(items) - 1] == items[0]:
                dont_stop = True
                break
        return items

# Training
def make_bp_trial(
        list_len, 
        fixed_order = False, 
        serial_recall = False, 
        skip_items = 0,
        use_items = 0,
        use_freq = None
    ):    
    trial = {}
                
    #list_len = Params.n_tokens
    trial_input_token = np.zeros(shape=(list_len, Params.n_tokens))
    trial_input_item = np.zeros(shape=(list_len, Params.n_items))
    trial_input_recall_cue = np.zeros(shape=(list_len, 1))
    trial_output_item = np.zeros(shape=(list_len, Params.n_items))
    #letters = np.random.permutation(Params.n_tokens)

    items = gen_items(fixed_order, skip_items, use_items, use_freq)
    tokens = np.random.permutation(Params.n_tokens)
    
    token_i = tokens[0]
    tokens = tokens[1:]
    item_i = items[0]
    items = items[1:]
    
    trial_input_item[0][item_i] = 1
    trial_output_item[0][item_i] = 1
    trial_input_token[0][token_i] = 1
    trial_input_recall_cue[0][0] = 0
    
    trial[token_i] = item_i
    
    token_seq = []
    token_seq.append(token_i)
    
    for i in range(1, list_len):
        
        if len(items) < 1:            
            items = gen_items(fixed_order, skip_items, use_items, use_freq)

        if len(tokens) < 1:            
            tokens = np.random.permutation(Params.n_tokens)
            
#        if serial_recall:
#            do_recall = (i >= list_len / 2)
#        else:
#            do_recall = np.random.rand() > 0.5
            
        do_recall = (i >= list_len / 2)
            
        if do_recall:
            # Recall
            if serial_recall:        
                rand_token = token_seq[0]                
                token_seq = token_seq[1:]
            else:
                rand_token = np.random.choice(list(trial.keys()))
                
            trial_output_item[i, trial[rand_token]] = 1
                        
            trial_input_token[i, rand_token] = 1
                    
            trial_input_recall_cue[i, 0] = 1 # This is the recall cue
                                                    # for this specific token
            
            
        else:
            # Encoding
            
            token_i = tokens[0]
            token_seq.append(token_i)
            tokens = tokens[1:]
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
    
def make_dual_trial(
        min_list_len, 
        max_list_len, 
        batch_size = 1, 
        reps = False,
        fixed_order = False,
        skip_items = 0,
        use_items = 0,
        use_freq = None
    ):

    n_steps = Params.max_list_len * 2    
    n_examples = batch_size * (max_list_len - min_list_len + 1)
    
    trial_input_token = np.zeros(shape=(n_examples, n_steps, Params.n_tokens))
    trial_input_token_mask = np.zeros(
                shape=(n_examples, n_steps, Params.n_tokens)
            )
    trial_input_recall_cue = np.zeros(shape=(n_examples, n_steps, 1))
    trial_input_item = np.zeros(shape=(n_examples, n_steps, Params.n_items))
    trial_output_item = np.zeros(shape=(n_examples, n_steps, Params.n_items))
           
    example_i = 0
    
    for b in range(batch_size):        
        for list_len in range(min_list_len, max_list_len + 1):
            tokens = np.random.permutation(Params.n_tokens)
            items = gen_items(fixed_order, skip_items, use_items, use_freq)
            for i in range(list_len):
                #encoding
                #token
                trial_input_token[example_i, i, tokens[i]] = 1
                trial_input_token_mask[example_i, i, :] = 0
                trial_input_token_mask[example_i, i + list_len, :] = 1
        #        trial_input_token[i + list_len, :] = 1
                #recall cue
                trial_input_recall_cue[example_i, i + list_len, 0] = 1
                #input item
                trial_input_item[example_i, i, items[i]] = 1        
                #output item
                trial_output_item[example_i, i, items[i]] = 1        
                trial_output_item[example_i, i + list_len, items[i]] = 1        
                
            example_i += 1
    return (
                [
                    trial_input_item,
                    trial_input_token,
                    trial_input_recall_cue,
                    trial_input_token_mask
                ], [
                    trial_output_item
                ]
            )
    
