import numpy as np

import models
from params import Params
from trials import make_item_sr_trial
   
sr_model = models.create_item_SR_model()
sr_model.load_weights("item-sr.weights.h5")

n_reps = 1000
results = {}

for test_pos in range(len(Params.tests)):
    
    for test_item in Params.tests:
    
        if test_item not in results:
            results[test_item] = np.zeros(shape=len(Params.tests))

        for i in range(n_reps):
        
            if Params.tests[test_pos] == test_item:
                test = (test_pos, test_item)
            else:
                item = Params.tests[test_pos]
                while item == Params.tests[test_pos]:
                    item = np.random.randint(0, Params.n_tokens)
                test = (test_pos, item)
                
            trial = make_item_sr_trial(Params.test_list_len, test)
            target = trial[1][0][Params.test_list_len:,]
            output = sr_model.predict(trial[0])[0][Params.test_list_len:,]
            
            if np.argmax(target[test_pos]) == np.argmax(output[test_pos]):
                results[test_item][test_pos] += 1

for test_item in results:
    print("{}:\t\t{}".format(
            test_item, 
            "\t".join(map(str, results[test_item] / float(n_reps)))
            )
        )