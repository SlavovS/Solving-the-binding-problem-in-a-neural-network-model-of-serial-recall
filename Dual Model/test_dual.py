from __future__ import print_function

import numpy as np

import models
from keras.models import load_model
from params import Params
from trials import make_dual_trial

bp_model = models.create_BP_model()
sr_model = models.create_token_SR_model()
#
bp_model.load_weights("weights/bp.weights.18.h5")
sr_model.load_weights("token-sr.weights.h5")

dual_model = models.create_dual_model(bp_model, sr_model)
#bp_model.load_weights("bp.weights.h5")    

#models.get_model_layer(dual_model, "bp_lstm").set_weights(
#        models.get_model_layer(bp_model, "bp_lstm").get_weights()            
#    )
#
#models.get_model_layer(dual_model, "bp_output").set_weights(
#        models.get_model_layer(bp_model, "bp_output").get_weights()            
#    )
#    
#dual_model.load_weights("weights/dual.weights.27.h5")

n_rep = 1000
use_items = 4
skip_items = 0
orders = [None]#[True, False]
Params.n_dual_train_items = Params.n_items

#for e in [10]:
 #   print("\nEpoch {}".format(e))
#    dual_model.load_weights("weights/dual.weights.{}.h5".format(e))
for tls in [4]:#[1, 2, 3, 4, 5, 6, 7, 8, 9]:            
    pos_acc = np.zeros(shape=(2, tls))
    total_acc = [0.0, 0.0]
    avg_acc = [0.0, 0.0]
    for _ in range(n_rep):    
        
        for order in orders:

            dual_trial = make_dual_trial(
                    tls, 
                    tls, 
                    1, 
                    fixed_order=order, 
                    use_items=use_items,
                    skip_items=skip_items,
                    use_freq="low",
                )[0]
            dual_prediction = dual_model.predict(dual_trial)
            dual_target = dual_trial[0][0][:tls]
        
            acc = 1
            acc2 = 0.0
            indx = int(order) if order != None else 0
            
            for pos in range(0, tls):
                if (
                        np.argmax(dual_prediction[0][pos + tls]) 
                        == np.argmax(dual_target[pos])
                    ):
                    pos_acc[indx, pos] += 1.0
                    acc2 += 1.0
                else:
                    acc = 0.0
                    
            total_acc[indx] += acc
            avg_acc[indx] += (acc2 / tls)
               
    print("\n")
    print("List length: {}\n".format(tls))
    print("Serial recall curve:\n")
    print("\trandom order:\t{}".format(
            " ".join(
                    map(
                        lambda x: "{:.3f}".format(x), 
                        pos_acc[0] / n_rep
                    )
                )))
    print("\tfixed order:\t{}".format(
            " ".join(
                    map(
                        lambda x: "{:.3f}".format(x), 
                        pos_acc[1] / n_rep
                    )
                )))
    print("\n")
    print("Overall accuracy:\n")
    print("\trandom order:\t{:.2f}\t{:.2f}".format(total_acc[0] / n_rep, avg_acc[0] / n_rep))
    print("\tfixed order:\t{:.2f}\t{:.2f}".format(total_acc[1] / n_rep, avg_acc[1] / n_rep))


#dual_model.save_weights("dual.weights.h5")