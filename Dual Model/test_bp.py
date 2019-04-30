import numpy as np

from params import Params
from trials import make_bp_trial

import models

bp_model = models.create_BP_model()
bp_model.load_weights("bp.weights.h5")

results = np.zeros(shape=Params.max_list_len * 2)

for i in range(1000):
    trial = make_bp_trial(Params.max_list_len * 2)    
    target = trial[1][0]
    prediction = bp_model.predict(trial[0])[0]
    for l in range(Params.max_list_len * 2):
       results[l] += np.argmax(prediction[l]) == np.argmax(target[l])
print(results)
