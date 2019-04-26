import keras
from keras.layers import Input, LSTM, TimeDistributed, Dense
from keras.models import Model

from params import Params

def create_SR_model():
    sr_input = Input(
                    shape=(None, Params.n_tokens),
                    name="sr_input"
                )
    
    sr_input_recall_cue = Input(
            shape = (None, 1), 
            name="sr_input_recall_cue"
            )
    
    sr_all_inputs = keras.layers.concatenate([sr_input, sr_input_recall_cue])
    
    sr_lstm = LSTM(
                units = 50, 
                return_sequences=True,
                name="sr_lstm"
            )(sr_all_inputs)
    
    sr_output = TimeDistributed(
                Dense(
                    units=Params.n_tokens,
                    activation="softmax",
                ),
                name="sr_output"
            )(sr_lstm)
    
    sr_model = Model(
            inputs = [sr_input, sr_input_recall_cue], 
            outputs = [sr_output]
        )
    
    return sr_model

def create_BP_model():
    bp_item_input = Input(shape=(None, Params.n_items), name="bp_item_input")
    bp_token_input = Input(shape=(None, Params.n_tokens), name="bp_token_input")
    bp_recall_cue_input = Input(shape=(None, 1), name="bp_recall_cue_input")
    bp_all_inputs = keras.layers.concatenate(
            [
                    bp_item_input, 
                    bp_token_input, 
                    bp_recall_cue_input
            ])
    bp_lstm = LSTM(
            units = 100, 
            return_sequences=True,
            name="bp_lstm"
        )(bp_all_inputs)
    bp_output = TimeDistributed(
                Dense(
                    units=Params.n_items, 
                    activation="softmax"                
                ),
                name="bp_output"
            )(bp_lstm)
    
    bp_model = Model(
            inputs = [bp_item_input, bp_token_input, bp_recall_cue_input], 
            outputs = [bp_output]
        )    
    
    return bp_model

def get_model_layer(model, layer_name):
    for layer in model.layers:
        if layer.name == layer_name:
            return layer
        
def create_DUAL_model(bp_model, sr_model):
    
    dual_bp_all_input = keras.layers.concatenate(
                [
                    #item input
                    bp_model.inputs[0],
                    #token output of SR model
                    sr_model.output, 
                    #copy recall cue from SR model
                    keras.layers.Lambda(lambda x: x, name="bp_recall_cue")(
                            sr_model.inputs[1]
                        )
                ]
            )
    dual_lstm = LSTM(
            units = 100, 
            return_sequences=True,
            name="bp_lstm"
        )(dual_bp_all_input)
    
    dual_output = TimeDistributed(
                Dense(
                    units = Params.n_items, 
                    activation="softmax"                
                ),
                name="bp_output"
            )(dual_lstm)
    
    dual_model = Model(
                inputs = [               
                        #item input
                        bp_model.inputs[0],
                        #token input
                        sr_model.inputs[0],
                        #recall cue input
                        sr_model.inputs[1],
                ],
                outputs = [dual_output]
            )
    
    #load input->lstm weights from BP model
    get_model_layer(dual_model, "bp_lstm").set_weights(
            get_model_layer(bp_model, "bp_lstm").get_weights()
        )
    
    #load lstm->output weights from BP model
    get_model_layer(dual_model, "bp_output").set_weights(
            get_model_layer(bp_model, "bp_output").get_weights()
        )

    return dual_model