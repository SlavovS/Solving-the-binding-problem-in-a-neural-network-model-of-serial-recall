import keras
from keras.layers import Input, LSTM, TimeDistributed, Dense, SimpleRNN
from keras.models import Model
import keras.backend as K

from params import Params

def create_SR_model(n_inputs, n_hidden_units):
    sr_input = Input(
                    shape=(None, n_inputs),
                    name="sr_input"
                )
    
    sr_input_recall_cue = Input(
            shape = (None, 1), 
            name="sr_input_recall_cue"
            )
    
    sr_all_inputs = keras.layers.concatenate([sr_input, sr_input_recall_cue])
    
    sr_lstm = LSTM(
                units = n_hidden_units, 
                return_sequences=True,
                name="sr_lstm"
            )(sr_all_inputs)
    
    sr_output = TimeDistributed(
                Dense(
                    units=n_inputs,
                    activation="softmax",
                ),
                name="sr_output"
            )(sr_lstm)
    
    sr_model = Model(
            inputs = [sr_input, sr_input_recall_cue], 
            outputs = [sr_output]
        )
    
    return sr_model

def create_token_SR_model():
    return create_SR_model(n_inputs=Params.n_tokens, n_hidden_units=50)

def create_item_SR_model():
    return create_SR_model(n_inputs=Params.n_items, n_hidden_units=100)

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
        
def create_dual_model(bp_model, sr_model):
    
    token_mask = keras.layers.Input(shape=(None, Params.n_tokens), name="token_mask")
       
    sr_model_output = keras.layers.Multiply()([
                sr_model.output,
                token_mask
            ])
    
    dual_token_input = keras.layers.Add()([
                sr_model_output,
                sr_model.inputs[0]
            ])
    
        
    dual_bp_all_input = keras.layers.concatenate(
                [
                    #item input
                    bp_model.inputs[0],
                    #token input
#                    dual_token_input,
                    sr_model.output,
                    #recall cue from SR model
                    sr_model.inputs[1]
#                    keras.layers.Lambda(lambda x: x, name="bp_recall_cue")(
#                            sr_model.inputs[1]
#                        )
                ]
            )
    dual_lstm = LSTM(
            units = 100, 
            return_sequences=True,
            name="bp_lstm"
        )(dual_bp_all_input)
    
    def train_softmax(x):
        return keras.backend.concatenate([
                keras.activations.softmax(x[:, :Params.n_dual_train_items]),
                keras.activations.softmax(x[:, Params.n_dual_train_items:])
            ])
            
    
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
                        #token mask
#                        token_mask
                ],
                outputs = [dual_output]
            )
        
    #load input->lstm weights from BP model
    get_model_layer(dual_model, "bp_lstm").set_weights(
            get_model_layer(bp_model, "bp_lstm").get_weights()            
        )
    get_model_layer(dual_model, "bp_lstm").trainable = False    
    
    #load lstm->output weights from BP model
    get_model_layer(dual_model, "bp_output").set_weights(
            get_model_layer(bp_model, "bp_output").get_weights()            
        )
    get_model_layer(dual_model, "bp_output").trainable = False
    
    return dual_model