import numpy as np
# initilize the networks
networks = {
    "forget_gatekeeper" : model(input_shape, output_shape), # FIXME: input_shape and output_shape need to be defined
    "input_gatekeeper"  : model(input_shape, output_shape), # FIXME: input_shape and output_shape need to be defined
    "input_ignore_gatekeeper" : model(input_shape, output_shape), # FIXME: input_shape and output_shape need to be defined
    "output_gatekeeper" : model(input_shape, output_shape), # FIXME: input_shape and output_shape need to be defined
}

# https://www.youtube.com/watch?v=8HyCNIVRbSU
def lstm_unit(current_sensory_input,  previous_predicton, memory):
    global networks
    # 
    # Forget what needs to be forgotten
    #
    what_to_forget = np.sigmoid(networks["forget_gatekeeper"] (current_sensory_input, previous_prediction))
    memory *= what_to_forget
    
    # 
    # update the memory with relevent information
    # 
    what_to_remember = np.tanh(   networks["input_gatekeeper"]         (each_input, previous_prediction))
    what_to_ignore   = np.sigmoid(networks["input_ignore_gatekeeper"]  (each_input, previous_prediction))
    what_to_remember *= what_to_ignore
    memory           += what_to_remember
    
    # 
    # Use the memory to make a prediction
    # 
    what_to_output   = np.sigmoid(networks["output_gatekeeper"]  (each_input, previous_prediction))
    prediction       = np.tanh(memory) * what_to_output
    
    return prediction, memory
