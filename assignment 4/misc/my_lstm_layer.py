import numpy as np

def lstm_layer(batches, sensory_output_shape, motor_output_shape):
    # initilize the network 
    network = {
        "sensory_input_weights": np.random.random(batches[0][0].shape),
        "forgetter_weights"    : np.random.random(output_shape),
        "motor_output_weights" : np.random.random(motor_output_shape),
        "bias"                 : np.random.random((output_features,))
    }
    
    how_to_respond_at_each_moment_for_all_batches = []
    for each_batch in batches:
        how_to_respond_at_each_moment = []
        # create a state that will change after each input
        previous_prediction = np.zeros(*output_shape)
        for each_input in each_batch:
            # 
            # Calculate the next state
            # 
            what_to_remember       = np.dot(network["forgetter_weights"]    , previous_prediction)
            what_is_happening      = np.dot(network["sensory_input_weights"], each_input)
            new_thoughts           = np.tanh(scaled_input + scaled_prev_prediction + network["bias"])
            how_to_respond         = np.dot(network["motor_output_weights"], new_thoughts)
            # record the prediction
            how_to_respond_at_each_moment.append(how_to_respond)
            # start the next iteration
            previous_prediction = how_to_respond
        # record the predictions
        how_to_respond_at_each_moment_for_all_batches.append(how_to_respond_at_each_moment_for_all_batches)
    
    return how_to_respond_at_each_moment_for_all_batches