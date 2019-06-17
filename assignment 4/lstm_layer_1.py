import numpy as np

def lstm_layer(batches, sensory_output_shape, motor_output_shape):
    # initilize the networks
    networks = {
        "selector" : model(input_shape, output_shape), # FIXME: input_shape and output_shape need to be defined
        "forgetter": model(input_shape, output_shape), # FIXME: input_shape and output_shape need to be defined
        "ignorer"  : model(input_shape, output_shape), # FIXME: input_shape and output_shape need to be defined
        "main"  : model(input_shape, output_shape), # FIXME: input_shape and output_shape need to be defined
    }
    
    how_to_respond_at_each_moment_for_all_batches = []
    for each_batch in batches:
        how_to_respond_at_each_moment = []
        # create a state that will change after each input
        previous_prediction = np.zeros(*output_shape)
        for each_input in each_batch:
            # 
            # Calculate each network (a "pre" Calculation)
            # 
            what_to_ignore   = np.sigmoid(networks["ignorer"]   (each_input, previous_prediction))
            what_to_forget   = np.sigmoid(networks["forgetter"] (each_input, previous_prediction))
            what_to_output   = np.sigmoid(networks["selector"]  (each_input, previous_prediction))
            # 
            # Run the thought process pipeline
            # 
            new_all_thoughts      = np.tanh(networks["main"]         (each_input, previous_prediction))
            new_relevent_thoughts = new_all_thoughts  * what_to_ignore
            relevent_memories     = what_to_forget    * old_info
            new_info              = relevent_thoughts + relevent_memories
            how_to_respond        = what_to_output    * np.sigmoid(new_info)
            
            # record the prediction
            how_to_respond_at_each_moment.append(how_to_respond)
            
            # start the next iteration
            previous_prediction = how_to_respond
            old_info = new_info
            
        # record the predictions
        how_to_respond_at_each_moment_for_all_batches.append(how_to_respond_at_each_moment_for_all_batches)
    
    return how_to_respond_at_each_moment_for_all_batches