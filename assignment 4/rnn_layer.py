import numpy as np

def rnn_layer(batches, output_shape):
    # initilize the network 
    network = {
        "input_weights": np.random.random((output_features, input_features)),
        "prev_prediction_weights": np.random.random((output_features, output_features)),
        "bias": np.random.random((output_features,))
    }
    
    all_predictions_for_all_batches = []
    for each_batch in batches:
        all_predictions = []
        # create a state that will change after each input
        previous_prediction = np.zeros(*output_shape)
        for each_input in each_batch:
            # 
            # Calculate the next state
            # 
            scaled_prev_prediction = np.dot(network["prev_prediction_weights"], previous_prediction)
            scaled_input           = np.dot(network["input_weights"]          , each_input)
            raw_prediction = scaled_input + scaled_prev_prediction + network["bias"]
            new_prediction = np.tanh(raw_prediction)
            # record the prediction
            all_predictions.append(new_prediction)
            # start the next iteration
            previous_prediction = new_prediction
        # record the predictions
        all_predictions_for_all_batches.append(all_predictions_for_all_batches)
    
    return all_predictions_for_all_batches