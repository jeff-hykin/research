import numpy as np
import requests
import pickle
import os
from os.path import isfile
from keras.models import load_model

def vectorize_sequences(sequences, dimension=10000):
    import numpy as np
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
          results[i, sequence] = 1.
    return results

def cross_validate(data, labels, train_and_validate_function, number_of_folds=6):
    import numpy as np
    """
    data
        needs to have its first dimension (the len()) be the number of data points
    train_and_validate_function
        needs to have 4 arguments, train_data, train_labels, test_data, and test_labels
        it should return accuracy information as output
    """
    # check number of folds
    if (len(data) % number_of_folds):
        raise "The data needs to be divisible by the number of folds"
    
    results = []
    batch_size = int(len(data) / number_of_folds)
    for batch_number in range(number_of_folds):
        print("\nOn fold:",batch_number+1)
        start_index = batch_number * batch_size
        end_index = (batch_number + 1) * batch_size
        test_data = data[start_index:end_index]
        test_labels = labels[start_index:end_index]
        train_data   = np.concatenate((  data[0:start_index],   data[end_index:len(data)-1]))
        train_labels = np.concatenate((labels[0:start_index], labels[end_index:len(data)-1]))
        results.append(train_and_validate_function(train_data, train_labels, test_data, test_labels))
    return results

def download_file_from_google_drive(id, destination):
    import requests
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value
        return None
    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


def google_words():
    import pickle
    import os
    print("loading the google words. This will take a bit")
    model_path = "setup/google_word2vec.nosync.model"
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(model_path)
    with open(model_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)

    return pickle.loads(bytes_in)

def large_pickle_load(file_path):
    import pickle
    import os
    max_bytes = 2**31 - 1
    bytes_in = bytearray(0)
    input_size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f_in:
        for _ in range(0, input_size, max_bytes):
            bytes_in += f_in.read(max_bytes)
    return pickle.loads(bytes_in)

def large_pickle_save(variable, file_path):
    import pickle
    bytes_out = pickle.dumps(variable, protocol=4)
    max_bytes = 2**31 - 1
    with open(file_path, 'wb') as f_out:
        for idx in range(0, len(bytes_out), max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])

# a decorator for caching models
def cache_model_as(name_of_model, skip=False): 
    # dont edit the next line
    def inner(function_getting_wrapped): 
        # dont edit the next line
        def wrapper():
            # 
            # EDIT this part
            # 
            
            model_file_path = name_of_model + ".nosync.h5"
            other_data_path = name_of_model + ".nosync.pickle"
            
            # check if model was already saved
            from os.path import isfile
            from keras.models import load_model
            other_data = []
            # if both files exist, then load them
            if isfile(model_file_path) and not skip:
                print("loading model from local files")
                # load json and create model
                model = load_model(model_file_path)
                if isfile(other_data_path):
                    other_data = large_pickle_load(other_data_path)
                
            # if the model doesn't exist yet
            else:
                print('generating model')
                # run the function to get the model
                model, *other_data = function_getting_wrapped()
                # serialize model to JSON
                print('saving model')
                model.save(model_file_path)
                if len(other_data) > 0:
                    # save all the other data as a pickle file
                    large_pickle_save(other_data, other_data_path)
                
            # return the trained model
            return (model, *other_data)
            
        # dont edit the next line
        return wrapper 
    # dont edit the next line
    return inner