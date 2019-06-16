import numpy as np
import requests
import pickle
import os
from os.path import isabs, isfile, isdir, join, dirname, basename, exists, splitext
from os import remove, getcwd, makedirs, listdir, rename, rmdir
from shutil import move
from keras.models import load_model
import tempfile
import tarfile    

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

def make_sure_containing_folder_exists(a_path):
    from os.path import isabs, isfile, isdir, join, dirname, basename, exists
    import os
    # make abs if its not
    if not isabs(a_path):
        a_path = join(os.getcwd(), a_path)
    
    parent_folder = dirname(a_path)
    # create the data folder if it doesnt exist
    if not exists(parent_folder):
        os.makedirs(parent_folder)

def download(url, file_path):
    import requests
    r = requests.get(url)
    with open(file_path, 'wb') as f:
        f.write(r.content)


# a decorator for caching models
def cache_model_as(name_of_model, skip=False): 
    # dont edit the next line
    def inner(function_getting_wrapped): 
        # dont edit the next line
        def wrapper(*args, **kwargs):
            # 
            # EDIT this part
            # 
            model_file_path = name_of_model + ".nosync.h5"
            other_data_path = name_of_model + ".nosync.pickle"
            make_sure_containing_folder_exists(model_file_path)
            make_sure_containing_folder_exists(other_data_path)
            
            # check if model was already saved
            from os.path import isfile
            from keras.models import load_model
            other_data = []
            # if both files exist, then load them
            if isfile(model_file_path) and not skip:
                print(f"loading model {name_of_model} from local files\n\n")
                # load json and create model
                model = load_model(model_file_path)
                if isfile(other_data_path):
                    other_data = large_pickle_load(other_data_path)
                
            # if the model doesn't exist yet
            else:
                print(f'generating model {name_of_model}\n\n')
                # run the function to get the model
                model, *other_data = function_getting_wrapped(*args, **kwargs)
                # serialize model to JSON
                print(f'saving model {name_of_model}\n\n')
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

# a decorator for caching models
def cache_output_as(name_of_data, skip=False): 
    # dont edit the next line
    def inner(function_getting_wrapped): 
        # dont edit the next line
        def wrapper(*args, **kwargs):
            # 
            # EDIT this part
            # 
            
            # check if data was already saved
            from os.path import isfile
            import os
            
            data_path = name_of_data + ".nosync.pickle"
            make_sure_containing_folder_exists(data_path)
            
            # create the data folder if it doesnt exist
            if not os.path.exists(directory):
                os.makedirs(directory)
            
            # if both files exist, then load them
            if isfile(data_path) and not skip:
                print(f"loading {name_of_data} from local files\n\n")
                data = large_pickle_load(data_path)

            # if the data doesn't exist yet
            else:
                print(f'running function {name_of_data}\n\n')
                # run the function to get the model
                data = function_getting_wrapped(*args, **kwargs)
                # serialize model to JSON
                print(f'saving data {name_of_data}\n\n')
                # save the data using pickle
                large_pickle_save(other_data, data_path)
                
            # return the trained model
            return data
            
        # dont edit the next line
        return wrapper 
    # dont edit the next line
    return inner

def untar(tar_path, folder_path):
    from os import listdir, rename, rmdir
    from shutil import move
    import tempfile
    import tarfile
    
    print("untar-ing download")
    if not isdir(folder_path):
        tf = tarfile.open(tar_path)
        tf.extractall(path=folder_path)
    else:
        raise "there is already a {folder_path}, please remove/rename it before proceding"

def remove_wrapper_folder(folder_path):
    from os.path import isabs, isfile, isdir, join, dirname, basename, exists, splitext
    from os import remove, getcwd, makedirs, listdir, rename, rmdir
    from shutil import move
    import tempfile
    a_temp_location = tempfile.mkdtemp(prefix='wrapper_remover_')
    # pop it out of a wrapper folder
    files = listdir(folder_path)
    if len(files) == 1:
        # rename the file the same as the containing folder
        new_name = basename(folder_path)
        rename(join(folder_path, files[0]), join(folder_path,new_name))
        internal_file = join(folder_path, new_name)
        # move it to a temp location
        move(internal_file, a_temp_location)
        # remove the empty directory
        rmdir(folder_path)
        # # move it to what was the containing dir
        move(join(a_temp_location, new_name), dirname(folder_path))
    # if there's no files then remove the dir, there's probably an issue since the untar should be doing something
    elif len(files) == 0:
        rmdir(folder_path)

def easy_download(url, destination_folder, new_name):
    import regex as re
    destination_path = join(destination_folder, new_name)
    
    # 
    # check if already exists
    # 
    if exists(destination_path):
        raise f"There's already a {destination_path}"
    else:
        # check if its a zip or tar
        base, extension = splitext(destination_path)
        if extension == ".zip" or extension == ".gz":
            if exists(base):
                    raise f"There's already a {base2}"
            base2, extension2 = splitext(base)
            if extension2 == ".tar":
                if exists(base2):
                    raise f"There's already a {base2}"
                
    # 
    # download the file
    # 
    match = re.match(r'(https?:)?drive\.google\.com\/file\/.+?\/(?P<id>.+?)\/',url)
    # if its a google drive url, then extract the id and use it
    if match != None:
        download_file_from_google_drive(match.groups('id'), destination)
    # if its a normal url
    else:
        download(url, destination_path)
    
    # 
    # auto extract it
    # 
    base, extension1 = splitext(destination_path)
    print('extension1 = ', extension1)
    if extension1 == ".gz":
        base2, extension2 = splitext(base)
        if extension2 == ".tar":
            base = base2
        # move everything to a containing folder
        untar(destination_path, base)
    elif extension2 == ".zip":
        import zipfile
        zip_ref = zipfile.ZipFile(destination_path, 'r')
        zip_ref.extractall(base)
        zip_ref.close()
    
    # 
    # clean up the extraction
    # 
    remove_wrapper_folder(base)