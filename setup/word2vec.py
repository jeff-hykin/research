import pickle
import gensim
import os
from pathlib import Path
# allow relative imports, see https://stackoverflow.com/a/11158224/4367134
import os,sys,inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from common_tools import download_file_from_google_drive

# 
# get the google file
#
google_file_name = 'setup/GoogleNews.nosync.bin'
google_download_exists = os.path.isfile(google_file_name)
if not google_download_exists:
    print("downloading the google file for Word2Vec. It's 1.5Gb so its going to take awhile")
    download_file_from_google_drive('0B7XkCwpI5KDYNlNUTTlSS21pQmM', google_file_name+".gz")
    os.system(f"gunzip {google_file_name}.gz")

# Load Google's pre-trained Word2Vec model.
print("Loading google's model (this is going to take awhile)")
model = gensim.models.KeyedVectors.load_word2vec_format(google_file_name, binary=True)  

# Save it so that it doesn't need to be built every time
print("saving the trained model")
bytes_out = pickle.dumps(model)
max_bytes = 2**31 - 1
with open("setup/google_word2vec.nosync.model", 'wb') as f_out:
    for idx in range(0, len(bytes_out), max_bytes):
        f_out.write(bytes_out[idx:idx+max_bytes])
print("finished saving model")
