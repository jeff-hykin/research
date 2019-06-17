# allow relative imports, see https://stackoverflow.com/a/11158224/4367134
import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from common_tools import google_words()

def word2vec(word):
    vw = google_words()
    return vw[word]