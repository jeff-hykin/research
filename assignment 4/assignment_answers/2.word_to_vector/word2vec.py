# allow relative imports, see https://stackoverflow.com/a/11158224/4367134
import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '../../..'))
from common_tools import google_words()

def word2vec(word):
    vw = google_words()
    return vw[word]
    
from gensim.models import Word2Vec


def nearest_word(word):
    model = gensim.models.KeyedVectors.load_word2vec_format("../../../setup/GoogleNews.nosync.bin", binary=True)
    ms = model.most_similar(word, 10)
    for x in ms:
        print x[0],x[1]