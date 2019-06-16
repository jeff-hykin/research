from os.path import isabs, isfile, isdir, join, dirname, basename, exists
from os import remove, getcwd, makedirs
# allow relative imports, see https://stackoverflow.com/a/11158224/4367134
import os, sys, inspect
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from common_tools import make_sure_containing_folder_exists, download, easy_download

try:
    easy_download(
        url="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        destination_folder=dirname(__file__),
        new_name="imdb_database.nosync.tar.gz"
    )
except:
    pass

