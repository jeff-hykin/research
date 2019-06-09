import os
from pathlib import Path
common_tools = (lambda p,i={}:exec(Path(os.path.join(os.path.dirname(__file__),p)).read_text(),{},i)or i)('../common_tools.py')
# 
# get the google file
#
google_file_name = 'cats_and_dogs.nosync.zip'
# go to where this file is
os.chdir(os.path.dirname(__file__))
google_download_exists = os.path.isdir("test1") and os.path.isdir("train")
if not google_download_exists:
    print("downloading the google file for Cats and Dogs. It's 800Mb so its going to take awhile")
    common_tools['download_file_from_google_drive']('1JjA15jy42W90q5Oi-o4jK9qLklHgZses', google_file_name)
    # unzip it
    os.system(f"unzip {google_file_name}")
    os.system(f"unzip test1.zip")
    os.system(f"unzip train.zip")
    # rename it
    os.rename("test1", "cats_and_dogs_test.nosync") 
    os.rename("train", "cats_and_dogs_train.nosync") 
    # remove zip files
    os.remove("test1.zip")
    os.remove("train.zip")
    os.remove("sampleSubmission.csv")
    os.remove("cats_and_dogs.nosync.zip")

