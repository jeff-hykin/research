def get_dataset():
    import os
    import shutil
    import zipfile
    from pathlib import Path
    common_tools = (lambda p,i={}:exec(Path(os.path.join(os.path.dirname(__file__),p)).read_text(),globals(),i)or i)('../common_tools.py')

    training_folder_name   = 'train.nosync'
    validation_folder_name = 'validate.nosync'
    testing_folder_name    = 'test.nosync'
    folders = [ training_folder_name, validation_folder_name, testing_folder_name ]
    dir_of_this_file = os.path.dirname(__file__)
    dirs = [ os.path.join(dir_of_this_file, each) for each in folders ]

    # go to where this file is
    current_dir = os.getcwd()
    os.chdir(dir_of_this_file)

    # check if they already exist
    all_exist = True
    for each in folders:
        all_exist = all_exist and os.path.isdir(each)
    if all_exist:
        os.chdir(current_dir)
        return dirs

    # make the folders if they dont exist
    for each in folders:
        try:
            shutil.rmtree(each)
        except:
            pass
        os.mkdir(each)

    # 
    # download if doesnt exist
    #
    # print("downloading the google file for Cats and Dogs. It's 800Mb so its going to take awhile")
    dataset_name = "dataset.zip"
    common_tools['download_file_from_google_drive']('1JjA15jy42W90q5Oi-o4jK9qLklHgZses', dataset_name)

    # unzip the download
    with zipfile.ZipFile(dataset_name, 'r') as zip_obj:
        zip_obj.extractall('./'+os.path.splitext(dataset_name)[0])
    # unzip the training set
    with zipfile.ZipFile('train.zip', 'r') as zip_obj:
        zip_obj.extractall('./train')
    # rename it
    os.rename("train", training_folder_name) 
    # remove zip files
    os.remove("test1.zip")
    os.remove("train.zip")
    os.remove("sampleSubmission.csv")
    os.remove(dataset_name)

    from glob import glob
    import math

    # open up the training folder
    all_train = glob(training_folder_name + "/*")
    fraction_of_train_data = math.floor(len(all_train) / 4)

    # move the files
    def move(dest, all_train=all_train, fraction_of_train_data=fraction_of_train_data):
        for each in list(all_train)[-fraction_of_train_data:]:
            os.rename(each, os.path.join(dest, os.path.basename(each)))
            # remove that file from the list
            all_train.pop()
            
    # move the data
    move(validation_folder_name)

    # move the data
    move(testing_folder_name)
    
    # change back
    os.chdir(current_dir)
    return dirs