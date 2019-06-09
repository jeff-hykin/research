from torch.utils.data.dataset import Dataset
from torchvision import transforms
from download_dogs_and_cats import get_dataset
from keras.preprocessing import image
import os
from os import path
import numpy as np

def image_as_tensor(path_to_image, target_size):
    img = image.load_img(path_to_image, target_size=target_size)
    # convert it to an array
    data = image.img_to_array(img)
    return data

class CatsAndDogs(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, transform=None, train_validate_or_test="train", file_extension=".jpg"):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        
        self.transform = transform
        # make sure the data exists
        train_dir, validate_dir, test_dir = get_dataset()
        # pick which one
        if train_validate_or_test == "train":
            self.dir = train_dir
        if train_validate_or_test == "validate":
            self.dir = validate_dir
        if train_validate_or_test == "test":
            self.dir = test_dir
        # create a mapping from ints to file names
        self.path_for = []
        list_of_image_filenames = os.listdir(self.dir)
        for each_file_name in list_of_image_filenames:
            # only add files with the correct extension
            if each_file_name[-len(file_extension):] == file_extension:
                self.path_for.append(path.join(self.dir, each_file_name))

    def __len__(self):
        return len(self.path_for)

    def __getitem__(self, index):
        # 
        # get the data
        #
        path_to_image = self.path_for[index]
        data = image_as_tensor(path_to_image=path_to_image, target_size=(32,32))
        
        #
        # get the label
        # 
        if "cat" in path.basename(path_to_image):
            label = 0
        else: # must be a dog
            label = 1
            
        # 
        # transform the sample 
        # 
        if self.transform:
            data = self.transform(data)
        return data, label