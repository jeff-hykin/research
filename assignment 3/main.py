from keras import layers 
from keras import models

# 
# settings
# 
number_of_colors_per_pixel = 1
dimensions_of_pictures = (28, 28, number_of_colors_per_pixel)
feature_size = (3,3)
number_of_features = 1 # ???

# 
# Model
# 
model = models.Sequential()
model.add(layers.Conv2D(32, feature_size, activation='relu', input_shape=dimensions_of_pictures))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, feature_size, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, feature_size, activation='relu'))


import os, shutil

original_dataset_dir = '/Users/fchollet/Downloads/kaggle_original_data'

base_dir = '/Users/fchollet/Downloads/cats_and_dogs_small'         ; os.mkdir(base_dir)
train_dir           = os.path.join(base_dir       , 'train'      ) ; os.mkdir(train_dir)
validation_dir      = os.path.join(base_dir       , 'validation' ) ; os.mkdir(validation_dir)
test_dir            = os.path.join(base_dir       , 'test'       ) ; os.mkdir(test_dir)
train_cats_dir      = os.path.join(train_dir      , 'cats'       ) ; os.mkdir(train_cats_dir)
train_dogs_dir      = os.path.join(train_dir      , 'dogs'       ) ; os.mkdir(train_dogs_dir)
validation_cats_dir = os.path.join(validation_dir , 'cats'       ) ; os.mkdir(validation_cats_dir)
validation_dogs_dir = os.path.join(validation_dir , 'dogs'       ) ; os.mkdir(validation_dogs_dir)
test_cats_dir       = os.path.join(test_dir       , 'cats'       ) ; os.mkdir(test_cats_dir)
test_dogs_dir       = os.path.join(test_dir       , 'dogs'       ) ; os.mkdir(test_dogs_dir)

def copy(name, dest, the_range=(0,1000), source=original_dataset_dir):
    fnames = [(name+'.{}.jpg').format(i) for i in range(*the_range)]
    for fname in fnames:
        src = os.path.join(source, fname)
        dst = os.path.join(dest, fname)
        shutil.copyfile(src, dst)

copy('cat', train_cats_dir     , the_range=(0   ,1000))
copy('cat', validation_cats_dir, the_range=(1000,1500))
copy('cat', test_cats_dir      , the_range=(1500,2000))
copy('dog', train_dogs_dir     , the_range=(0   ,1000))
copy('dog', validation_dogs_dir, the_range=(1000,1500))
copy('dog', test_dogs_dir      , the_range=(1500,2000))