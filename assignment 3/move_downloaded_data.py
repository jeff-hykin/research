import os, shutil

original_dataset_dir = '/Users/fchollet/Downloads/kaggle_original_data'
base_dir            = '/Users/fchollet/Downloads/cats_and_dogs_small' ;     os.mkdir(base_dir)
# data groups
train_dir           = os.path.join(base_dir      , 'train')       ; os.mkdir(train_dir)
validation_dir      = os.path.join(base_dir      , 'validation')  ; os.mkdir(validation_dir)
test_dir            = os.path.join(base_dir      , 'test')        ; os.mkdir(test_dir)
# cats
train_cats_dir      = os.path.join(train_dir     , 'cats')        ; os.mkdir(train_cats_dir)
validation_cats_dir = os.path.join(validation_dir, 'cats')        ; os.mkdir(validation_cats_dir)
test_cats_dir       = os.path.join(test_dir      , 'cats')        ; os.mkdir(test_cats_dir)
# dogs
train_dogs_dir      = os.path.join(train_dir     , 'dogs')        ; os.mkdir(train_dogs_dir)
validation_dogs_dir = os.path.join(validation_dir, 'dogs')        ; os.mkdir(validation_dogs_dir)
test_dogs_dir       = os.path.join(test_dir      , 'dogs')        ; os.mkdir(test_dogs_dir)

def move(name, a_range, target):
    global original_dataset_dir
    fnames = [name+'.{}.jpg'.format(i) for i in range(*a_range)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(target, fname)
        shutil.copyfile(src, dst)

ranges = [ (0,1000), (1000, 1500), (1500, 2000) ]
    

fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)
fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)