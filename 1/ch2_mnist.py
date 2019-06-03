from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
from pathlib import Path
# a relative import
common_tools = (lambda p,i={}:exec(Path(os.path.join(os.path.dirname(__file__),p)).read_text(),{},i)or i)('../common_tools.py')

# 
# get & reshape data
# 
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images  = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# combine the data
data = np.concatenate((train_images, test_images))
labels = np.concatenate((train_labels, test_labels))

# 
# Create the model
#
def create_model():
    network = models.Sequential()
    network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    network.add(layers.Dense(10, activation='softmax'))
    network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return network

# 
# Train and test
# 
def train_and_test_function(train_data, train_labels, test_data, test_labels):
    network = create_model()
    network.fit(train_data, train_labels, epochs=5, batch_size=128)
    return network.evaluate(test_data, test_labels)


results = common_tools["cross_validate"](data, labels, train_and_test_function, number_of_folds=7)
for each in results:
    print(each)