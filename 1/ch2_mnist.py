from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
from sklearn.model_selection import StratifiedKFold

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

def cross_validate(data, labels, train_and_test_function, number_of_folds=6):
    """
    data
        needs to have its first dimension (the len()) be the number of data points
    train_and_test_function
        needs to have 4 arguments, train_data, train_labels, test_data, and test_labels
        it should return accuracy information as output
    """
    # check number of folds
    if (len(data) % number_of_folds):
        raise "The data needs to be divisible by the number of folds"
    
    results = []
    batch_size = int(len(data) / number_of_folds)
    for batch_number in range(number_of_folds):
        print("\nOn fold:",batch_number+1)
        start_index = batch_number * batch_size
        end_index = (batch_number + 1) * batch_size
        test_data = data[start_index:end_index]
        test_labels = labels[start_index:end_index]
        train_data   = np.concatenate((  data[0:start_index],   data[end_index:len(data)-1]))
        train_labels = np.concatenate((labels[0:start_index], labels[end_index:len(data)-1]))
        results.append(train_and_test_function(train_data, train_labels, test_data, test_labels))
    return results

results = cross_validate(data, labels, train_and_test_function, number_of_folds=7)
for each in results:
    print(each)