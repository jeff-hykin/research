#%%
from keras import models
from keras import layers
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.datasets import reuters
from common_tools import vectorize_sequences

#%% 
# get & reshape data
#%%
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
# convert to one-hot encoding
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)
# vectorize the data as well
x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


#%%
# Create model
#%%
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])


#%%
# Train and test the model
#%%
# set aside validation data
x_validation = x_train[:1000]
partial_x_train = x_train[1000:]
y_validation = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]
history = model.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_validation, y_validation))

#%%
# Graph the loss
#%%
import matplotlib.pyplot as plt
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.ylim([0,2.5]) # set to be the same as 
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%
# Graph the accuracy
#%%
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%
# Retrain with the optimal number of epochs
#%%
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_validation, y_validation))
results = model.evaluate(x_test, one_hot_test_labels)
print('results = ', results)

#%%
