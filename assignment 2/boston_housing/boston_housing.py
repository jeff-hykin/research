from keras import models
from keras import layers
import numpy as np
from keras.datasets import boston_housing
import matplotlib.pyplot as plt
import os
from pathlib import Path
# a relative import
common_tools = (lambda p,i={}:exec(Path(os.path.join(os.path.dirname(__file__),p)).read_text(),{},i)or i)('../../common_tools.py')

#%%
# Get data
#%%
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

num_epochs = 500
def train_and_evaluate_function(train_data, train_labels, test_data, test_labels):
    global num_epochs
    model = build_model()
    history = model.fit(train_data, train_labels, epochs=num_epochs, batch_size=1)
    val_mse, val_mae = model.evaluate(test_data, test_labels)
    return val_mae, history

#%%
# Run cross validation
#%%
all_results = common_tools["cross_validate"](train_data, train_targets, train_and_evaluate_function, number_of_folds=4)


#%%
# Graph of validations
#%%
all_scores = [ each[0] for each in all_results ]  
print('all_scores = ', all_scores)
all_mae_histories = [ each[1].history['val_mean_absolute_error'] for each in all_results ]
average_mae_history = [ np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]
plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

#%%
# smaller graph
#%%
def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points
smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()


#%%
# Retrain
#%%
model = build_model()
model.fit(train_data, train_targets, epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print('test_mse_score = ', test_mse_score)
print('test_mae_score = ', test_mae_score)