from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator


# 
# settings
# 
number_of_colors_per_pixel = 3
dimensions_of_pictures = (150, 150, number_of_colors_per_pixel)
feature_size = (3,3)
number_of_features = 1 # ???



train_dir      = "../setup/cats_and_dogs_train.nosync"
validation_dir = "../setup/cats_and_dogs_validate.nosync"

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen  = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )
validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

# 
# Model
# 
model = models.Sequential()
model.add(layers.Conv2D(32, feature_size, activation='relu', input_shape=dimensions_of_pictures))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, feature_size, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, feature_size, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, feature_size, activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

 
model.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(lr=1e-4),
metrics=['acc'])