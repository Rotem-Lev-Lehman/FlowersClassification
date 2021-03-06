# must use these rows:
import keras
from keras import layers


# keep the function's name as it is...
def initializeModel(model, img_size, classes):
    # first layer, must have: input_shape=(img_size, img_size, 3)
    model.add(keras.layers.Conv2D(128, kernel_size = (3, 3), padding='same', input_shape=(img_size, img_size, 3), activation='relu'))

    # middle layers - do whatever you want with them:
    model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.2))

    # use model.add() to add any layers you like
    # read Keras documentation to find which layers you can use:
    #           https://keras.io/layers/core/
    #           https://keras.io/layers/convolutional/
    #           https://keras.io/layers/pooling/
    #

    # last layer should be with softmax activation function - do not change!!!
    model.add(layers.Dense(classes, activation='softmax'))


def getName():
    return 'model_rotem_11'
