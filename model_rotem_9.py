# must use these rows:
import keras
from keras import layers
from keras import models
from keras import applications


# keep the function's name as it is...
def initializeModel(img_size, classes):
    model = applications.VGG16(include_top=False, weights='imagenet')
    # first layer, must have: input_shape=(img_size, img_size, 3)
    model.add(keras.layers.Flatten(input_shape=(img_size, img_size, 3)))
    # middle layers - do whatever you want with them:

    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    # use model.add() to add any layers you like
    # read Keras documentation to find which layers you can use:
    #           https://keras.io/layers/core/
    #           https://keras.io/layers/convolutional/
    #           https://keras.io/layers/pooling/
    #

    # last layer should be with softmax activation function - do not change!!!
    model.add(layers.Dense(classes, activation='softmax'))
    return model


def getName():
    return 'model_rotem_9'
