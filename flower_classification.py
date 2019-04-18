import keras
from keras import layers
from keras import models
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import sys
from PIL import Image
sys.modules['Image'] = Image


def plt_modle(model_hist):
    acc = model_hist.history['acc']
    val_acc = model_hist.history['val_acc']
    loss = model_hist.history['loss']
    val_loss = model_hist.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 6));
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, color='#0984e3', marker='o', linestyle='none', label='Training Accuracy')
    plt.plot(epochs, val_acc, color='#0984e3', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, color='#eb4d4b', marker='o', linestyle='none', label='Training Loss')
    plt.plot(epochs, val_loss, color='#eb4d4b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()

# Split images into Training and Validation Sets (20%)

train = ImageDataGenerator(rescale=1./255,horizontal_flip=True, shear_range=0.2, zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2, fill_mode='nearest', validation_split=0.2)

img_size = 128
batch_size = 20
t_steps = 3462/batch_size
v_steps = 861/batch_size
classes = 5
flower_path = "flowers";
train_gen = train.flow_from_directory(flower_path, target_size = (img_size, img_size), batch_size = batch_size, class_mode='categorical', subset='training')
valid_gen = train.flow_from_directory(flower_path, target_size = (img_size, img_size), batch_size = batch_size, class_mode = 'categorical', subset='validation')

# Model

model = models.Sequential()

model.add(keras.layers.Conv2D(64,kernel_size=(5,5), padding='same', input_shape=(128,128,3), activation='relu'))
model.add(keras.layers.Conv2D(80,kernel_size=(3,3), padding='same', activation='relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Conv2D(128, kernel_size=(7, 7), padding='same', activation='relu'))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(300,activation='relu'))

'''
model.add(Conv1D(60,32, strides=1, activation='relu',padding='causal',input_shape=(None,64,1)))
model.add(Conv1D(80,10, strides=1, activation='relu',padding='causal'))
model.add(Dropout(0.25))
model.add(Conv1D(100,5, strides=1, activation='relu',padding='causal'))
model.add(MaxPooling1D(1))
model.add(Dropout(0.25))
model.add(Dense(300,activation='relu'))
model.add(Dense(1,activation='relu'))
'''
'''
model.add(keras.layers.Conv1D(60,32, strides=1, activation='relu',padding='causal',input_shape=(img_size, img_size, 3)))
model.add(keras.layers.Conv1D(80,10, strides=1, activation='relu',padding='causal'))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.25))
#model.add(keras.layers.Conv1D(100,5, strides=1, activation='relu',padding='causal'))
#model.add(keras.layers.MaxPooling1D(1))
#model.add(keras.layers.Dropout(0.25))
'''
'''
model.add(keras.layers.MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last', input_shape=(128,128,3)))
filters = 20
kernel_size = 3
model.add(keras.layers.Conv1D(filters, kernel_size, strides=1, padding='valid', data_format='channels_last', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
#model.add(keras.layers.Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None))
'''
# use model.add() to add any layers you like
# read Keras documentation to find which layers you can use:
#           https://keras.io/layers/core/
#           https://keras.io/layers/convolutional/
#           https://keras.io/layers/pooling/
#

# last layer should be with softmax activation function - do not change!!!
model.add(layers.Dense(classes, activation='softmax'))
print(model.summary())
# fill optimizer argument using one of keras.optimizers.
# read Keras documentation : https://keras.io/models/model/
optimizer = 'sgd'
# fill loss argument using keras.losses.
# reads Keras documentation https://keras.io/losses/
loss = keras.losses.mean_squared_error
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# you can change number of epochs by changing the value of the 'epochs' paramter
model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs= 8 , validation_data=valid_gen, validation_steps=v_steps)
model.save('flowers_model.h5')
plt_modle(model_hist)

