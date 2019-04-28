import keras
from keras import layers
from keras import models
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import sys
import os
import pandas as pd
import numpy as np
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


def get_model_structure(img_size, classes):
    model = models.Sequential()

    model.add(keras.layers.Conv2D(96, kernel_size=(3, 3), padding='same', input_shape=(img_size, img_size, 3), activation='relu'))

    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(384, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, activation='relu'))
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


def load_trained_model(weights_path):
    global model, img_size, classes
    #from model_rotem_12 import initializeModel
    #model = models.Sequential()
    #initializeModel(model, img_size, classes)
    model = get_model_structure(img_size, classes)
    model.load_weights(weights_path)
    #print(model.summary())


def getPredictedValue(pred):
    '''
    maxProb = -1
    pred = -1
    for i in range(len(prediction)):
        if prediction[i] > maxProb:
            maxProb = prediction[i]
            pred = i
    '''
    if pred == 0:
        return 'daisy'
    elif pred == 1:
        return 'dandelion'
    elif pred == 2:
        return 'rose'
    elif pred == 3:
        return 'sunflower'
    else:  # pred == 4
        return 'tulip'


def predict(path):
    global model
    # from keras.preprocessing import image

    '''
    test_generator = test_datagen.flow_from_directory(
    directory=pred_dir,
    target_size=(28, 28),
    color_mode="rgb",
    batch_size=32,
    class_mode=None,
    shuffle=False
    )
    '''
    numOfFiles = 0
    for root, dirs, files in os.walk(path):
        numOfFiles += len(files)

    test_generator = train.flow_from_directory(
        directory=path,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    test_generator.reset()

    pred = model.predict_generator(test_generator, verbose=1, steps=numOfFiles / batch_size)

    predicted_class_indices = np.argmax(pred, axis=1)
    '''
    labels = test_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    '''
    predictions = [getPredictedValue(k) for k in predicted_class_indices]

    filenames = test_generator.filenames
    results = pd.DataFrame({"Filename": filenames, "Predictions": predictions})

    results.to_csv("output.csv", sep=',', index=False)

    '''
    img_path = 'flowers/rose/110472418_87b6a3aa98_m.jpg'
    # img = image.load_img(img_path, target_size=(224, 224)) # if a you want a spesific image size
    img = image.load_img(img_path, target_size = (img_size, img_size))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x * 1. / 255  # rescale as training
    '''
    '''
    test_gen = train.flow_from_directory(path, target_size=(img_size, img_size), batch_size=batch_size, class_mode='categorical')
    to_test = np.asarray(test_gen)
    print("hello")
    prediction = model.predict(to_test)  # Vector with the prob of each class
    print("hello from the other side")
    print(prediction)
    print(getPredictedValue(prediction[0]))
    '''
    '''
    c = model.predict_classes(test_set)

    for i in range(len(test_set)):
        print("X=%s, Predicted=%s" % (test_set[i], c[i]))
    '''

# Split images into Training and Validation Sets (20%)


train = ImageDataGenerator(rescale=1./255,horizontal_flip=True, shear_range=0.2, zoom_range=0.2,width_shift_range=0.2,height_shift_range=0.2, fill_mode='nearest', validation_split=0.2)

img_size = 128
batch_size = 20
t_steps = 3462/batch_size
v_steps = 861/batch_size
classes = 5
flower_path = "flowers"
train_gen = train.flow_from_directory(flower_path, target_size = (img_size, img_size), batch_size = batch_size, class_mode='categorical', subset='training')
valid_gen = train.flow_from_directory(flower_path, target_size = (img_size, img_size), batch_size = batch_size, class_mode = 'categorical', subset='validation')

'''
#test_gen = train.
load_trained_model('model_rotem_12_fittingModel13With72epoch.h5')
predict("C:\\Users\\Rotem\\Desktop\\תואר ראשון\\הכנה לפרוייקט\\עבודה 2\\flowers")
'''
# Model
model = get_model_structure(img_size, classes)
'''
model = models.Sequential()

# you can change the 'from' parameter to the file you are using your model on:
# example, change to from model_<my name>_<my index> import initializeModel
'''
'''
from model_rotem_12 import getName
load_trained_model('model_rotem_12_fittingModel13With72epoch.h5')
'''
'''
from model_rotem_12 import initializeModel, getName
initializeModel(model, img_size, classes)
'''
'''
from model_rotem_9 import initializeModel, getName
model = initializeModel(img_size, classes)
'''
'''
# this should print your model's structure...
print(model.summary())
'''
# fill optimizer argument using one of keras.optimizers.
# read Keras documentation : https://keras.io/models/model/
optimizer = 'sgd'
# fill loss argument using keras.losses.
# reads Keras documentation https://keras.io/losses/
loss = keras.losses.mean_squared_error
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
'''
from timeit import default_timer as timer
start = timer()
'''
# you can change number of epochs by changing the value of the 'epochs' paramter
model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs= 80 , validation_data=valid_gen, validation_steps=v_steps)

# model.save(getName() + '.h5')
model.save('flowers_model.h5')

plt_modle(model_hist)
'''
end = timer()
elapsed = end - start # Time in seconds
print('time in minutes: ')
print(elapsed / 60)
'''
