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


def load_trained_model(weights_path):
    global model
    from model_rotem_1 import initializeModel
    model = models.Sequential()
    initializeModel(model, img_size, classes)
    model.load_weights(weights_path)
    print(model.summary())


def predict(test_set):
    global model
    c = model.predict_classes(test_set)

    for i in range(len(test_set)):
        print("X=%s, Predicted=%s" % (test_set[i], c[i]))


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

#test_gen = train.
#load_trained_model('flowers_model.h5')
#predict(valid_gen)

# Model

model = models.Sequential()

# you can change the 'from' parameter to the file you are using your model on:
# example, change to from model_<my name>_<my index> import initializeModel
from model_rotem_10 import initializeModel, getName
initializeModel(model, img_size, classes)
'''
from model_rotem_9 import initializeModel, getName
model = initializeModel(img_size, classes)
'''
# this should print your model's structure...
print(model.summary())
# fill optimizer argument using one of keras.optimizers.
# read Keras documentation : https://keras.io/models/model/
optimizer = 'sgd'
# fill loss argument using keras.losses.
# reads Keras documentation https://keras.io/losses/
loss = keras.losses.mean_squared_error
model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

from timeit import default_timer as timer
start = timer()

# you can change number of epochs by changing the value of the 'epochs' paramter
model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs= 30 , validation_data=valid_gen, validation_steps=v_steps)

model.save(getName() + '.h5')
# model.save('flowers_model.h5')

plt_modle(model_hist)

end = timer()
elapsed = end - start # Time in seconds
print('time in minutes: ')
print(elapsed / 60)
