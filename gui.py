from tkinter import *
from tkinter import filedialog
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import numpy as np
import keras
from keras import layers
from keras import models

img_size = 128
batch_size = 20
classes = 5

root = Tk()
root.title("Flower Classification")
title = Label(root, text="Flower Classification \n\n")
title.grid(row = 0, column = 1, sticky = N)

chooseImagesTxt = Label(root, text="Choose Folder where there are flowers to classify\t")
chooseImagesTxt.grid(row = 4, column = 0, sticky=W)
classifyPathEntry = Entry(root, width=40, bg="white")
classifyPathEntry.grid(row = 4, column = 1, sticky = W)


def browse_classify_folders():
    global folder_path
    folder_path = filedialog.askdirectory()
    classifyPathEntry.delete(0, END)
    classifyPathEntry.insert(END, folder_path)


chooseClassifyFolder = Button(root, text="Browse", width = 6, command = browse_classify_folders)
chooseClassifyFolder.grid(row = 4, column = 2, sticky = W)


chooseModelTxt = Label(root, text="Choose Trained Model Path\t")
chooseModelTxt.grid(row = 5, column = 0, sticky=W)
modelPathEntry = Entry(root, width=40, bg="white")
modelPathEntry.grid(row = 5, column = 1, sticky = W)


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
    model = get_model_structure(img_size, classes)
    model.load_weights(weights_path)


def browse_model_folders():
    global model_filename
    model_filename = filedialog.askopenfile().name
    modelPathEntry.delete(0, END)
    modelPathEntry.insert(END, model_filename)


chooseModelFolder = Button(root, text="Browse", width = 6, command = browse_model_folders)
chooseModelFolder.grid(row = 5, column = 2, sticky = W)


def getPredictedValue(pred):
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


def predict(path_from, path_to):
    global model

    numOfFiles = 0
    for root, dirs, files in os.walk(path_from):
        numOfFiles += len(files)

    train = ImageDataGenerator(rescale=1. / 255, horizontal_flip=True, shear_range=0.2, zoom_range=0.2, width_shift_range=0.2, height_shift_range=0.2, fill_mode='nearest', validation_split=0.2)

    test_generator = train.flow_from_directory(
        directory=path_from,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    test_generator.reset()

    pred = model.predict_generator(test_generator, verbose=1, steps=numOfFiles / batch_size)

    predicted_class_indices = np.argmax(pred, axis=1)
    predictions = [getPredictedValue(k) for k in predicted_class_indices]

    filenames = test_generator.filenames
    results = pd.DataFrame({"Filename": filenames, "Predictions": predictions})

    results.to_csv(path_to, sep=',', index=False)
    return results


def show_prediction(results):
    import plotly as py
    import plotly.graph_objs as go

    df = results

    trace = go.Table(
        header=dict(values=list(df.columns),
                    fill=dict(color='#C2D4FF'),
                    align=['left'] * 2),
        cells=dict(values=[df.Filename, df.Predictions],
                   fill=dict(color='#F5F8FF'),
                   align=['left'] * 2))

    data = [trace]
    fig = dict(data=data)
    py.offline.plot(fig, filename='d3-cloropleth-map.html')


def make_prediction():
    global folder_path
    path_to = filedialog.asksaveasfile(mode='w', defaultextension=".csv")
    if path_to is None:  # asksaveasfile return `None` if dialog closed with "cancel".
        return
    results = predict(classifyPathEntry.get(), path_to.name)
    show_prediction(results)

    popup = Tk()
    popup.title("Predicted")
    title1_popup = Label(popup, text="Predicted The Images Successfully \n\n")
    title1_popup.grid(row=0, column=1, sticky=N)
    title2_popup = Label(popup, text="The results file has been saved where you asked, and you will now be able to view the results in a different screen \n\n")
    title2_popup.grid(row=2, column=1, sticky=N)
    popup.mainloop()


def load_model():
    load_trained_model(modelPathEntry.get())
    popup = Tk()
    popup.title("Loaded")
    title_popup = Label(popup, text="Loaded The Model Successfully \n\n")
    title_popup.grid(row = 0, column = 1, sticky = N)
    popup.mainloop()


predictBtn = Button(root, text="Load Model", width = 7, command = load_model)
predictBtn.grid(row = 5, column = 3, sticky = W)

predictBtn = Button(root, text="Predict", width = 7, command = make_prediction)
predictBtn.grid(row = 6, column = 2, sticky = W)

root.mainloop()
