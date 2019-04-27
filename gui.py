from tkinter import *
from tkinter import filedialog
from flower_classification import get_model_structure
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import os
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import QWidget,QScrollArea, QTableWidget, QVBoxLayout,QTableWidgetItem

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


def load_trained_model(weights_path):
    global model, img_size, classes
    model = get_model_structure(img_size, classes)
    model.load_weights(weights_path)
    # print(model.summary())


def browse_model_folders():
    global model_filename
    model_filename = filedialog.askopenfile().name
    modelPathEntry.delete(0, END)
    modelPathEntry.insert(END, model_filename)
    load_trained_model(model_filename)


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
    '''
    win = QWidget()
    scroll = QScrollArea()
    layout = QVBoxLayout()
    table = QTableWidget()
    scroll.setWidget(table)
    layout.addWidget(table)
    win.setLayout(layout)

    table.setColumnCount(len(results.columns))
    table.setRowCount(len(results.index))
    for i in range(len(results.index)):
        for j in range(len(results.columns)):
            table.setItem(i, j, QTableWidgetItem(str(results.iloc[i, j])))

    win.show()
    '''
    #popup = Tk()
    #popup.wm_title("!")
    import plotly.plotly as py
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
    py.iplot(data, filename='results')
    #popup.mainloop()


def make_prediction():
    global folder_path
    path_to = filedialog.asksaveasfile(mode='w', defaultextension=".csv")
    if path_to is None:  # asksaveasfile return `None` if dialog closed with "cancel".
        return
    results = predict(folder_path, path_to.name)
    show_prediction(results)


predictBtn = Button(root, text="Predict", width = 7, command = make_prediction)
predictBtn.grid(row = 6, column = 2, sticky = W)

root.mainloop()