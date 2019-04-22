from tkinter import *
from tkinter import filedialog

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
    filename = filedialog.askdirectory()
    classifyPathEntry.delete(0, END)
    classifyPathEntry.insert(END, filename)


chooseClassifyFolder = Button(root, text="Browse", width = 6, command = browse_classify_folders)
chooseClassifyFolder.grid(row = 4, column = 2, sticky = W)


chooseModelTxt = Label(root, text="Choose Trained Model Path\t")
chooseModelTxt.grid(row = 5, column = 0, sticky=W)
modelPathEntry = Entry(root, width=40, bg="white")
modelPathEntry.grid(row = 5, column = 1, sticky = W)


def browse_model_folders():
    global folder_path
    filename = filedialog.askdirectory()
    modelPathEntry.delete(0, END)
    modelPathEntry.insert(END, filename)


chooseModelFolder = Button(root, text="Browse", width = 6, command = browse_model_folders)
chooseModelFolder.grid(row = 5, column = 2, sticky = W)


def make_prediction():
    pass


predictBtn = Button(root, text="Predict", width = 7, command = make_prediction)
predictBtn.grid(row = 6, column = 2, sticky = W)

root.mainloop()