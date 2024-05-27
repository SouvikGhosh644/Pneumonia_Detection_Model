import tkinter
from tkinter import filedialog
from tkinter import font
from turtle import width
from PIL import Image, ImageTk
import pyttsx3



win = tkinter.Tk()
win.geometry("450x600")
win.maxsize(450,600)
win.minsize(450,600)
win.title("Penumonia Detection")


def upload_image():
    output.text=f"{''}\n {''}"
    output['text']=f"{''}\n {''}"
    e1.delete(0,tkinter.END)
    image_path = filedialog.askopenfilename()
    e1.insert(index=0, string=image_path)
    img = Image.open(image_path)
    img = img.resize((300,300))
    img = ImageTk.PhotoImage(img)
    i1 = tkinter.Label(win)
    i1.grid(row=1, column=1, pady=12)
    i1.image = img
    i1['image']=img
    predict_btn = tkinter.Button(win, text="Predict", command=predict)
    predict_btn.grid(row=2, column=1)

    predict_btn1 = tkinter.Button(win, text="Predict (New Version)", command=predict_updated)
    predict_btn1.grid(row=3, column=1, pady=12)

def predict():
    from email.mime import image
    from pyexpat import model
    import tensorflow as tf
    # from keras import models, layers
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

    MODEL = tf.keras.models.load_model(r"D:\Deep Learning & ML\pneumonia_detection\models_cnn\3")

    image_path = e1.get()

    image = cv2.imread(image_path)

    # print(image.shape)

    resized_image = cv2.resize(image, (256, 256))

    # print(resized_image.shape)

    img_array = tf.keras.preprocessing.image.img_to_array(resized_image)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = MODEL.predict(img_array)


    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    # confidence = np.max(predictions[0])
    confidence = round(100 * (np.max(predictions[0])), 2)
    
    output.grid(row=4, column=1, pady=12)
    output.text=f"Prediction: {predicted_class}\n Accuracy: {confidence}%\n (Old Version)"
    output['text']=f"Prediction: {predicted_class}\n Accuracy: {confidence}%\n (Old Version)"
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say(f"Prediction: {predicted_class}, accuracy: {confidence}%")
    engine.runAndWait()

def predict_updated():
    from email.mime import image
    from pyexpat import model
    import tensorflow as tf
    # from keras import models, layers
    import matplotlib.pyplot as plt
    import cv2
    import numpy as np

    CLASS_NAMES = ['NORMAL', 'PNEUMONIA']

    MODEL = tf.keras.models.load_model(r"D:\Deep Learning & ML\pneumonia_detection\model_cnn_updated\2")

    image_path = e1.get()

    image = cv2.imread(image_path)

    # print(image.shape)

    resized_image = cv2.resize(image, (224, 224))

    # print(resized_image.shape)

    img_array = tf.keras.preprocessing.image.img_to_array(resized_image)
    img_array = tf.expand_dims(img_array, 0) # Create a batch
    img_array = img_array / 255.0

    predictions = MODEL.predict(img_array)


    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    # confidence = np.max(predictions[0])
    confidence = round(100 * (np.max(predictions[0])), 2)
    
    output.grid(row=4, column=1, pady=12)
    output.text=f"Prediction: {predicted_class}\n Accuracy: {confidence}%\n (New Version)"
    output['text']=f"Prediction: {predicted_class}\n Accuracy: {confidence}%\n (New Version)"
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)
    engine.say(f"Prediction: {predicted_class}, accuracy: {confidence}%")
    engine.runAndWait()
    


l1 = tkinter.Label(win, text="Upload Image", pady=12)
l1.grid(row=0, column=0, sticky="e")
e1 = tkinter.Entry(win, width=45)
e1.grid(row=0, column=1, padx=12)
button_browse = tkinter.Button(win, text="Browse", command=upload_image)
button_browse.grid(row=0, column=2)

output = tkinter.Label(win, width=20, font=(20))



win.mainloop()