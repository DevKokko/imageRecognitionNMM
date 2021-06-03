import cv2
import tensorflow as tf
import numpy as np
import os


labels = ['blue', 'empty','yellow']

def prepare(filepath):
  try:
    IMG_SIZE = 68  # 50 in txt-based
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
    reshape = new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
  except Exception as e:
    print(str(e))
  return reshape  # return the image with shaping that TF wants.
  
  
  

model = tf.keras.models.load_model("model.h5")
prediction = model.predict([prepare('../blue.png')])  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT
prediction[0][0]
print(labels[int(prediction[0][0])])
