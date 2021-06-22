import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import resize
from imageio import imread
import io
import base64

labels = ['blue', 'empty','yellow']
img_size = 58

def decode_base64(base64data):
    converted_images = []
    test = ""
    for img in base64data:
        img_arr = imread(io.BytesIO(base64.b64decode(img)))#[...,::-1]
        test = img_arr
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB) 
        resized_arr = cv2.resize(img_arr, (img_size, img_size))
        converted_images.append(resized_arr)
    print(test)
    return np.array(converted_images)

def plot_spot(img, label):
    plt.figure(figsize = (5,5))
    plt.imshow(img)
    plt.title(label)
    plt.show()
