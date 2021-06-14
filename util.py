import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import resize
from imageio import imread
import io
import base64

labels = ['blue', 'empty','yellow']
img_size = 68

def decode_base64(base64data):
    converted_images = []
    for img in base64data:
        img_arr = imread(io.BytesIO(base64.b64decode(img)))[...,::-1]
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB) 
        resized_arr = cv2.resize(img_arr, (img_size, img_size))
        converted_images.append(resized_arr)
        
    return np.array(converted_images)

def get_data(data_dir):
    data = [] 
    
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)

    return np.array(data)

def plot_spot(img, label):
    plt.figure(figsize = (5,5))
    plt.imshow(img)
    plt.title(label)
    plt.show()
