import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


labels = ['blue', 'empty','yellow']
img_size = 68

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
