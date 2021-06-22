import tensorflow as tf
import numpy as np
from util import labels, img_size, get_data, plot_spot
import sys
from sklearn.metrics import classification_report

def testing():
    data = get_data("input/web_ui")
  
    x_test = []
    y_test = []

    for feature, label in data:
        x_test.append(feature)
        y_test.append(label)

    x_test = np.array(x_test)
    # print(x_test)
    
    # print(data[0])
    # sys.exit(1)

    model = tf.keras.models.load_model("model.h5")
    prediction_matrices = model.predict(x_test)  # REMEMBER YOU'RE PASSING A LIST OF THINGS YOU WISH TO PREDICT

    predictions = []
    for i, prediction in enumerate(prediction_matrices):
        # prediction = [prediction]
        # print(prediction)
        predictions.append(np.argmax(prediction))
        label = labels[np.argmax(prediction)]
        
        
       
        # assert predictions[-1] == y_test[i], "paixthke trolia"
        # plot_spot()
        # print("-> Predicted marker: ", label)
        # print("=" * 60)

    print(classification_report(y_test, predictions, target_names = ['Blue (Class 0)','Empty (Class 1)','Yellow (Class 2)']))

    print(prediction)
    # return
    prediction = prediction[0]
    print(prediction)
    print(labels[int(prediction)])





if __name__ == "__main__":
    testing()