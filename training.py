import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from sklearn.metrics import classification_report,confusion_matrix
import sys
import numpy as np
from util import labels, img_size, get_data, plot_spot


def build_model(filters, layersN):
    model = Sequential()
    model.add(Conv2D(64,3,padding="same", activation="relu", input_shape=(68,68,3)))
    model.add(MaxPool2D())

    model.add(Conv2D(64, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())

    model.add(Conv2D(128, 3, padding="same", activation="relu"))
    model.add(MaxPool2D())
    model.add(Dropout(0.4))

    model.add(Flatten())
    model.add(Dense(256,activation="relu"))
    model.add(Dense(3, activation="softmax"))

    model.summary()
    return model


def get_image_augmentation():
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=True)  # randomly flip images
    
    return datagen


def training():
    # train = get_data('input/train')
    train = get_data('input/test')

    l = []
    for i in train:
        if(i[1] == 0):
            l.append("blue")
        elif(i[1] == 1):
            l.append("empty")
        else:
            l.append("yellow")
            
    sns.set_style('darkgrid')
    sns.countplot(l)

    plot_spot(train[0][0], labels[train[0][1]])
    plot_spot(train[-1][0], labels[train[-1][1]])

    x_train = []
    y_train = []

    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    # for feature, label in val:
    #   x_val.append(feature)
    #   y_val.append(label)

    # Normalize the data
    x_train = np.array(x_train) / 255
    # x_val = np.array(x_val) / 255

    x_train.reshape(-1, img_size, img_size, 1)
    y_train = np.array(y_train)


    # x_val.reshape(-1, img_size, img_size, 1)
    # y_val = np.array(y_val)


    # get_image_augmentation().fit(x_train)
    print("-> x train shape", x_train.shape)
    print("-> y train shape", y_train.shape)
    
    model = build_model(0, 0)

    epochs=200
    opt = Adam(lr=0.000001)
    # utils.plot_model(model, to_file='model_plot4.png', show_shapes=True, show_layer_names=False)
    model.compile(optimizer=opt, loss = SparseCategoricalCrossentropy(from_logits=False), metrics = ['accuracy'])

    history = model.fit(x_train, y_train,
                shuffle=True,
                batch_size=32,
                epochs=epochs,
                verbose=2,
                validation_split=0.1)

    model.save('model2.h5')

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')

    plt.savefig('plot_test2.png', bbox_inches='tight')
    plt.show()

    # predictions = model.predict_classes(x_val)
    # predictions = predictions.reshape(1,-1)[0]
    # print(classification_report(y_val, predictions, target_names = ['Blue (Class 0)','Empty (Class 1)','Yellow (Class 2)']))


if __name__ == "__main__": 
    training()