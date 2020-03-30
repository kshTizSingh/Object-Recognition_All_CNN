# -*- coding: utf-8 -*-
"""
Created on Thu Oct 2 20:06:14 2019

"""
import keras
from keras.datasets import cifar10
from keras.utils import np_utils
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np

from PIL import Image
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

#Preview images randomly
#for i in range(0,9):
    #plt.subplot(330 + 1 + i)
    #img = X_train[i].transpose([0,1,2])
    #plt.imshow(img)
#
#plt.show()

# data preprocessing
seed = 6
np.random.seed(seed) 

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train = X_train / 255.0
X_test = X_test / 255.0

Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)
num_classes = Y_test.shape[1]
# importing layers - Convolution2D, Activation, Dropout, GlobalAveragePooling2d

from keras.models import Sequential
from keras.layers import Dropout, Activation, Conv2D, GlobalAveragePooling2D
from keras.optimizers import SGD

def allcnn(weights=None):
    # define model type - Sequential
    model = Sequential()

    # add model layers - Convolution2D, Activation, Dropout
    model.add(Conv2D(96, (3, 3), padding = 'same', input_shape=(32, 32, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(96, (3, 3), padding = 'same', strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (3, 3), padding = 'same', strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(192, (3, 3), padding = 'same'))
    model.add(Activation('relu'))
    model.add(Conv2D(192, (1, 1), padding = 'valid'))
    model.add(Activation('relu'))
    model.add(Conv2D(10, (1, 1), padding = 'valid'))

    # add GlobalAveragePooling2D layer with Softmax activation
    model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax'))
    
    # load the weights
    if weights:
        model.load_weights(weights)
    
    # return model
    return model

#define hyper parameter
    
learning_rate = 0.01
weight_decay = 1e-6
momentum  = 0.9

# biuld model

model = allcnn()

# define optimizer

sgd = SGD(lr = learning_rate, decay = weight_decay, momentum = momentum, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])


# model summary

model.summary()

epochs = 200
batch_size = 32

#fitting model

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs =epochs, batch_size = batch_size, verbose = 1)    
    
scores = model.evaluate(X_test, Y_test, verbose =1)
print('Accuray', scores[-1])

classes = range(0,10)

names = ['Airplane',
         'Automobile',
         'bird',
         'cat',
         'deer',
         'dog',
         'frog',
         'horse',
         'ship',
         'truck']

# dictionary of class labels

class_labels = dict(zip(classes, names))
print(class_labels)

#test for random 10 images in the dataset
batch = X_test[400:409]
labels  = np.argmax(Y_test[400:409], axis = -1)

prediction =model.predict(batch, verbose =1)
print(prediction)

# convert class probabilty to labels

#class_result = np.argmax(prediction, axis = -1)
#print(class_result)

#fig, axs = plt.subplots(3,3, figsize = (15,6))
#fig.subplots_adjust(hspace = 1)
#axs = axs.flatten()
#
#for i, img in enumerate(batch):
#    for key, value in class_labels.items():
#        if class_result[i]==key:
#            title  = 'predictions: {} \nActual: {}'.format(class_labels[key], class_labels[labels[i]])
#            axs[i].set_title(title)
#            axs[i].axes.get_xaxis().set_visible(False)
#            axs[i].axes.get_yaxis().set_visible(False)
#    axs[i].imshow(img.transpose([0,1,2]))
#    
#plt.show()





