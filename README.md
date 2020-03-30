# Object Recognition using CNN
Keras implementation for object recognition on the cifar10 dataset

# Model used (C)

 ![](CNN_Snip_C.jpg)

# Layers for reference
```
model = Sequential()

    
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
```
# Prediction
The accuracy for the reduced epochs of 200 for reduced training time because of a weaker GPU, the accuracy comes out to be 82.65 %.
Expected with a pre-trained model - 87.51 %.

# References
*The All Convolutional Net : arXiv:1412.6806*
