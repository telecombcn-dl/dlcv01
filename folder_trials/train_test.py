from __future__ import print_function

import numpy as np

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import mnist
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.models import Sequential
from keras.utils import np_utils

batch_size = 128
nb_classes = 10
nb_epoch = 10

# input image dimensions
img_rows, img_cols = 28, 28
# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()

model.add(Convolution2D(32, nb_conv, nb_conv,
                        border_mode='valid',
                        activation='relu',
                        input_shape=(1, img_rows, img_cols),
                        name='conv1_1'))
model.add(Convolution2D(32, nb_conv, nb_conv, activation='relu', name='conv1_2'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), name='pool1'))
model.add(Convolution2D(64, nb_conv, nb_conv, activation='relu', name='conv2'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), name='pool2'))

model.add(Flatten())
model.add(Dense(128, activation='relu', name='fc1'))
model.add(Dense(nb_classes, activation='softmax', name='output'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.summary()

callbacks = [
    ModelCheckpoint('./model_snapshot/weights_e{epoch:02d}.hdf5'),
    EarlyStopping(patience=1)
]

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test), callbacks=callbacks)
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
