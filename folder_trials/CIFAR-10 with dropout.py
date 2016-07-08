import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.layers.core import Activation

seed = 7
numpy.random.seed(seed)

# load data / CIFAR-10 dataset
(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
#X_train = X_train[:1000]
#Y_train = Y_train[:1000]


# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
num_classes = Y_test.shape[1]

# creacion del modelo
model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(3, 32, 32), border_mode='same', activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.5)) # Dropout de 20%
#model.add(Convolution2D(32, 3, 3, activation='relu', border_mode='same', W_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
#model.add(Dense(512, activation='relu', W_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

# compilar modelo
epochs = 5 
lrate = 0.1
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Build
#model = baseline_model()

model.save_weights('test.hdf5')
# Fit 
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath="savings_e{epoch}.hdf5", verbose=1)
history = model.fit(X_train, Y_train, batch_size=128, nb_epoch=5, verbose=1, validation_data=(X_test, Y_test), callbacks=[checkpointer])
#model.fit(X_train, Y_train, validation_data=(X_test, Y_test), nb_epoch=epochs, batch_size=32)
#model.fit(X_train, Y_train, nb_epoch=epochs, batch_size=32)


# from keras.callbacks import ModelCheckpoint




# Final evaluation of the model
scores = model.evaluate(X_test, Y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))

