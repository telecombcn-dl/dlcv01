# Mnist 
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras.layers.core import Activation
import matplotlib.pyplot as plt
# seed random
seed = 7
numpy.random.seed(seed)
# Load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
#X_train = X_train[:1000]
#y_train = y_train[:1000]
#X_test = X_test[:1000]
#y_test = y_test[:1000]

# Paso float
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# Normalizo inputs de 0-1
X_train = X_train / 255
X_test = X_test / 255

Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)
num_classes = Y_test.shape[1]
def baseline_model():
	# Creo modelo
	model = Sequential()
	model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.0))
	model.add(Flatten())
#	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compilo modelo
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# Build
model = baseline_model()

model.save_weights('test.hdf5')
# Fit 
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath="savings_e{epoch}.hdf5", verbose=1)
history = model.fit(X_train, Y_train, batch_size=128, nb_epoch=10, verbose=1, validation_data=(X_test, Y_test), callbacks=[checkpointer])
#model.fit(X_train, y_train, validation_data=(X_test, y_test), nb_epoch=10, batch_size=200, verbose=1)



#model = Sequential()
#model.add(Dense(10, input_dim=784, init='uniform'))
#model.add(Activation('softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

'''
saves the model weights after each epoch if the validation loss decreased
'''
#checkpointer = ModelCheckpoint(filepath="savings_e{epoch}.hdf5", verbose=1)
#model.fit(X_train, Y_train, batch_size=128, nb_epoch=20, verbose=1, validation_data=(X_test, Y_test), callbacks=[checkpointer])

# Evaluacion
scores = model.evaluate(X_test, Y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

plt.subplot(2, 1, 1)
plt.title('Loss')
plt.xlabel('Epoch')
plt.plot(history.history['loss'], '-o', label='Training loss')
plt.plot(history.history['val_loss'], '-o', label='Validation loss')
plt.legend(loc='upper right')

plt.subplot(2, 1, 2)
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.plot(history.history['acc'], '-o', label='Training accuracy')
plt.plot(history.history['val_acc'], '-o', label='Validation accuracy')
plt.legend(loc='lower right')
plt.show()
