# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 18:58:19 2016

@author: and_ma

"""
## Following this tutorial:
## https://github.com/wxs/keras-mnist-tutorial

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn import metrics
import time
# 1. Loading data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print("Dataset info data")
print("Train dataset size",X_train.shape)
print("Test dataset size",X_test.shape)

# 2. Viusalising the data
'''
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(X_train[i], cmap='gray', interpolation='none')
    plt.title("Class {}".format(y_train[i]))
 '''
   
# 3.reshaping images and normalising    
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print("Training matrix shape", X_train.shape)
print("Testing matrix shape", X_test.shape)    

nb_classes = len(np.unique(y_test))
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

## 4. Build Network
## Build the neural-network. Here we'll do a simple 3 layer fully connected network.
# input -> 784(28*28)
# Layer-1 -> 512 Neurons
# Layer-2 -> 512 Neurons
# Layer-3 -> 10 Neurons

# 4.1 Create sequential Object
model = Sequential()
# 4.2 Create Layer-1
model.add(Dense(512,input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# 4.3 Create Layer-2
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

## experimenting
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.2))

# 4.4 Output Layer-3
model.add(Dense(10))
model.add(Activation('softmax'))

# 4.4 Specifying loss function and optimizer
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

# 4.5 Training the Model
#Hyper-parameters

batch_size = 128
nb_epoch = 4
show_accuracy = True
verbose = 1

start_time = time.time()

model.fit(X_train, Y_train,batch_size=batch_size, nb_epoch=nb_epoch,\
          show_accuracy=show_accuracy, verbose=1,\
          validation_data=(X_test, Y_test))

print("Training required time is %s seconds" % (time.time() - start_time))

score = model.evaluate(X_test, Y_test,show_accuracy=True, verbose=0)
print('Test score:', score)

# 4.6 Inspecting Output
# The predict_classes function outputs the highest probability class
# according to the trained classifier for each input example.
#predicted_classes = model.predict_classes(X_test)

predicted_classes_labels = model.predict(X_test)
# Check which items we got right / wrong
correct_indices = np.nonzero(predicted_classes_labels == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes_labels != y_test)[0]


# Sklearn Metrics
'''
acc = metrics.accuracy_score(predicted_classes_labels, y_test)
f1 = metrics.f1_score(predicted_classes_labels, y_test)
precision = metrics.precision_score(predicted_classes_labels, y_test)
recall = metrics.recall_score(predicted_classes_labels, y_test)

print ('Logistic regresion accuracy: ',acc)
print ('Logistic regression F1-Score: ',f1)
print ('Logistic regression precision: ',precision)
print ('Logistic regression recall: ',recall)
'''

# 4.7 Visually checking output
'''
plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[correct].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[correct], y_test[correct]))
    
plt.figure()
for i, incorrect in enumerate(incorrect_indices[:9]):
    plt.subplot(3,3,i+1)
    plt.imshow(X_test[incorrect].reshape(28,28), cmap='gray', interpolation='none')
    plt.title("Predicted {}, Class {}".format(predicted_classes[incorrect], y_test[incorrect]))
'''    
    


