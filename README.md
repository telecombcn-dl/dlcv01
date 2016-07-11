# Project Work at DLCV

This is the project repository for the group 1 at the DLCV. The Team is made up by:

| ![Javier de la Rica][image-javier] | ![Eva Julian][image-eva] | ![Alberto Montes][image-alberto] | ![Andrés Rojas][image-andres] | ![Daniel Saez][image-daniel] |
| :---: | :---: | :---: | :---: | :---: |
| Javier de la Rica | Eva Julian | Alberto Montes | Andrés Rojas | Daniel Saez |

It is going to be explained below what has been done during the [Deep Learning for Computer Vison](http://TelecomBCN.DeepLearning.Barcelona) course at UPC at Summer 2016.

![Universitat Politècnica de Catalunya][image-upc-logo]

# Task 1

The objective of the first task is to create a neuronal network and to try different layers and different amount of them to observe the behavior and causes of applying one kind of layer or another, playing with the amount of neurons and the kind of activation we apply to them.


# Task 2

The main aim of this second task is to train our own neuronal networks with some different input databases so that we can study the stage of training a neuronal network trying to overfitting it, increasing the data with augmentation of the input database or regularizing the network by using different values of drop out. This may lead us to some different results in terms of accuracy and loss, so we can conclude which metric and values are the most accurate.

# Task 3

For the visualization task it was trained a network with the MNIST dataset. On the one hand, it was trained with the original input dataset, and on the other hand, we tried to train it with an input data with gaussian noise added, so that we are able to compare the results obtained and study if the noise does actually affect to the result.


# Task 4



# Task 5

In this task we wanted to train a neural network to detect the rotation angle of an image. This can for example be used to automatically straighten photos that have been taken with a camera. We used a simple architecture with 1 convolutional layer and 2 fully connected layers. The last layer contains 360 neurons to classify all the possible rotation angles. We use the angle error instead of the classification error because we wanted to measure how close the true angle and the predicted angles are, (the classification error would only help us evaluate when the predicted angle and the true angle are exactly the same).




<!--Images-->
[image-javier]: https://raw.githubusercontent.com/telecombcn-dl/dlcv01/master/misc/javier_rica.jpeg "Javier de la Rica"
[image-eva]: https://raw.githubusercontent.com/telecombcn-dl/dlcv01/master/misc/eva_julian.jpg "Eva Julian"
[image-alberto]: https://raw.githubusercontent.com/telecombcn-dl/dlcv01/master/misc/alberto_montes.jpg "Alberto Montes"
[image-andres]: https://raw.githubusercontent.com/telecombcn-dl/dlcv01/master/misc/andres_rojas.jpg "Andrés Rojas"
[image-daniel]: https://raw.githubusercontent.com/telecombcn-dl/dlcv01/master/misc/daniel_saez.jpg "Daniel Saez"

[image-upc-logo]: https://raw.githubusercontent.com/telecombcn-dl/dlcv01/master/misc/upc_etsetb.jpg
