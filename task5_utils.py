import random
import numpy as np
from scipy.ndimage.interpolation import rotate
import matplotlib.pyplot as plt
from skimage.util import crop
from scipy.misc import imread
from skimage.transform import resize
import os


def rotate_image(image, angles=range(360)):
    rotation_angle = random.choice(angles)
    rotated_image = rotate(image, rotation_angle, reshape=False)
    return rotated_image, rotation_angle


def rotate_images(images, num_rotated_per_image, angles=range(360)):
    rotated_images = []
    rotation_angles = []
    for image in (images):
        for n in range(num_rotated_per_image):
            image = np.squeeze(image)
            rotated_image, rotation_angle = rotate_image(image, angles)
            rotated_image = rotated_image.reshape(1, rotated_image.shape[0], rotated_image.shape[1])
            rotated_images.append(rotated_image)
            rotation_angles.append(rotation_angle)
    return np.asarray(rotated_images), np.asarray(rotation_angles)


def angle_to_class_label(y, angles=range(360)):
    class_names = {}
    for i, angle in enumerate(angles):
        class_names[angle] = i
    class_labels = [class_names[angle] for angle in y]
    return np.asarray(class_labels)


def class_label_to_angle(y, angles=range(360)):
    class_names = {}
    for i, angle in enumerate(angles):
        class_names[i] = angle
    class_labels = [class_names[i] for i in y]
    return np.asarray(class_labels)


def plot_examples(X, y, y_angles, y_predicted_angles,
                  num_test_images=5, number=None, angle=None, only_errors=False):

    if only_errors:
        mask = np.where(y_predicted_angles != y_angles)[0]
        X = X[mask]
        y = y[mask]
        y_predicted_angles = y_predicted_angles[mask]
        y_angles = y_angles[mask]

    if angle is not None and number is not None:
        indices = np.intersect1d(np.where(y_angles == angle)[0], np.where(y == number)[0])
    elif angle is not None:
        indices = np.where(y_angles == angle)[0]
    elif number is not None:
        indices = np.where(y == number)[0]
    else:
        indices = len(y_angles)

    mask = np.random.choice(indices, num_test_images)
    true_angles = y_angles[mask]
    predicted_angles = y_predicted_angles[mask]

    plt.rcParams['figure.figsize'] = (10.0, 2 * num_test_images)
    fig_number = 0
    for i in range(num_test_images):
        rotated_image = X[mask[i]][0]
        original_image = rotate(rotated_image, -true_angles[i])
        corrected_image = rotate(rotated_image, -predicted_angles[i])

        fig_number += 1
        plt.subplot(num_test_images, 3, fig_number)
        plt.imshow(original_image)

        fig_number += 1
        plt.subplot(num_test_images, 3, fig_number)
        plt.title('Angle: {0}'.format(true_angles[i]))
        plt.imshow(rotated_image)

        fig_number += 1
        plt.subplot(num_test_images, 3, fig_number)
        reconstructed_angle = angle_difference(predicted_angles[i], true_angles[i])
        plt.title('Angle: {0}'.format(reconstructed_angle))
        plt.imshow(corrected_image)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)


def plot_error_hist(y, y_angles, y_predicted_angles, error_type='angle'):
    error_per_class = []
    classes = range(np.max(y) + 1)
    for i in classes:
        mask = np.where(y == i)
        y_test_angle_i = y_angles[mask]
        y_predicted_angles_i = y_predicted_angles[mask]
        if error_type == 'classification':
            error_per_class.append((float(np.sum(y_test_angle_i != y_predicted_angles_i)) /
                                    len(y_predicted_angles_i)) * 100)
        elif error_type == 'angle':
            error_per_class.append(float(np.mean(abs(angle_difference(y_test_angle_i, y_predicted_angles_i)))))

    plt.rcParams['figure.figsize'] = (10.0, 10.0)
    ax = plt.subplot(111)
    ax.bar(classes, error_per_class, align='center')
    ax.set_xticks(classes)


def angle_difference(x, y):
    return 180 - abs(abs(x - y - 180))


def open_and_resize(im, size):
    h, w, c = im.shape
    if h < w:
        before_n = (w - h) / 2
        after_n = w - (h + before_n)
        im_cropped = crop(im, ((0, 0), (before_n, after_n), (0, 0)))
    else:
        before_n = (h - w) / 2
        after_n = h - (w + before_n)
        im_cropped = crop(im, ((before_n, after_n), (0, 0), (0, 0)))
    im_cropped = resize(im_cropped, (size, size, c))
    return im_cropped


def load_and_resize_images(folder, size):
    images = []
    image_paths = os.listdir(folder)
    for image_path in image_paths:
        im = imread(os.path.join(folder, image_path))
        images.append(open_and_resize(im, size))
    return np.asarray(images)
