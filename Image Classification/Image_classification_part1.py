#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import zipfile

__arthur__ = "张俊鹏"

# Cat vs. Dog Image Classification
# Exercise 1: Building a Convnet from Scratch
# Estimated completion time: 20 minutes
#
# In this exercise, we will build a classifier model from scratch that is able to distinguish dogs from cats. We will follow these steps:
#
# Explore the example data
# Build a small convnet from scratch to solve our classification problem
# Evaluate training and validation accuracy
# Let's go!

# Explore the Example Data
# Let's start by downloading our example data, a .zip of 2,000 JPG pictures of cats and dogs, and extracting it locally in /data.
#
# NOTE: The 2,000 images used in this exercise are excerpted from the "Dogs vs. Cats" dataset available on Kaggle, which contains 25,000 images. Here, we use a subset of the full dataset to decrease training time for educational purposes.

# local_zip = "../data/cats_and_dogs_filtered.zip"
# zip_ref = zipfile.ZipFile(local_zip, "r")
# zip_ref.extractall("../data/")
# zip_ref.close()

# The contents of the .zip are extracted to the base directory /data/cats_and_dogs_filtered,
# which contains train and validation subdirectories for the training and validation datasets
# (see the Machine Learning Crash Course for a refresher on training, validation, and test sets),
# which in turn each contain cats and dogs subdirectories. Let's define each of these directories:

base_dir = "../data/cats_and_dogs_filtered"
train_dir = os.path.join(base_dir, "train")
validation_dir = os.path.join(base_dir, "validation")

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, "cats")

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, "dogs")

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, "cats")

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, "dogs")

# Now, let's see what the file-names look like in the cats and dogs train directories(file naming conventions are the same in the validation directory):
train_cat_fnames = os.listdir(train_cats_dir)
print(train_cat_fnames[:10])

train_dog_fnames = os.listdir(train_dogs_dir)
print(train_dog_fnames[:10])

# Let's find out the total number of cat and dog images in the train and validation directories.
print("total training cat images:", len(os.listdir(train_cats_dir)))
print("total training dog images:", len(os.listdir(train_dogs_dir)))
print("total validation cat images:", len(os.listdir(validation_cats_dir)))
print("total validation dog images:", len(os.listdir(validation_dogs_dir)))

# For both cats and dogs, we have 1,000 training images and 500 test images.
# Now let's take a look at a few pictures to get a better sense of what the cat and dog dataset look like. First, configure the matplot parameters:

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Parameter for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

# Now, display a batch of 8 cat and 8 dog pictures. You can rerun the cell to see a fresh batch each time:
# Set up matplotlib fig, and size it to fit 4x4 pics.
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_cat_pix = [os.path.join(train_cats_dir, fname)
                for fname in train_cat_fnames[pic_index - 8:pic_index]]
next_dog_pix = [os.path.join(train_dogs_dir, fname)
                for fname in train_dog_fnames[pic_index - 8:pic_index]]

for i, img_path in enumerate(next_cat_pix + next_dog_pix):
    # Set up subplot; subplot indices start at 1
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis("Off")  # Don't show axes (or gridlines)

    img = mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

# Building a Small Convnet from Scratch to Get to 72% Accuracy
# The images that will go into our convnet are 150x150 color images(in the next section on Data Preprocessing, we'll add handing to resize all the image to
# 150x150 before feeding them into the neural network).
# Let's code up the architecture. We will stack 3 {convolution+relu+maxpooling} modules.Our convolutions operate on 3x3 windows and our maxpooling layers
# operate on 2x2 windows. Our first convolution extracts 16 filters, the following one extracts 32 filters, and the last one extracts 64 filters.
# NOTE: This a configuration that is widely used and known to work well for image classification. Also, Since we have relatively few training examples(1,000),
# using just three convolutional modules keeps the model small, which lowers the risk of overfitting(which we'll explore in more depth in Exercise 2).

from tensorflow.keras import layers
from tensorflow.keras import Model

# Our input feature map is 150x150x3: 150x150 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = layers.Input(shape=(150, 150, 3))

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation="relu")(img_input)
# Arguments:
#       filters: Integer, the dimensionality of the output space
#           (i.e. the number of output filters in the convolution).
#       kernel_size: An integer or tuple/list of 2 integers, specifying the
#           height and width of the 2D convolution window.
#           Can be a single integer to specify the same value for
#           all spatial dimensions.
#       strides: An integer or tuple/list of 2 integers,
#           specifying the strides of the convolution along the height and width.
#           Can be a single integer to specify the same value for
#           all spatial dimensions.
#           Specifying any stride value != 1 is incompatible with specifying
#           any `dilation_rate` value != 1.
#       padding: one of `"valid"` or `"same"` (case-insensitive).
#       data_format: A string,
#           one of `channels_last` (default) or `channels_first`.
#           The ordering of the dimensions in the inputs.
#           `channels_last` corresponds to inputs with shape
#           `(batch, height, width, channels)` while `channels_first`
#           corresponds to inputs with shape
#           `(batch, channels, height, width)`.
#           It defaults to the `image_data_format` value found in your
#           Keras config file at `~/.keras/keras.json`.
#           If you never set it, then it will be "channels_last".
#       dilation_rate: an integer or tuple/list of 2 integers, specifying
#           the dilation rate to use for dilated convolution.
#           Can be a single integer to specify the same value for
#           all spatial dimensions.
#           Currently, specifying any `dilation_rate` value != 1 is
#           incompatible with specifying any stride value != 1.
#       activation: Activation function to use.
#           If you don't specify anything, no activation is applied
#           (ie. "linear" activation: `a(x) = x`).
#       use_bias: Boolean, whether the layer uses a bias vector.
#       kernel_initializer: Initializer for the `kernel` weights matrix.
#       bias_initializer: Initializer for the bias vector.
#       kernel_regularizer: Regularizer function applied to
#           the `kernel` weights matrix.
#       bias_regularizer: Regularizer function applied to the bias vector.
#       activity_regularizer: Regularizer function applied to
#           the output of the layer (its "activation")..
#       kernel_constraint: Constraint function applied to the kernel matrix.
#       bias_constraint: Constraint function applied to the bias vector.
x = layers.MaxPooling2D(2)(x)
# Arguments:
#       pool_size: integer or tuple of 2 integers,
#           factors by which to downscale (vertical, horizontal).
#           (2, 2) will halve the input in both spatial dimension.
#           If only one integer is specified, the same window length
#           will be used for both dimensions.
#       strides: Integer, tuple of 2 integers, or None.
#           Strides values.
#           If None, it will default to `pool_size`.
#       padding: One of `"valid"` or `"same"` (case-insensitive).
#       data_format: A string,
#           one of `channels_last` (default) or `channels_first`.
#           The ordering of the dimensions in the inputs.
#           `channels_last` corresponds to inputs with shape
#           `(batch, height, width, channels)` while `channels_first`
#           corresponds to inputs with shape
#           `(batch, channels, height, width)`.
#           It defaults to the `image_data_format` value found in your
#           Keras config file at `~/.keras/keras.json`.
#           If you never set it, then it will be "channels_last".


# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)

# On top of it we stick two fully-connected layers. Because we are facing a two-class classification problem, i.e. a binary classification problem, we will end our network with a sigmoid activation, so that the output of our network will be a single scalar between 0 and 1, encoding the probability that the current image is class
# 1(as opposed to class 0).

# Flatten feature map to a 1-dim tensor so we can add fully connected layers
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation="relu")(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation="sigmoid")(x)

# Create model:
# input = input feature map
# output = input feature map + stacked convolution/maxpooling layers + fully
# connected layer + sigmoid output layer
model = Model(img_input, output)

# Let's summarize the model architecture.
model.summary()

# The "output shape" column shows how the size of your feature map evolves in each successive layer. The convolution layers reduce the size of the feature maps by
# a bit due to padding, and each pooling layer halves the feature map.

# Next, we'll configure the specifications for model training. We will train our model with the binary_crossentropy loss, because it's a binary classification problem
# and our final activation is a sigmoid.(For a refresher on loss metrics, see the Machine Learning Crash Course.) We will use the rmsprop optimizer with a learning
# rate of 0.001. During training, we will want to monitor classification accuracy.
# NOTE: In this case, using the RMSprop optimization algorithm is preferable to stochastic gradient descent(SGD), because RMSprop automates learning-rate tuning
# for us.(Other optimizers, such as Adam and Adagrad, also automatically adapt the learning rate during training, and would work equally well here.)

from tensorflow.keras.optimizers import RMSprop

model.compile(loss="binary_crossentropy",
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

# Data preprocessing
# Let's set up data generators that will read pictures in our source folders, convert them to float32 tensors, and feed them(with their labels) to our network. We'll
# have one generator for the training images and one for the validation images. Our generators will yield batches of 20 images of size 150x150 and their labels(binary).
# As you may already know, data that goes into neural networks should usually be normalized in some way to make it more amenable to processing by the network.
# (It is uncommon to feed raw pixels into a convnet.) In our case, we will preprocess our images by normalizing the pixel values to be in the [0,1] range(originally
# all value are in the [0,255] range).
# In Keras this can be done via the keras.preprocessing.image.ImageDataGenerator class using the rescale parameter. This ImageDataGenerator class allows you
# to instantiate generators of augmented image batches (and their labels) via .flow(data, labels) or .flow_from_directory(directory). These generators can
# then be used with the Keras model methods that accept data generators as inputs: fit_generator, evaluate_generator, and predict_generator.

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1. / 255)
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All images will be resized to 150x150
    batch_size=20,
    # Since we use binary_crossentropy loss, we need binary labels
    class_mode="binary"
)

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode="binary"
)

# Training
# Let's train on all 2,000 images available, for 15 epochs, and validate on all 1,000 test images.(This may take a few minutes on run.)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,  # 2000 images = batch_size * steps
    epochs=15,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2
)

# Visualizing Intermediate Representations
# To get a feel for what kind of features our convnet has learned, one fun thing to do is to visualize how an input gets transformed as it goes through the convnet.
# Let's pick a random cat or dog image from the training set, and then generate a figure where each row is the output of a layer, and each image in the row is a
# specific filter in that output feature map. Return this cell to generate intermediate representations for a variety of training images.

import numpy as np
import random
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Ignoring error, the code can be proceed.
np.seterr(divide='ignore', invalid='ignore')

# Let's define a new Model that will take an image as input, and will output
# intermediate representations for all layers in the previous model after the first.
successive_outputs = [layer.output for layer in model.layers[1:]]
visualization_model = Model(img_input, successive_outputs)

# Let's prepare a random input image of a cat or dog from the training set.
cat_img_files = [os.path.join(train_cats_dir, f) for f in train_cat_fnames]
dog_img_files = [os.path.join(train_dogs_dir, f) for f in train_dog_fnames]
img_path = random.choice(cat_img_files + dog_img_files)

img = load_img(img_path, target_size=(150, 150))  # this is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# Rescale by 1/255
x /= 255

# Let's run our image through our network, thus obtaining all
# intermediate representations for this image.
successive_feature_maps = visualization_model.predict(x)

# These are the names of the layers, so can have them as part of our plot
layer_names = [layer.name for layer in model.layers]

# Now let's display our representations.
for layer_name, feature_map in zip(layer_names, successive_feature_maps):
    if len(feature_map.shape) == 4:
        # Just do this for the conv / maxpool layers, not the fully-connected layers
        n_features = feature_map.shape[-1]  # number of features in feature map
        # The feature map has shape (1, size, size, n_features)
        size = feature_map.shape[1]
        # We will tile our images in this matrix
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
            # Postprocess the feature to make it visually palatable
            x = feature_map[0, :, :, i]  # change what feature_map refer to
            x -= x.mean()
            x /= x.std()
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype("uint8")
            # We'll tile each filter into this big horizontal grid
            display_grid[:, i * size: (i + 1) * size] = x
        # Display the grid
        scale = 20. / n_features
        plt.tight_layout()
        plt.figure(figsize=(scale * n_features, scale))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect="auto", cmap="viridis")

plt.show()
# As you can see we go from the raw pixels of the images to increasingly abstract and compact representations.
# The representations downstream start highlighting what the network pays attention to, and they show fewer and
# fewer features being "activated"; most are set to zero. This is called "sparsity". Representation sparsity is
# a key feature of deep learning.
# These representations carry increasingly less information about the original pixels of the image, but increasingly
# refined information about the class of the image. You can think of a convnet(or a deep network in general) as
# an information distillation pipeline.


# Evaluating Accuracy and Loss for the Model
# Let's plot the training/validation accuracy and loss as collected during training.

# Retrieve a list of accuracy results on training and test data sets for each training epoch
acc = history.history["acc"]
val_acc = history.history["val_acc"]

# Retrieve a list of list results on training and test data sets for each training epoch.
loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title("Training and validation accuracy")

plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title("Training and validation loss")

plt.show()

# As you can see, we are overfitting like it's getting out of fashion. Our training accuracy (in
# blue) gets close to 100%(!) while our validation accuracy (in green) stalls as 70%. Our
# validation loss reaches its minimum after only five epochs.
# Since we have a relatively small number of training examples(2000), overfitting should be
# our number one concern. Overfitting happens when a model exposed to too few examples learns
# patterns that do not generalize to new data, i.e. when the model starts using irrelevant
# features for making predictions. For instance, if you, as a human, only see three images of
# people who are lumberjacks, and three images of people who are sailors, and among them the
# only person waring a cap is a lumberjack, you might start thinking that waring a cap is
# a sign of being a lumberjack as opposed to a sailor. You would then make a pretty lousy
# lumberjack/sailor classifier.
# Overfitting is the central problem in machine learning: given that we are fitting the parameters
# of our model to a given dataset, how can we make sure that the representations learned by
# the model will be applicable to data never seen before? How do we avoid learning things that
# are specific to the training data?
# In the next exercise, we'll look at ways to prevent overfitting in the cat vs. dog classification model.


# Clean Up
# Before running the net exercise, run the following cell to terminate the kernel and free memory resources.
import os,signal
os.kill(os.getpid(),signal.SIGKILL)