# Cat vs. Dog Image Classification
# Exercise 2: Reducing Overfitting
# Estimated completion time:30 minutes
# In this notebook we will build on the model we created in Exercise 1 to classify cats vs.
# dogs, and improve accuracy by employing a couple strategies to reduce overfitting: data augmentation
# and dropout

# We will follow these steps:
# 1.Explore how data augmentation works by making random transformations to training images.
# 2.Add data augmentation to our data preprocessing.
# 3.Add dropout to the convnet.
# 4.Retrain the model and evaluate loss and accuracy.
# Let's get started!

# Exploring Data Augmentation
# Let's get familiar with the concept of data augmentation, an essential way to fight overfitting
# for computer vision models.

# In order to make the most of our few training examples, we will "augment" them via a number of
# random transformations, so that at training time, our model will never see the exact same picture
# twice. This helps prevent overfitting and helps prevent overfitting and helps the model generalize
# better.

# This can be done by configuring a number of random transformations to be performed on the images read
# by our ImageDataGenerator instance. Let's get started with an example:

from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
# These are just a few of the options available (for more, see the Keras documentation. Let's
#  quickly go over what we just wrote:
#     rotation_range is a value in degrees (0-180), a range within which to randomly rotate pictures.
#     width_shift and height_shift are ranges (as a fraction of total width or height)
#       within which to randomly translate pictures vertically or horizontally.
#     shear_image is randomly applying shearing transformations.
#     zoom_range is for randomly inside pictures.
#     horizontal_flip is for randomly flipping half of the images horizontally. This is relevant
#       when there are no assumptions of horizontal asymmetry (e.g. real-world pictures).
#     fill_mode is the strategy used for filling in newly created pixels, which can appear
#       after a rotation or a width/height shift.
# Let's take a look at our augmented images.First let's set up our example files, as in Exercise 1.

# NOTE: The 2,000 images used in this exercise are excerpted from the "Dogs vs.Cats" dataset
# available on Kaggle, which contains 25,000 images. Here, we use a subset of the full dataset to
# decrease training time for educational purposes.

# !wget --no-check-certificate \
#    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip -O \
#    /tmp/cats_and_dogs_filtered.zip

import os
import zipfile

# local_zip = "../data/cats_and_dogs_filtered.zip"
# zip_ref = zipfile.ZipFile(local_zip, "r")
# zip_ref.extractall("../data/")
# zip_ref.close()

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

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

# Next, let's apply the datagen transformations to a cat image from the training set to produce
# five random variants. Rerun the cell a few times to see fresh batches of random variants.

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

img_path = os.path.join(train_cats_dir, train_cat_fnames[2])
img = load_img(img_path, target_size=(150, 150))  # This is a PIL image
x = img_to_array(img)  # Numpy array with shape (150, 150, 3)
x = x.reshape((1,) + x.shape)  # Numpy array with shape (1, 150, 150, 3)

# The .flow() command below generates batches of randomly transformed images
# It will loop indefinitely, so we need to `break` the loop at some point!

# Those code for see what ImageDataGenerator can deliver
# i = 0
# for batch in datagen.flow(x, batch_size=1):
#     plt.figure(i)
#     imgplot = plt.imshow(array_to_img(batch[0]))
#     i += 1
#     plt.show()
#     if i % 5 == 0:
#         break

# Add Data Augmentation to the Preprocessing Step
# Now let's add our data-augmentation transformations from Exploring Data Augmentation to
# our data preprocessing configuration:

# Adding rescale, rotation_range, width_shift_range, height_shift_range,
# shear_range, zoom_range, and horizontal flip to our ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator(rescale=1. / 255)
# Flow training images in batches of 32 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
    train_dir,  # This is the source directory for training images
    target_size=(150, 150),  # All iamges will be resized to 150x150
    batch_size=20,
    class_mode="binary"  # Since we use binary_crossentropy loss, we need binary labels
)
# Flow validation images in batches of 32 using test_datagen generator
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode="binary"
)
# If we train a new network using this data augmentation configuration, our network will never
# see the same input twice. However the inputs that is sees are still heavily intercorrelated, so
# this might not be quite enough to completely get rid of overfitting.

# Adding Dropout
# Another popular strategy for fighting overfitting is to use dropout.
# TIP: To learn more about dropout, see Training Neural Networks in Machine Learning Crash Course.
# Let's reconfigure our convnet architecture from Exercise 1 to add some dropout, right before
# the final classification layer:

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop

# Our input feature map is 150x150x3; 150x150 for the image pixels, and 3 for
# the three color channels: R, G, and B
img_input = layers.Input(shape=(150, 150, 3))

# First convolution extracts 16 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(16, 3, activation="relu")(img_input)
x = layers.MaxPooling2D(2)(x)

# Second convolution extracts 32 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)

# Third convolution extracts 64 filters that are 3x3
# Convolution is followed by max-pooling layer with a 2x2 window
x = layers.Convolution2D(64, 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)

# Flatten feature map to a 1-dim tensor
x = layers.Flatten()(x)

# Create a fully connected layer with ReLU activation and 512 hidden units
x = layers.Dense(512, activation="relu")(x)

# Add a dropout rate of 0.5
x = layers.Dropout(0.5)(x)

# Create output layer with a single node and sigmoid activation
output = layers.Dense(1, activation="sigmoid")(x)

# Configure and compile the model
model = Model(img_input, output)
model.compile(loss="binary_crossentropy",
              optimizer=RMSprop(lr=0.001),
              metrics=["acc"])

# Retrain the Model
# With data augmentation and dropout in place, let's retrain our convnet model. This time, let's
# train on all 2,000 images available, for 30 epochs, and validate on all 1,000 test images.
# (This may take a few minutes to run.)
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2
)
# Note that with data augmentation in place, the 2,000 training images are randomly
# transformed each time a new training epoch runs, which means that the model will never
# see the same image twice during training.

# Evaluate the Results
# Let's evaluate the results of model training with data augmentation and dropout:

# Retrieve a list of accuracy results on training and test data sets for each training epoch
acc = history.history["acc"]
val_acc = history.history["val_acc"]

# Retrieve a list of list of list results on training and test data set for each training epoch
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

# Result:
# Epoch 29/30
#  - 28s - loss: 0.5344 - acc: 0.7430 - val_loss: 0.5052 - val_acc: 0.7580
# Epoch 30/30
#  - 28s - loss: 0.5194 - acc: 0.7360 - val_loss: 0.4941 - val_acc: 0.7530
# Much better! We are no longer overfitting, and we have gained ~3 validation accuracy
# percentage points (see the green line in the top chart). In fact, judging by our training profile,
# we could keep fitting our model for 30+ more epochs and we could probably get to ~80%!

# Clean Up for linux
# Before running the next exercise, run the following cell to terminate the kernel and free memory resources:
# import os, signal
# os.kill(os.getpid(), signal.SIGKILL)
