# Cat vs. Dog Image Classification
# Exercise 3: Feature Extraction and Fine-Tuning
# Estimated completion time: 30 minutes
# In Exercise 1, we built a convnet from scratch,and were able to achieve an accuracy of
# about 70%. With the addition of data augmentation and dropout in Exercise 2, we were able
# to increase accuracy to about 80%. That seems decent, but 20% is still to high of an error
# rate. Maybe we just don't have enough training data available to properly solve the problem.
# What other approaches can we try?
# In this exercise, we'll look at two techniques for repurposing feature data generated from
# image models that have already been trained on large sets of data, feature extraction and
# fine tuning, and use them to improve the accuracy of our cat vs. dog classification model.

# Feature Extraction Using a Pretrained Model
# One thing that is commonly done in computer vision is to take a model trained on a very
# large dataset, run it on your own, smaller dataset, and extract the intermediate
# representations (features) that the model generates. These representations are frequently
# informative for your own computer vision task, even though the task may be quite different
# from the problem that the original model was trained on. This versatility and repurposability
# of convnets is one of the most interesting aspects of deep learning.
# In our case, we will use the Inception V3 model developed at Google, and pre-trained on
# ImageNet, a large data set of web images (1.4M images and 1000 classes). This is a
# powerful model; let's see what the features that it has learned can do for our cat vs. dog
# problem.
# First, we need to pick which intermediate layer of Inception V3 we will use for feature
# extraction. A common practice is to use the output of the very last layer before the Flatten
# operation, the so-called "bottleneck layer". The reasoning here is that the following fully
# connected layers will be too specialized for the task the network was trained on, and thus
# the features learned by these layers won't be very useful for a new task. The bottleneck
# features, however, retain much generality.
# Let's instantiate an Inception V3 model preloaded with weights trained on ImageNet:

import os

from tensorflow.keras import layers
from tensorflow.keras import Model
# New let's download the weights:
# !wget --no-check-certificate \
#     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
#     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = "../data/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3), include_top=False, weights=None
)
pre_trained_model.load_weights(local_weights_file)
# By specifying the include_top=False argument, we load a network that doesn't include the
# classification layers at the top-ideal for feature extraction.

# Let's make the model non-trainable, since we will only use it for feature extraction; we won't
# update the weights of the pretrained model during training.

for layer in pre_trained_model.layers:
    layer.trainable = False

# The layer we will use for feature extraction in Inception v3 is called mixed7. It is not the
# bottleneck of the network, but we are using it to keep a sufficiently large feature map (7x7 in
# this case). (Using the bottleneck layer would have resulting in a 3x3feature map, which is a
# bit small.) Let's get the output from mixed7.

last_layer = pre_trained_model.get_layer("mixed7")
print("last layer output shape:", last_layer.output_shape)
last_output = last_layer.output

# Now let's stick a fully connected classifier on top of last_output:
from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation="relu")(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation="sigmoid")(x)

# Configure and compile the model
model = Model(pre_trained_model.input, x)
model.compile(loss="binary_crossentropy",
              optimizer=RMSprop(lr=0.0001),
              metrics=["acc"])

# For examples and data preprocessing, let's use the same files and train_generator as we
# did in Exercise 2.
# NOTE: The 2,000 images used in this exercise are excerpted from the "Dogs vs. Cats"
# dataset available on Kaggle, which contains 25,000 images. Here we use a subset of full
# dataset to decrease training time for educational purposes.

# !wget --no-check-certificate \
#    https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip -O \
#    /tmp/cats_and_dogs_filtered.zip

import os
import zipfile

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# local_zip = '/tmp/cats_and_dogs_filtered.zip'
# zip_ref = zipfile.ZipFile(local_zip, 'r')
# zip_ref.extractall('/tmp')
# zip_ref.close()

# Define our example directories and files
base_dir = '../data/cats_and_dogs_filtered'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# Directory with our training cat pictures
train_cats_dir = os.path.join(train_dir, 'cats')

# Directory with our training dog pictures
train_dogs_dir = os.path.join(train_dir, 'dogs')

# Directory with our validation cat pictures
validation_cats_dir = os.path.join(validation_dir, 'cats')

# Directory with our validation dog pictures
validation_dogs_dir = os.path.join(validation_dir, 'dogs')

train_cat_fnames = os.listdir(train_cats_dir)
train_dog_fnames = os.listdir(train_dogs_dir)

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Note that the validation data should not be augmentation!
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode="binary"
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode="binary"
)

# Finally, let's train the model using the features we extracted. We'll train on all 2000 images
# available, for 2 epochs, and validate on all 1,000 test images.

# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=2,
#     validation_data=validation_generator,
#     validation_steps=50,
#     verbose=2
# )

# Result:
# Epoch 1/2
#  - 99s - loss: 0.5079 - acc: 0.7485 - val_loss: 0.3501 - val_acc: 0.9080
# Epoch 2/2
#  - 104s - loss: 0.3952 - acc: 0.8285 - val_loss: 0.7386 - val_acc: 0.8600
# You can see that we reach a validation accuracy of 88-90% very quickly. This is much better
# than the small model we trained from scratch.

# Futher Improving Accuracy with Fine-Tuning
# In our feature-extraction experiment, we only tried adding two classification layers on top of
# an Inception V3 layer. The weights of pretrained network were not updated during training.
# One way to increase performance even further is to "fine-tune" the weights of the top
# layers of the pretrained model alongside the training of the top-level classifier. A couple of
# important notes on fine-tuning:
#     1,Fine-tuning should only be attempted after you have trained the top-level classifier
#       with the pretrained model set to non-trainable. If you add a randomly initialized
#       classifier on top of a pretrained model and attempt to train all layers jointly, the
#       magnitude of the gradient updates will be too large (due to the random weights from
#       the classifier), and your pretrained model will just forget everything it has learned.
#     2,Additionally, we fine-tune only the top layers of the pre-trained model rather than all
#       layers of the pretrained model because, in a convnet, the higher up a layer is ,the more
#       specialized it is. The first few layers in a convnet learn very simple and generic
#       features, which generalize to almost all types of images. But as you go higher up, the
#       the features are increasingly specific to the dataset that the model is trained on. The goal
#       of fine-tuning is to adapt these specialized features to work with the new dataset.
# All we need to do to implement fine-tuning is to set the top layers of Inception V3 to be
# trainable, recompile the model (necessary for these changes to take effect), and resume
# training. Let's unfreeze all layers belonging to the mixed7 modeule-i.e., all layers found after
# mixed6-and recompile the model:
from tensorflow.keras.optimizers import SGD

unfreeze = False
# Unfreeze all models after "mixed6"
for layer in pre_trained_model.layers:
    if unfreeze:
        layer.trainable = True
    if layer.name == "mixed6":
        unfreeze = True

# As an optimizer, here we will use SGD
# with a very low learning rate (0.00001)
model.compile(loss="binary_crossentropy",
              optimizer=SGD(
                  lr=0.00001,
                  momentum=0.9),
              metrics=["acc"])

# Now let's retrain the model. We'll train on all 2000 images available, for 50 epochs, and
# validate on all 1,000 test images. (This may take 15-20 minutes to run.)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=50,
    verbose=2
)

# We are seeing a nice improvement, with the validation loss going from ~1.7 down to ~1.2,
# and accuracy going from 88% to 92%. That's a 4.5% relative improvement in accuracy.
# Let's plot the training and validation loss and accuracy to show it conclusively:

import matplotlib.pyplot as plt
import matplotlib.image as mping

# Retrieve a list of accuracy results on training and test data
# sets for each training epoch
acc = history.history["acc"]
val_acc = history.history["val_acc"]

# Retrieve a list of list results on training and test data
# sets for each training epoch
loss = history.history["loss"]
val_loss = history.history["val_loss"]

# Get number of epochs
epochs = range(len(acc))

# Plot training and validation accuracy per epoch
plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title("Training and validation accuracy")
plt.show()


plt.figure()

# Plot training and validation loss per epoch
plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title("Training and validation loss")
plt.show()

# Congratulations! Using feature extracting and fine-tuning, you're built an image
# classification model that can identify cats vs. dogs in images with over 90% accuracy.
