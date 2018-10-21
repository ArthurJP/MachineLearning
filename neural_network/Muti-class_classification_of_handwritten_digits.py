#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import glob
import math
import os

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

__arthur__ = "张俊鹏"

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

mnist_dataFrame = pd.read_csv("../data/mnist_train_small.csv", sep=",", header=None)

# Use just the first 10,000 records for training/validation.
mnist_dataFrame = mnist_dataFrame.head(10000)

mnist_dataFrame = mnist_dataFrame.reindex(np.random.permutation(mnist_dataFrame.index))


# print(mnist_dataFrame.head)
# 这里都是像素点，只有0和1


def parse_labels_and_features(dataset):
    """Extracts labels and features.

        This is a good place to scale or transform the features if needed.

    Args:
        dataset: A Pandas `Dataframe`, containing the label on the first column and
        monochrome pixel values on the remaining columns, in row major order.
    Returns:
        A `tuple` `(labels, features)`:
        labels: A Pandas `Series`.
        features: A Pandas `DataFrame`.
    """
    labels = dataset[0]  # 第一位保存了已经标记的数值值

    # DataFrame.loc index ranges are inclusive at both ends.
    features = dataset.loc[:, 1:784]  # 将数据每784（28x28）分为一块
    # Scale the data to [0, 1] by dividing out the max value, 255, from [0, 255].
    features = features / 255  # 将图像的值从0-255变为0-1

    return labels, features


training_targets, training_examples = parse_labels_and_features(mnist_dataFrame[:7500])
print(training_examples.describe())
print(training_targets.describe())

validation_targets, validation_examples = parse_labels_and_features(mnist_dataFrame[7500:10000])
print(validation_targets.describe())
print(validation_targets.describe())

random_example = np.random.choice(training_examples.index)
_, ax = plt.subplots()
ax.matshow(training_examples.loc[random_example].values.reshape(28, 28))
ax.set_title("Label: %i" % training_targets.loc[random_example])
ax.grid(False)  # 取消栅格
plt.show()


# 任务 1：为 MNIST 构建线性模型
# 首先，我们创建一个基准模型，作为比较对象。LinearClassifier 可提供一组 k 类一对多分类器，每个类别（共 k 个）对应一个分类器。
# 您会发现，除了报告准确率和绘制对数损失函数随时间变化情况的曲线图之外，我们还展示了一个混淆矩阵。混淆矩阵会显示错误分类为其他类别的类别。哪些数字相互之间容易混淆？
# 另请注意，我们会使用 log_loss 函数跟踪模型的错误。不应将此函数与用于训练的 LinearClassifier 内部损失函数相混淆。

def construct_feature_columns():
    """Construct the TensorFlow Feature Columns

    Returns:
        A set of feature columns
    """

    # There are 784 pixels in each image.
    return set([tf.feature_column.numeric_column("pixels", shape=784)])


# 在本次练习中，我们会对训练和预测使用单独的输入函数，并将这些函数分别嵌套在 create_training_input_fn() 和 create_predict_input_fn() 中，这样一来，我们就可以调用这些函数，以返回相应的 _input_fn，并将其传递到 .train() 和 .predict() 调用。

def create_training_input_fn(features, labels, batch_size, num_epoches=None, shuffle=True):
    """A custom input_fn for sending MNIST data to the estimator for training.

    Args:
        features: The training features.
        labels: The training labels.
        batch_size: Batch size to use during training.

    Returns:
        A function that returns batches of training features and labels during
        training.
    """

    def _input_fn(num_epochs=None, shuffle=True):
        # Input pipelines are reset with each call to .train(). to ensure model
        # gets a good sampling of data, even when number of steps is small, we
        # shuffle all the data before creating the Dataset object
        idx = np.random.permutation(features.index)
        raw_features = {"pixels": features.reindex(idx)}
        raw_targets = np.array(labels[idx])

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))  # warning:2GB limit
        ds = ds.batch(batch_size).repeat(num_epoches)

        if shuffle:
            ds = ds.shuffle(10000)

        # Return the next batch of data.
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn


def create_predict_input_fn(features, labels, batch_size):
    """A custom input_fn for sending mnist data to the estimator for predictions.

    Args:
        features: The features to base predictions on.
        labels: The labels of the prediction examples.

    Returns:
        A function that returns features and labels for predictions.
    """

    def _input_fn():
        raw_features = {"pixels": features.values}
        raw_targets = np.array(labels)

        ds = Dataset.from_tensor_slices((raw_features, raw_targets))  # warning: 2GB limit
        ds = ds.batch(batch_size)

        # Return the next batch of data.
        feature_batch, label_batch = ds.make_one_shot_iterator().get_next()
        return feature_batch, label_batch

    return _input_fn


def train_linear_classification_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear classification model for the MNIST digits dataset.

    In addition to training, this function also prints training progress information,
    a plot of the training and validation loss over time, and a confusion
    matrix.

    Args:
        learning_rate: An `int`, the learning rate to use.
        steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
        batch_size: A non-zero `int`, the batch size.
        training_examples: A `DataFrame` containing the training features.
        training_targets: A `DataFrame` containing the training labels.
        validation_examples: A `DataFrame` containing the validation features.
        validation_targets: A `DataFrame` containing the validation labels.

    Returns:
        The trained `LinearClassifier` object.
    """
    periods = 10

    steps_per_period = steps / periods
    # Create the input functions.
    predict_training_input_fn = create_predict_input_fn(
        training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(
        validation_examples, validation_targets, batch_size)
    training_input_fn = create_training_input_fn(
        training_examples, training_targets, batch_size)

    # Create a LinearClassifier object.
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    classifier = tf.estimator.LinearClassifier(
        feature_columns=construct_feature_columns(),
        n_classes=10,  # 分类结果数，默认为2即二分类
        optimizer=my_optimizer,
        config=tf.estimator.RunConfig(keep_checkpoint_max=1)  # 保存的中间文件数，降低系统负载，默认为5
    )

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss error (on validation data):")
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute probabilities.
        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item["probabilities"] for item in training_predictions])
        training_pred_class_id = np.array([item["class_ids"][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)  # 转化为10个类目的矩阵
        # Converts a class vector (integers) to binary class matrix.
        # Arguments:
        #       y: class vector to be converted into a matrix
        #           (integers from 0 to num_classes).
        #       num_classes: total number of classes.
        #   Returns:
        #       A binary matrix representation of the input. The classes axis is placed
        #       last.

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item["probabilities"] for item in validation_predictions])
        validation_pred_class_id = np.array([item["class_ids"][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

        # Compute training and validation errors.
        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print("Model training finished.")
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, "events.out.tfevents*")))

    # Calculate final predictions (not probabilities, as above).
    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item["class_ids"][0] for item in final_predictions])

    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()

    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class).
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    return classifier


# classifier = train_linear_classification_model(
#     learning_rate=0.03,
#     steps=1000,
#     batch_size=30,
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets)


def train_nn_classification_model(
        learning_rate,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a neural network classification model for the MNIST digits dataset.

    In addition to training, this function also prints training progress information,
    a plot of the training and validation loss over time, as well as a confusion
    matrix.

    Args:
      learning_rate: An `int`, the learning rate to use.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      hidden_units: A `list` of int values, specifying the number of neurons in each layer.
      training_examples: A `DataFrame` containing the training features.
      training_targets: A `DataFrame` containing the training labels.
      validation_examples: A `DataFrame` containing the validation features.
      validation_targets: A `DataFrame` containing the validation labels.

    Returns:
      The trained `DNNClassifier` object.
    """

    periods = 10
    # Caution: input pipelines are reset with each call to train.
    # If the number of steps is small, your model may never see most of the data.
    # So with multiple `.train` calls like this you may want to control the length
    # of training with num_epochs passed to the input_fn. Or, you can do a really-big shuffle,
    # or since it's in-memory data, shuffle all the data in the `input_fn`.
    steps_per_period = steps / periods
    # Create the input functions.
    predict_training_input_fn = create_predict_input_fn(training_examples, training_targets, batch_size)
    predict_validation_input_fn = create_predict_input_fn(validation_examples, validation_targets, batch_size)
    training_input_fn = create_training_input_fn(training_examples, training_targets, batch_size)

    # Create a DNNClassifier object.
    my_optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    classifier = tf.estimator.DNNClassifier(
        feature_columns=construct_feature_columns(),
        n_classes=10,  # 一共有10种数字
        hidden_units=hidden_units,
        optimizer=my_optimizer,
        config=tf.contrib.learn.RunConfig(keep_checkpoint_max=1)
    )

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("LogLoss error (on validation data):")
    training_errors = []
    validation_errors = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        classifier.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )

        # Take a break and compute probabilities.
        training_predictions = list(classifier.predict(input_fn=predict_training_input_fn))
        training_probabilities = np.array([item['probabilities'] for item in training_predictions])
        training_pred_class_id = np.array([item['class_ids'][0] for item in training_predictions])
        training_pred_one_hot = tf.keras.utils.to_categorical(training_pred_class_id, 10)

        validation_predictions = list(classifier.predict(input_fn=predict_validation_input_fn))
        validation_probabilities = np.array([item['probabilities'] for item in validation_predictions])
        validation_pred_class_id = np.array([item['class_ids'][0] for item in validation_predictions])
        validation_pred_one_hot = tf.keras.utils.to_categorical(validation_pred_class_id, 10)

        # Compute training and validation errors.
        training_log_loss = metrics.log_loss(training_targets, training_pred_one_hot)
        validation_log_loss = metrics.log_loss(validation_targets, validation_pred_one_hot)
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, validation_log_loss))
        # Add the loss metrics from this period to our list.
        training_errors.append(training_log_loss)
        validation_errors.append(validation_log_loss)
    print("Model training finished.")
    # Remove event files to save disk space.
    _ = map(os.remove, glob.glob(os.path.join(classifier.model_dir, 'events.out.tfevents*')))

    # Calculate final predictions (not probabilities, as above).
    final_predictions = classifier.predict(input_fn=predict_validation_input_fn)
    final_predictions = np.array([item['class_ids'][0] for item in final_predictions])

    accuracy = metrics.accuracy_score(validation_targets, final_predictions)
    print("Final accuracy (on validation data): %0.2f" % accuracy)

    # Output a graph of loss metrics over periods.
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(training_errors, label="training")
    plt.plot(validation_errors, label="validation")
    plt.legend()
    plt.show()

    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(validation_targets, final_predictions)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class).
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()

    return classifier


classifier = train_nn_classification_model(
    learning_rate=0.05,
    steps=1000,
    batch_size=30,
    hidden_units=[100, 100],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

mnist_test_dataframe = pd.read_csv(
    "../data/mnist_test.csv",
    sep=",",
    header=None)

test_targets, test_examples = parse_labels_and_features(mnist_test_dataframe)
test_examples.describe()

predict_test_input_fn = create_predict_input_fn(
    test_examples, test_targets, batch_size=100)

test_predictions = classifier.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item['class_ids'][0] for item in test_predictions])

accuracy = metrics.accuracy_score(test_targets, test_predictions)
print("Accuracy on test data: %0.2f" % accuracy)

# 任务 3：可视化第一个隐藏层的权重。
# 我们来花几分钟时间看看模型的 weights_ 属性，以深入探索我们的神经网络，并了解它学到了哪些规律。
# 模型的输入层有 784 个权重，对应于 28×28 像素输入图片。第一个隐藏层将有 784×N 个权重，其中 N 指的是该层中的节点数。我们可以将这些权重重新变回 28×28 像素的图片，具体方法是将 N 个 1×784 权重数组变形为 N 个 28×28 大小数组。
# 运行以下单元格，绘制权重曲线图。请注意，此单元格要求名为 "classifier" 的 DNNClassifier 已经过训练。
print(classifier.get_variable_names())

weights0 = classifier.get_variable_value("dnn/hiddenlayer_0/kernel")
print("weights0 shaper:", weights0.shape)

num_nodes = weights0.shape[1]
num_rows = int(math.ceil(num_nodes / 10.0))
fig, axes = plt.subplots(num_rows, 10, figsize=(20, 2 * num_rows))
for coef, ax in zip(weights0.T, axes.ravel()):
    # Weights in coef is reshaped from 1x784 to 28x28.
    ax.matshow(coef.reshape(29, 29), cmap=plt.cm.pink)
    ax.set_xticks(())
    ax.set_yticks(())

plt.show()
