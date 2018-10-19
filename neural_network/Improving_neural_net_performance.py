#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 提高神经网络性能
# 学习目标：通过将特征标准化并应用各种优化算法来提高神经网络的性能
# 注意：本练习中介绍的优化方法并非专门针对神经网络；这些方法可有效改进大多数类型的模型。

from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

__arthur__ = "张俊鹏"

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataFrame = pd.read_csv("../data/california_housing_train_for_tensorFlow.csv", sep=",")

california_housing_dataFrame = california_housing_dataFrame.reindex(
    np.random.permutation(california_housing_dataFrame.index))


def preprocess_features(california_housing_dataFrame):
    """Prepares input features from California housing data set.

    Args:
      california_housing_dataFrame: A Pandas DataFrame expected to contain data
        from the California housing data set.
    Returns:
      A DataFrame that contains the features to be used for the model, including
      synthetic features.
    """
    selected_features = california_housing_dataFrame[
        ["latitude",
         "longitude",
         "housing_median_age",
         "total_rooms",
         "total_bedrooms",
         "population",
         "households",
         "median_income"]]
    processed_features = selected_features.copy()
    # Create a synthetic feature.
    processed_features["rooms_per_person"] = (
            california_housing_dataFrame["total_rooms"] /
            california_housing_dataFrame["population"])
    return processed_features


def preprocess_targets(california_housing_dataFrame):
    """Prepares target features (i.e., labels) from California housing data set.

    Args:
      california_housing_dataFrame: A Pandas DataFrame expected to contain data
        from the California housing data set.
    Returns:
      A DataFrame that contains the target feature.
    """
    output_targets = pd.DataFrame()
    # Scale the target to be in units of thousands of dollars.
    output_targets["median_house_value"] = (
            california_housing_dataFrame["median_house_value"] / 1000.0)
    return output_targets


# Choose the first 12000 (out of 17000) examples for training.
training_examples = preprocess_features(california_housing_dataFrame.head(12000))
training_targets = preprocess_targets(california_housing_dataFrame.head(12000))

# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = preprocess_features(california_housing_dataFrame.tail(5000))
validation_targets = preprocess_targets(california_housing_dataFrame.tail(5000))

# Double-check that we've done the right thing.
print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())

print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())


# 训练神经网络
# 接下来，我们将训练神经网络。

def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

     Args:
       input_features: The names of the numerical input features to use.
     Returns:
       A set of feature columns
     """
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a neural network model.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key: np.array(value) for key, value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


def train_nn_regression_model(
        my_optimizer,
        steps,
        batch_size,
        hidden_units,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a neural network regression model.

    In addition to training, this function also prints training progress information,
    as well as a plot of the training and validation loss over time.

    Args:
      my_optimizer: An instance of `tf.train.Optimizer`, the optimizer to use.
      steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
      batch_size: A non-zero `int`, the batch size.
      hidden_units: A `list` of int values, specifying the number of neurons in each layer.
      training_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataFrame` to use as input features for training.
      training_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataFrame` to use as target for training.
      validation_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataFrame` to use as input features for validation.
      validation_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataFrame` to use as target for validation.

    Returns:
      A tuple `(estimator, training_losses, validation_losses)`:
        estimator: the trained `DNNRegressor` object.
        training_losses: a `list` containing the training loss values taken during training.
        validation_losses: a `list` containing the validation loss values taken during training.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a DNNRegressor object.
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
        feature_columns=construct_feature_columns(training_examples),
        hidden_units=hidden_units,
        optimizer=my_optimizer
    )

    # Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples,
                                            training_targets["median_house_value"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                    training_targets["median_house_value"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                      validation_targets["median_house_value"],
                                                      num_epochs=1,
                                                      shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        dnn_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # Take a break and compute predictions.
        training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item['predictions'][0] for item in training_predictions])

        validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

        # Compute training and validation loss.
        training_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(training_predictions, training_targets))
        validation_root_mean_squared_error = math.sqrt(
            metrics.mean_squared_error(validation_predictions, validation_targets))
        # Occasionally print the current loss.
        print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
        # Add the loss metrics from this period to our list.
        training_rmse.append(training_root_mean_squared_error)
        validation_rmse.append(validation_root_mean_squared_error)
    print("Model training finished.")

    # Output a graph of loss metrics over periods.
    plt.ylabel("RMSE")
    plt.xlabel("Periods")
    plt.title("Root Mean Squared Error vs. Periods")
    plt.tight_layout()
    plt.plot(training_rmse, label="training")
    plt.plot(validation_rmse, label="validation")
    plt.legend()
    plt.show()

    print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
    print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

    return dnn_regressor, training_rmse, validation_rmse


# _ = train_nn_regression_model(
#     my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
#     steps=5000,
#     batch_size=70,
#     hidden_units=[10, 10],
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets)


# 线性缩放
# 将输入标准化以使其位于 (-1, 1) 范围内可能是一种良好的标准做法。这样一来，SGD 在一个维度中采用很大步长（或者在另一维度中采用很小步长）时不会受阻。数值优化的爱好者可能会注意到，这种做法与使用预调节器 (Preconditioner) 的想法是有联系的。
def linear_scale(series):
    min_val = series.min()
    max_val = series.max()
    scale = (max_val - min_val) / 2.0
    return series.apply(lambda x: ((x - min_val) / scale) - 1.0)


# 任务 1：使用线性缩放将特征标准化
# 将输入标准化到 (-1, 1) 这一范围内。
# 花费 5 分钟左右的时间来训练和评估新标准化的数据。您能达到什么程度的效果？
# 一般来说，当输入特征大致位于相同范围时，神经网络的训练效果最好。
# 对您的标准化数据进行健全性检查。（如果您忘了将某个特征标准化，会发生什么情况？）
def normalize_linear_scale(examples_dataFrame):
    """Returns a version of the input `DataFrame` that has all its features normalized linearly."""
    processed_features = pd.DataFrame()
    processed_features["latitude"] = linear_scale(examples_dataFrame["latitude"])
    processed_features["longitude"] = linear_scale(examples_dataFrame["longitude"])
    processed_features["housing_median_age"] = linear_scale(examples_dataFrame["housing_median_age"])
    processed_features["total_rooms"] = linear_scale(examples_dataFrame["total_rooms"])
    processed_features["total_bedrooms"] = linear_scale(examples_dataFrame["total_bedrooms"])
    processed_features["population"] = linear_scale(examples_dataFrame["population"])
    processed_features["households"] = linear_scale(examples_dataFrame["households"])
    processed_features["median_income"] = linear_scale(examples_dataFrame["median_income"])
    processed_features["rooms_per_person"] = linear_scale(examples_dataFrame["rooms_per_person"])
    return processed_features


normalized_dataFrame = normalize_linear_scale(preprocess_features(california_housing_dataFrame))
normalized_training_examples = normalized_dataFrame.head(12000)
normalized_validation_examples = normalized_dataFrame.tail(5000)

print("normalized_training_examples:")
display.display(normalized_training_examples.describe())
print("normalized_validation_examples:")
display.display(normalized_validation_examples.describe())

# _ = train_nn_regression_model(
#     my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.005),
#     steps=2000,
#     batch_size=50,
#     hidden_units=[10,10],
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets)

# 任务 2：尝试其他优化器
# 使用 AdaGrad 和 Adam 优化器并对比其效果。
# AdaGrad 优化器是一种备选方案。AdaGrad 的核心是灵活地修改模型中每个系数的学习率，从而单调降低有效的学习率。该优化器对于凸优化问题非常有效，但不一定适合非凸优化问题的神经网络训练。您可以通过指定 AdagradOptimizer（而不是 GradientDescentOptimizer）来使用 AdaGrad。请注意，对于 AdaGrad，您可能需要使用较大的学习率。
# 对于非凸优化问题，Adam 有时比 AdaGrad 更有效。要使用 Adam，请调用 tf.train.AdamOptimizer 方法。此方法将几个可选超参数作为参数，但我们的解决方案仅指定其中一个 (learning_rate)。在应用设置中，您应该谨慎指定和调整可选超参数。

# _, adagrad_training_losses, adagrad_validation_losses = train_nn_regression_model(
#     my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.5),
#     steps=500,
#     batch_size=100,
#     hidden_units=[10, 10],
#     training_examples=normalized_training_examples,
#     training_targets=training_targets,
#     validation_examples=normalized_validation_examples,
#     validation_targets=validation_targets)
#
# _, adam_training_losses, adam_validation_losses = train_nn_regression_model(
#     my_optimizer=tf.train.AdamOptimizer(learning_rate=0.009),
#     steps=500,
#     batch_size=100,
#     hidden_units=[10, 10],
#     training_examples=normalized_training_examples,
#     training_targets=training_targets,
#     validation_examples=normalized_validation_examples,
#     validation_targets=validation_targets)
#
# plt.ylabel("RMSE")
# plt.xlabel("Periods")
# plt.title("Root Mean Squared Error vs. Periods")
# plt.plot(adagrad_training_losses, label="Adagrad training")
# plt.plot(adagrad_validation_losses, label="Adagrad validation")
# plt.plot(adam_training_losses, label="Adam training")
# plt.plot(adam_validation_losses, label="Adam validation")
# plt.legend()
# plt.show()


# 任务 3：尝试其他标准化方法
# 尝试对各种特征使用其他标准化方法，以进一步提高性能。
# 如果仔细查看转换后数据的汇总统计信息，您可能会注意到，对某些特征进行线性缩放会使其聚集到接近 -1 的位置。
# 例如，很多特征的中位数约为 -0.8，而不是 0.0。
training_examples.hist(bins=20, figsize=(18, 20), xlabelsize=2)
plt.show()


# 拉大左侧的离散程度
def log_normalize(series):
    return series.apply(lambda x: math.log(x + 1.0))


def clip(series, clip_to_min, clip_to_max):
    return series.apply(lambda x: (
        min(max(x, clip_to_min), clip_to_max)))


def z_score_normalize(series):
    mean = series.mean()
    std_dv = series.std()
    return series.apply(lambda x: (x - mean) / std_dv)


def binary_threshold(series, threshold):
    return series.apply(lambda x: (1 if x > threshold else 0))


def normalize(examples_dataFrame):
    """Returns a version of the input `DataFrame` that has all its features normalized."""
    processed_features = pd.DataFrame()

    processed_features["households"] = log_normalize(examples_dataFrame["households"])
    processed_features["median_income"] = log_normalize(examples_dataFrame["median_income"])
    processed_features["total_bedrooms"] = log_normalize(examples_dataFrame["total_bedrooms"])

    processed_features["latitude"] = linear_scale(examples_dataFrame["latitude"])
    processed_features["longitude"] = linear_scale(examples_dataFrame["longitude"])
    processed_features["housing_median_age"] = linear_scale(examples_dataFrame["housing_median_age"])

    processed_features["population"] = linear_scale(clip(examples_dataFrame["population"], 0, 5000))
    processed_features["rooms_per_person"] = linear_scale(clip(examples_dataFrame["rooms_per_person"], 0, 5))
    processed_features["total_rooms"] = linear_scale(clip(examples_dataFrame["total_rooms"], 0, 10000))

    return processed_features


normalized_dataFrame = normalize(preprocess_features(california_housing_dataFrame))
normalized_training_examples = normalized_dataFrame.head(12000)
normalized_validation_examples = normalized_dataFrame.tail(5000)


# _ = train_nn_regression_model(
#     my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.15),
#     steps=1000,
#     batch_size=50,
#     hidden_units=[10, 10],
#     training_examples=normalized_training_examples,
#     training_targets=training_targets,
#     validation_examples=normalized_validation_examples,
#     validation_targets=validation_targets)

def position_normalize(examples_dataFrame):
    processed_features = pd.DataFrame()
    processed_features["latitude"] = linear_scale(examples_dataFrame["latitude"])
    processed_features["longitude"] = linear_scale(examples_dataFrame["longitude"])

    return processed_features


position_dataFrame = normalize(preprocess_features(california_housing_dataFrame))
position_training_examples = position_dataFrame.head(12000)
position_validation_examples = position_dataFrame.tail(5000)

_ = train_nn_regression_model(
    my_optimizer=tf.train.AdagradOptimizer(learning_rate=0.05),
    steps=500,
    batch_size=50,
    hidden_units=[10, 10, 5, 5],
    training_examples=position_training_examples,
    training_targets=training_targets,
    validation_examples=position_validation_examples,
    validation_targets=validation_targets)
