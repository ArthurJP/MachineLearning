#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
# 学习目标：
#
# 使用多个特征而非单个特征来进一步提高模型的有效性
# 调试模型输入数据中的问题
# 使用测试数据集检查模型是否过拟合验证数据

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

california_housing_dataFrame = pd.read_csv("data/california_housing_train_for_tensorFlow.csv", sep=",")


# california_housing_dataFrame = california_housing_dataFrame.reindex(
#     np.random.permutation(california_housing_dataFrame.index))

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
            california_housing_dataFrame["total_rooms"] / california_housing_dataFrame["population"]
    )
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
            california_housing_dataFrame["median_house_value"] / 1000.0
    )
    return output_targets


# 对于训练集，我们从共 17000 个样本中选择前 12000 个样本。
# training_examples = preprocess_features(california_housing_dataFrame.head(12000))
training_examples = preprocess_features(california_housing_dataFrame.sample(12000))
print(training_examples.describe())
# training_targets = preprocess_targets(california_housing_dataFrame.head(12000))
training_targets = preprocess_targets(california_housing_dataFrame.sample(12000))
print(training_targets.describe())

# 对于验证集，我们从共 17000 个样本中选择后 5000 个样本。
# validation_examples = preprocess_features(california_housing_dataFrame.tail(5000))
validation_examples = preprocess_features(california_housing_dataFrame.sample(5000))
print(validation_examples.describe())
# validation_targets = preprocess_targets(california_housing_dataFrame.tail(5000))
validation_targets = preprocess_targets(california_housing_dataFrame.sample(5000))
print(validation_targets.describe())

# 任务 1：检查数据
# 我们根据基准预期情况检查一下我们的数据：
# 对于一些值（例如 median_house_value），我们可以检查这些值是否位于合理的范围内（请注意，这是 1990 年的数据，不是现在的！）。
# 对于 latitude 和 longitude 等其他值，我们可以通过 Google 进行快速搜索，并快速检查一下它们与预期值是否一致。
# 如果您仔细看，可能会发现下列异常情况：
# median_income 位于 3 到 15 的范围内。我们完全不清楚此范围究竟指的是什么，看起来可能是某对数尺度？无法找到相关记录；我们所能假设的只是，值越高，相应的收入越高。
# median_house_value 的最大值是 500001。这看起来像是某种人为设定的上限。
# rooms_per_person 特征通常在正常范围内，其中第 75 百分位数的值约为 2。但也有一些非常大的值（例如 18 或 55），这可能表明数据有一定程度的损坏。


# 任务 2：绘制纬度/经度与房屋价值中位数的曲线图
plt.figure(figsize=(13, 8))

ax = plt.subplot(1, 2, 1)
ax.set_title("Validation Data")

ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(validation_examples["longitude"],
            validation_examples["latitude"],
            cmap="coolwarm",
            c=validation_targets["median_house_value"] / validation_targets["median_house_value"].max())

ax = plt.subplot(1, 2, 2)
ax.set_title("Training Data")

ax.set_autoscaley_on(False)
ax.set_ylim([32, 43])
ax.set_autoscalex_on(False)
ax.set_xlim([-126, -112])
plt.scatter(training_examples["longitude"],
            training_examples["latitude"],
            cmap="coolwarm",
            c=training_targets["median_house_value"] / training_targets["median_house_value"].max())
_ = plt.plot()

plt.show()


# 任务 3：返回来看数据导入和预处理代码，看一下您是否发现了任何错误
# 样本要使用sample采样


# 任务 4：训练和评估模型
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model of multiple features.

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

    # Construct a data-set, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features, targets))  # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
        ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels


# 由于我们现在使用的是多个输入特征，因此需要把用于将特征列配置为独立函数的代码模块化。
# （目前此代码相当简单，因为我们的所有特征都是数值，但当我们在今后的练习中使用其他类型的特征时，会基于此代码进行构建。）

def construct_feature_columns(input_features):
    """Construct the TensorFlow Feature Columns.

    Args:
        input_features: The names of the numerical input features to use.
    Returns:
        A set of feature columns
    """
    return set([tf.feature_column.numeric_column(my_feature)
                for my_feature in input_features])


def train_model(
        learning_rate,
        steps,
        batch_size,
        training_examples,
        training_targets,
        validation_examples,
        validation_targets):
    """Trains a linear regression model of multiple features.

        In addition to training, this function also prints training progress information,
        as well as a plot of the training and validation loss over time.

    Args:
        learning_rate: A `float`, the learning rate.
        steps: A non-zero `int`, the total number of training steps. A training step
        consists of a forward and backward pass using a single batch.
        batch_size: A non-zero `int`, the batch size.
        training_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataFrame` to use as input features for training.
        training_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataFrame` to use as target for training.
        validation_examples: A `DataFrame` containing one or more columns from
        `california_housing_dataFrame` to use as input features for validation.
        validation_targets: A `DataFrame` containing exactly one column from
        `california_housing_dataFrame` to use as target for validation.

    Returns:
        A `LinearRegressor` object trained on the training data.
    """

    periods = 10
    steps_per_period = steps / periods

    # Create a linear regressor object.
    my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
    linear_regressor = tf.estimator.LinearRegressor(
        feature_columns=construct_feature_columns(training_examples),
        optimizer=my_optimizer
    )

    # 1. Create input functions.
    training_input_fn = lambda: my_input_fn(training_examples, training_targets["median_house_value"],
                                            batch_size=batch_size)
    predict_training_input_fn = lambda: my_input_fn(training_examples, training_targets["median_house_value"],
                                                    num_epochs=1,
                                                    shuffle=False)
    predict_validation_input_fn = lambda: my_input_fn(validation_examples, validation_targets["median_house_value"],
                                                      num_epochs=1, shuffle=False)

    # Train the model, but do so inside a loop so that we can periodically assess
    # loss metrics.
    print("Training model...")
    print("RMSE (on training data):")
    training_rmse = []
    validation_rmse = []
    for period in range(0, periods):
        # Train the model, starting from the prior state.
        linear_regressor.train(
            input_fn=training_input_fn,
            steps=steps_per_period
        )
        # 2. Take a break and compute predictions.
        training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
        training_predictions = np.array([item["predictions"][0] for item in training_predictions])
        validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
        validation_predictions = np.array([item["predictions"][0] for item in validation_predictions])

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

    return linear_regressor


linear_regressor = train_model(
    # TWEAK THESE VALUES TO SEE HOW MUCH YOU CAN IMPROVE THE RMSE
    learning_rate=0.00002,
    steps=500,
    batch_size=5,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

plt.show()

# 任务 5：基于测试数据进行评估
california_housing_test_data = pd.read_csv("data/california_housing_test.csv", sep=",")

test_examples = preprocess_features(california_housing_test_data)
test_targets = preprocess_targets(california_housing_test_data)

predict_test_input_fn = lambda: my_input_fn(test_examples, test_targets["median_house_value"], num_epochs=1,
                                            shuffle=False)

test_predictions = linear_regressor.predict(input_fn=predict_test_input_fn)
test_predictions = np.array([item["predictions"][0] for item in test_predictions])

root_mean_squared_error = math.sqrt(
    metrics.mean_squared_error(test_predictions, test_targets))

print("Final RMSE (on test data): %0.2f" % root_mean_squared_error)
