#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot

__arthur__ = "张俊鹏"

print(pd.__version__)

# Series，它是单一列。DataFrame 中包含一个或多个 Series，每个 Series 均有一个名称
city_names = pd.Series(['San Francisco', "San Jose", "Sacramento"])
population = pd.Series([852469, 1015785, 485199])

# 从文件中获取数据
california_housing_dataframe = pd.read_csv("data/california_housing_train_for_pandas.csv", sep=',')
print(california_housing_dataframe.describe())  # 对数据进行描述：最值、均值、标准值……
print(california_housing_dataframe.head())  # 取前五行数据
print(california_housing_dataframe.hist('housing_median_age'))  # 一个列中值的分布
pyplot.show()

# DataFrame，您可以将它想象成一个关系型数据表格，其中包含多个行和已命名的列。
cities = pd.DataFrame({"City name": city_names, "Population": population})
print(type(cities["City name"]))
print("----------")
print(cities["City name"])

print(type(cities[0:2]))
print(cities[0:2])

print(type(cities))

# 操控数据
print(population / 1000)

# numpy函数
print(np.log(population))

# 创建了一个指明 population 是否超过 100 万的新 Series
rest = population.apply(lambda val: val > 1000000)
print(rest)
print(type(rest))

cities['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities['Population density'] = cities['Population'] / cities["Area square miles"]
print(cities)
print(cities.describe())

cities['Is wide and has saint name'] = (
        cities['Area square miles'] > 50 & cities['City name'].apply(lambda name: name.startswith('San')))
print(cities)

# 索引

# Series 和 DataFrame 对象也定义了 index 属性，该属性会向每个 Series 项或 DataFrame 行赋一个标识符值。
# 默认情况下，在构造时，pandas 会赋可反映源数据顺序的索引值。索引值在创建后是稳定的；也就是说，它们不会因为数据重新排序而发生改变。
print(city_names.index)
print(cities.index)

# 调用 DataFrame.reindex 以手动重新排列各行的顺序。例如，以下方式与按城市名称排序具有相同的效果：
print(cities.reindex([2, 0, 1]))

# 重建索引是一种随机排列 DataFrame 的绝佳方式。在下面的示例中，我们会取用类似数组的索引，
# 然后将其传递至 NumPy 的 random.permutation 函数，该函数会随机排列其值的位置。
# 如果使用此重新随机排列的数组调用 reindex，会导致 DataFrame 行以同样的方式随机排列。
print("first")
print(cities.reindex(np.random.permutation(cities.index)))
print("second")
print(cities.reindex(np.random.permutation(cities.index)))
print("third")
print(cities.reindex(np.random.permutation(cities.index)))



# 这种行为是可取的，因为索引通常是从实际数据中提取的字符串（请参阅 pandas reindex 文档，查看索引值是浏览器名称的示例）。
# 在这种情况下，如果允许出现“丢失的”索引，您将可以轻松使用外部列表重建索引，因为您不必担心会将输入清理掉。
print(cities.reindex([0, 4, 5, 2]))
