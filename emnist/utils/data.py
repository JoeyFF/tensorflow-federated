import numpy
import tensorflow as tf
from typing import List
import numpy as np
import random


# 数据预处理
def preprocess(x, y, batch_size=32):
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x[..., tf.newaxis] / 255, tf.float32),
         tf.cast(y, tf.int32)))
    return dataset.shuffle(1000).batch(batch_size)


#
# def dataset_for_clients(data, clients_num):
#     """将数据均分给clients
#     Args:
#         data: Numpy数组格式的训练数据
#         clients_num: 终端数量
#     Returns:
#         数据集List
#         每个元素为tf.data.Dataset
#         元素个数为clients_num
#     """
#     if clients_num <= 0 or clients_num > 100:
#         raise ValueError('The number of client must between 1 and 100')
#     data_size = int(data.shape[0] / clients_num)
#     return [data[(i * data_size):((i + 1) * data_size)] for i in range(clients_num)]


def datasets_for_clients(images, labels, *, clients_num, batch_size=32) -> List[tf.data.Dataset]:
    """将数据均分给clients

    Args:
        images: Numpy数组格式的图片
        labels: Numpy数组格式的标签
        clients_num: 终端数量
        batch_size: 批量大小
    Returns:
        clients_datasets: 数据集List
        每个元素为tf.data.Dataset
        元素个数为clients_num
    """
    if clients_num <= 0 or clients_num > 100:
        raise ValueError('The number of client must between 1 and 100')

    data_size = int(images.shape[0] / clients_num)
    images_slice = [images[(i * data_size):((i + 1) * data_size)] for i in range(clients_num)]
    labels_slice = [labels[(i * data_size):((i + 1) * data_size)] for i in range(clients_num)]
    clients_datasets = []
    for i in range(clients_num):
        clients_datasets.append(preprocess(images_slice[i],
                                           labels_slice[i],
                                           batch_size))

    return clients_datasets


def datasets_random_for_clients(images, labels, *, clients_num, batch_size=32) -> List[tf.data.Dataset]:
    """将数据随机分给clients

    Args:
        images: Numpy数组格式的图片
        labels: Numpy数组格式的标签
        clients_num: 终端数量
        batch_size: 批量大小
    Returns:
        clients_datasets: 数据集List
        每个元素为tf.data.Dataset
        元素个数随机
    """
    if clients_num <= 0 or clients_num > 100:
        raise ValueError('The number of client must between 1 and 100')

    # 在0-batch_num产生随机分割序号
    batch_num = int(images.shape[0] / batch_size)
    slice_index = [0,]
    random_index = sorted(random.sample(range(1, batch_num), clients_num - 1))
    for index in random_index:
        slice_index.append(index*batch_size)
    slice_index.append(batch_num*batch_size)

    images_slice = [images[slice_index[i]:slice_index[i+1]] for i in range(clients_num)]
    labels_slice = [labels[slice_index[i]:slice_index[i+1]] for i in range(clients_num)]

    clients_datasets = []
    for i in range(clients_num):
        clients_datasets.append(preprocess(images_slice[i],
                                           labels_slice[i],
                                           batch_size))

    return clients_datasets
