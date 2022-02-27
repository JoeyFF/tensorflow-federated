import os
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
import nest_asyncio
from collections import OrderedDict
from matplotlib import pyplot as plt
import utils
import time

# 环境变量设置
nest_asyncio.apply()
is_training = True
is_ascending = False
is_descending = False
image_path = './results/images/differ_epochs.png'
model_path = './results/models/keras_model_mnist.h5'
logs_path = './results/logs/'
case = ['ascend', 'descend', 'average']

# 定义FL超参数
# EXPERIMENTS = 4
CLIENTS_NUM = 6  # 终端数
ROUNDS = 10  # 联邦学习轮数
EPOCHS_ASC = [tf.Variable(0, dtype=tf.int32, name=f'client{i}_epochs') for i in range(CLIENTS_NUM)]  # 本地训练轮数
EPOCHS_DSC = [tf.Variable(ROUNDS+1, dtype=tf.int32, name=f'client{i}_epochs') for i in range(CLIENTS_NUM)]
EPOCHS_AVG = [tf.Variable(10, dtype=tf.int32, name=f'client{i}_epochs') for i in range(CLIENTS_NUM)]
BATCH_SIZE = 32  # 批量大小
LEARNING_RATE = 0.02


# 准备数据
# emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
clients_datasets = utils.datasets_for_clients(x_train, y_train, clients_num=CLIENTS_NUM, batch_size=BATCH_SIZE)
dataset_train = utils.preprocess(x_train, y_train, batch_size=BATCH_SIZE)
dataset_test = utils.preprocess(x_test, y_test, batch_size=BATCH_SIZE)
input_spec = clients_datasets[0].element_spec


# 构建keras模型
def create_mnist_model():
    return tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu',
                               input_shape=(None, None, 1)),
        tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(10),
        tf.keras.layers.Softmax()
    ])


# 根据tf.keras模型构建tff模型
def model_tff():
    return tff.learning.from_keras_model(
        keras_model=create_mnist_model(),
        input_spec=input_spec,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy()]
    )


# 评估模型
def evaluate(model_weights, dataset):
    model = create_mnist_model()
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
    )
    model.set_weights(model_weights)
    return model.evaluate(dataset)


# 初始化模型并返回模型参数
@tff.tf_computation
def model_init():
    model = model_tff()
    return model.trainable_variables


# 定义模型和数据集的规格
# 非联邦类型
model_type = model_init.type_signature.result
dataset_type = tff.SequenceType(input_spec)
epochs_type = tff.TensorType(tf.int32)

# 联邦类型
federated_global_model_type = tff.FederatedType(model_type, placement=tff.SERVER)
# federated_local_state_type = tff.FederatedType(global_state_type, placement=tff.CLIENTS)
federated_dataset_type = tff.FederatedType(dataset_type, placement=tff.CLIENTS)
federated_epochs_type = tff.FederatedType(epochs_type, placement=tff.CLIENTS)


# 初始化全局模型参数为FederatedType
@tff.federated_computation
def init_global_model():
    return tff.federated_value(model_init(), tff.SERVER)


@tf.function
def train(model, dataset, global_model_weights, optimizer, epochs):
    """本地训练实现

    :param epochs: 训练轮数
    :param model: tff包装的keras模型
    :param dataset: 本地训练数据集
    :param global_model_weights: 服务器分发的全局模型
    :param optimizer: 优化器
    :return: 本地模型参数

    """

    local_model_weights = model.trainable_variables
    # 把本地模型的每一层参数都赋值为全局模型的的参数
    tf.nest.map_structure(lambda x, y: x.assign(y),
                          local_model_weights, global_model_weights)
    tf.print(epochs)
    for epoch in range(epochs):
        for batch in dataset:
            with tf.GradientTape() as tape:
                # outputs(BatchOutput): loss,predictions,num_examples
                outputs = model.forward_pass(batch)
            # 梯度下降
            grads = tape.gradient(outputs.loss, local_model_weights)
            optimizer.apply_gradients(zip(grads, local_model_weights))

    return local_model_weights


# 本地训练
@tff.tf_computation(model_type, dataset_type, epochs_type)
def local_train(global_model_weights, local_dataset, local_epochs):
    model = model_tff()
    optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE)
    return train(model, local_dataset, global_model_weights, optimizer, local_epochs)


@tf.function
def global_model_update(model, aggregated_model_weights):
    global_model_weights = model.trainable_variables
    tf.nest.map_structure(lambda x, y: x.assign(y),
                          global_model_weights, aggregated_model_weights)
    return global_model_weights


# 将聚合结果更新到全局模型
@tff.tf_computation(model_type)
def global_update(aggregated_model_weights):
    model = model_tff()
    # tf.print(aggregated_model_weights[0])
    return global_model_update(model, aggregated_model_weights)


# 联邦学习流程
@tff.federated_computation(federated_global_model_type, federated_dataset_type, federated_epochs_type)
def next_fn(global_model_weights, local_dataset, local_epochs):
    """ 联邦学习流程

    :param local_epochs: 本地训练轮数
    :param global_model_weights: 全局模型参数
    :param local_dataset: 终端本地数据集
    :return: 本轮联邦学习后更新的全局模型参数
    """
    # 服务器将全局模型分发给每个终端 Server -> Clients
    global_model_weights_for_clients = tff.federated_broadcast(global_model_weights)

    # current_epochs = tff.federated_map(tff.tf_computation(lambda x: tf.add(x, 1)), local_epochs)
    # 终端更新本地模型 Clients -> Clients
    local_model_weights = tff.federated_map(
        local_train, (global_model_weights_for_clients, local_dataset, local_epochs))
    # 服务器聚合所有终端的本地模型 Clients -> Server
    # 考虑tff.aggregators如何实现
    # 注意tff.federated_mean的可选参数weight，可设置每个终端的本地模型聚合权重
    aggregated_model_weights = tff.federated_mean(local_model_weights)
    # 更新全局模型
    global_model_weights_updated = tff.federated_map(global_update, aggregated_model_weights)

    return global_model_weights_updated


if __name__ == '__main__':
    if is_training:
        federated_algorithm = tff.templates.IterativeProcess(
            initialize_fn=init_global_model,
            next_fn=next_fn
        )

        # 初始化
        state = federated_algorithm.initialize()
        loss = []
        acc = []
        start_time = time.time()

        if is_ascending:
            # 升序：增加本地训练轮数,最大=ROUNDS
            flag = 0
            EPOCHS = EPOCHS_ASC
        elif is_descending:
            # 降序：减少本地训练轮数,最小1
            flag = 1
            EPOCHS = EPOCHS_DSC
        else:
            # 平均：本地训练轮数固定为5
            flag = 2
            EPOCHS = EPOCHS_AVG

        for r in range(ROUNDS):
            print(f'Round:{r + 1}/{ROUNDS}...')

            if flag == 0:
                if EPOCHS[0].numpy() < ROUNDS:
                    for e in EPOCHS:
                        e.assign_add(1)
            elif flag == 1:
                if EPOCHS[0].numpy() > 1:
                    for e in EPOCHS:
                        e.assign_add(-1)

            # 开始迭代
            state = federated_algorithm.next(state, clients_datasets, EPOCHS)
            # 记录每轮联邦学习后的损失函数
            loss_val, acc_val = evaluate(state, dataset_test)
            loss.append(loss_val)
            acc.append(acc_val)

            # 损失函数<0.50时终止
            # if loss_val <= 0.50:
            #     break
            # 准确率>0.80时终止
            # if acc_val >= 0.80:
            #     break

        print('time_used:{:.2f} s'.format(time.time()-start_time))
        np.save(os.path.join(logs_path, case[flag]), acc)

    else:
        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        lines = ['o-', 's--', 'x:']
        for i in range(len(case)):
            content = np.load(os.path.join(logs_path, case[i]+'.npy'))
            ax.plot(np.linspace(1,ROUNDS,ROUNDS), content, lines[i])

        ax.set_title('FedAvg')
        ax.set_xlabel('Rounds')
        ax.set_ylabel('Accuracy')
        ax.set_xlim(0, ROUNDS+1)
        ax.set_ylim(0, 1)
        ax.set_xticks(np.arange(0,ROUNDS+1,2))
        plt.legend(labels=case,loc='lower right')
        plt.savefig(image_path)
