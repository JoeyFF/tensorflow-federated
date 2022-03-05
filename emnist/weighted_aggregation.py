import tensorflow as tf
import tensorflow_federated as tff
import nest_asyncio
from matplotlib import pyplot as plt
from utils import data

nest_asyncio.apply()
is_training = True

""" Weighted Aggregation 加权聚合
    将每个终端本地训练的准确度作为聚合权重
"""

# 定义FL超参数
EXPERIMENTS = 4
EPOCHS = 5  # 本地训练轮数
ROUNDS = 20  # 联邦学习轮数
BATCH_SIZE = 32  # 批量大小
LEARNING_RATE = 0.02
CLIENTS_NUM = 6  # 终端数

# 准备数据
# emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
# 随机分配数据集给终端
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
clients_datasets = data.datasets_random_for_clients(x_train, y_train, clients_num=CLIENTS_NUM, batch_size=BATCH_SIZE)
print([len(clients_datasets[i]) for i in range(CLIENTS_NUM)])
# dataset_train = utils.preprocess(x_train, y_train, batch_size=BATCH_SIZE)
dataset_test = data.preprocess(x_test, y_test, batch_size=BATCH_SIZE)
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
model_weights_type = model_init.type_signature.result
dataset_type = tff.SequenceType(input_spec)

# 联邦类型
federated_model_weights_type = tff.FederatedType(model_weights_type, placement=tff.SERVER)
federated_dataset_type = tff.FederatedType(dataset_type, tff.CLIENTS)
federated_reputation_type = tff.FederatedType(tf.float32, tff.CLIENTS)
federated_is_weighted_type = tff.FederatedType(tf.int8, tff.SERVER)

# 初始化全局模型参数为FederatedType
@tff.federated_computation
def init_global_model():
    return tff.federated_value(model_init(), tff.SERVER)


@tf.function
def train(model, dataset, global_model_weights, optimizer):
    """本地训练实现

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
    batches_acc = tf.constant(0.0, dtype=tf.float32)
    reputation = tf.constant(0.0, dtype=tf.float32)
    batches_num = tf.cast(len(dataset), tf.float32)
    for epoch in range(EPOCHS):
        for index, batch in enumerate(dataset):
            with tf.GradientTape() as tape:
                # outputs(BatchOutput): loss,predictions,num_examples
                outputs = model.forward_pass(batch)
            # 梯度下降
            predictions = tf.argmax(outputs.predictions, axis=1, output_type=tf.int32)
            batch_acc = tf.reduce_mean(tf.cast(tf.equal(batch[1], predictions),dtype=tf.float32))
            batches_acc = tf.add(batches_acc, batch_acc)
            grads = tape.gradient(outputs.loss, local_model_weights)
            optimizer.apply_gradients(zip(grads, local_model_weights))
            # accuracy.reset_states()
        reputation = tf.add(reputation, batches_acc)
        batches_acc = 0.0

    tf.print('Reputation:', reputation/EPOCHS)

    return local_model_weights, reputation/EPOCHS


# 本地训练
@tff.tf_computation(model_weights_type, dataset_type)
def local_train(global_model_weights, local_dataset):
    model = model_tff()
    optimizer = tf.optimizers.SGD(learning_rate=LEARNING_RATE)
    return train(model, local_dataset, global_model_weights, optimizer)


@tf.function
def global_model_update(model, aggregated_model_weights):
    global_model_weights = model.trainable_variables
    tf.nest.map_structure(lambda x, y: x.assign(y),
                          global_model_weights, aggregated_model_weights)
    return global_model_weights


# 将聚合结果更新到全局模型
@tff.tf_computation(model_weights_type)
def global_update(aggregated_model_weights):
    model = model_tff()
    return global_model_update(model, aggregated_model_weights)


# 联邦学习流程
@tff.federated_computation(federated_model_weights_type, federated_dataset_type, federated_is_weighted_type)
def next_fn(global_model_weights, local_dataset, is_weighted):
    """ 联邦学习流程

    :param is_weighted: 是否加权聚合
    :param global_model_weights: 全局模型参数
    :param local_dataset: 终端本地数据集
    :return: 本轮联邦学习后更新的全局模型参数
    """
    # 服务器将全局模型分发给每个终端 Server -> Clients
    model_weights_for_clients = tff.federated_broadcast(global_model_weights)
    # 终端更新本地模型 Clients -> Clients
    local_model_weights, local_accuracy = tff.federated_map(
        local_train, (model_weights_for_clients, local_dataset))

    # 服务器聚合所有终端的本地模型 Clients -> Server
    # 考虑tff.aggregators如何实现
    # 注意tff.federated_mean的可选参数weight，可设置每个终端的本地模型聚合权重
    if is_weighted == 1:
        aggregated_model_weights = tff.federated_mean(local_model_weights, weight=local_accuracy)
    else:
        aggregated_model_weights = tff.federated_mean(local_model_weights)
    # 更新全局模型
    global_model_weights_updated = tff.federated_map(global_update, aggregated_model_weights)

    return global_model_weights_updated


def run(is_weighted=1):
    federated_algorithm = tff.templates.IterativeProcess(
        initialize_fn=init_global_model,
        next_fn=next_fn
    )

    # 初始化
    state = federated_algorithm.initialize()
    accuracy = []
    # reputation_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for r in range(ROUNDS):
        print(f'Round:{r + 1}/{ROUNDS}...')
        state = federated_algorithm.next(state, clients_datasets,is_weighted)
        loss_val, accuracy_val = evaluate(state, dataset_test)
        accuracy.append(accuracy_val)
    return accuracy


if __name__ == '__main__':
    if is_training:
        res_weighted = run(is_weighted=1)
        res_unweighted = run(is_weighted=0)
        plt.plot(res_weighted, 'ro-')
        plt.plot(res_unweighted, 'bx:')
        plt.title('FedAvg')
        plt.xlabel('Rounds')
        plt.ylabel('Accuracy')
        plt.legend(labels=['weighted', 'unweighted'], loc='lower right')
        plt.savefig('./results/images/weighted_aggregation.png')
    else:
        pass
