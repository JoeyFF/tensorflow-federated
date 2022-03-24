import tensorflow as tf
import tensorflow_federated as tff
import collections
from typing import OrderedDict


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


# 初始化模型并返回模型参数
@tff.tf_computation
def model_init():
    model = create_mnist_model()
    return model.trainable_variables


model_type = model_init.type_signature.result
CLIENT_NUM = 4
state_type = tff.StructType([('model', model_type), ('reputation', tf.float32)])
clients_state_type = tff.SequenceType(state_type)
federated_state_type = tff.FederatedType(state_type, tff.SERVER)
global_states_type = tff.FederatedType(clients_state_type, tff.CLIENTS)

@tff.tf_computation(state_type)
def calculate_state(state):
    model = create_mnist_model()
    tf.nest.map_structure(lambda x, y: x.assign(y), model.trainable_variables, list(state['model']))
    reputation = state['reputation'] + 1.0
    return collections.OrderedDict([('model', model.trainable_variables), ('reputation', reputation)])


@tff.federated_computation
def initialize_fn():
    return tff.federated_value(collections.OrderedDict([('model', model_init()),
                                                        ('reputation', 0.)]), tff.SERVER)


@tff.federated_computation(federated_state_type)
def next_fn(state):
    return tff.federated_map(calculate_state, state)


if __name__ == '__main__':
    iter_process = tff.templates.IterativeProcess(
        initialize_fn=initialize_fn,
        next_fn=next_fn
    )

    state = iter_process.initialize()
    for epoch in range(10):
        state = iter_process.next(state)
    print(state)
