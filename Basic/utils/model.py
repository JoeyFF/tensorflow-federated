import tensorflow as tf
import tensorflow_federated as tff


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
def model_tff(keras_model, input_spec):
    return tff.learning.from_keras_model(
        keras_model=create_mnist_model(),
        input_spec=input_spec,
        loss=tf.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.metrics.SparseCategoricalAccuracy()]
    )