import os
import datetime
import tensorflow as tf
import tensorflow_federated as tff
import matplotlib.pyplot as plt

# training setting
CLIENT_ID = 1
EPOCHS = 10
is_training = False

# path to saved_model
dir_path = './models/client@' + str(CLIENT_ID)
if not os.path.exists(dir_path):
    os.makedirs(dir_path)
file_path = os.path.join(dir_path, 'weights.h5')

# tensorboard config
current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
train_log_path = './logs/train/' + current_time
test_log_path = './logs/test/' + current_time
# test_summary_writer = tf.summary.create_file_writer(test_log_path)
train_summary_writer = tf.summary.create_file_writer(train_log_path)
train_loss = tf.metrics.Mean('loss', dtype=tf.float32)
train_acc = tf.metrics.SparseCategoricalAccuracy('accuracy', dtype=tf.float32)


def preprocess(x, y):
    dataset = tf.data.Dataset.from_tensor_slices(
        (tf.cast(x_train[..., tf.newaxis] / 255, tf.float32),
         tf.cast(y_train, tf.int64)))
    return dataset.shuffle(1000).batch(32)


def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        tf.debugging.assert_equal(predictions.shape, (32, 10))
        loss_val = loss(labels, predictions)
    loss_history.append(loss_val.numpy().mean())
    train_loss(loss_val)
    train_acc(labels, predictions)
    grads = tape.gradient(loss_val, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def train(epochs):
    for epoch in range(epochs):
        for images, labels in dataset_train:
            train_step(images, labels)
        print(f'Epoch:{epoch}')
        with train_summary_writer.as_default():
            tf.summary.scalar('loss', train_loss.result(), step=epoch)
            tf.summary.scalar('accuracy', train_acc.result(), step=epoch)
        train_loss.reset_states()
        train_acc.reset_states()


def validation(epochs):
    for images, labels in dataset_test:
        train_step(images, labels)


# load and preprocess data for training
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
dataset_train = preprocess(x_train, y_train)
# print(dataset_train.element_spec)
dataset_test = preprocess(x_test, y_test)

# build model by keras
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu',
                           input_shape=(None, None, 1)),
    tf.keras.layers.Conv2D(16, [3, 3], activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(10),
])

# loss and optimizer
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.optimizers.Adam()
loss_history = []

if is_training:
    train(EPOCHS)
    model.save_weights(file_path)
    # ckpt_path = "my_checkpoint"
    # checkpoint = tf.train.Checkpoint(model=my_model)
    # checkpoint.write(ckpt_path)
    print(f'train_loss={loss_history[-1]}')

else:
    model.compile(optimizer, loss)
    model.load_weights(file_path)
    # new_model = MyModel()
    # new_checkpoint = tf.train.Checkpoint(model=new_model)
    # new_checkpoint.restore("my_checkpoint")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=test_log_path, histogram_freq=1)
    ls = model.evaluate(dataset_test, callbacks=tensorboard_callback)
    print(f'test_loss={ls}')
