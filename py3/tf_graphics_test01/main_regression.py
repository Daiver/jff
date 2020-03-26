import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers

from tensorflow_graphics.geometry.transformation import quaternion
from tensorflow_graphics.math import vector
from tensorflow_graphics.notebooks import threejs_visualization
from tensorflow_graphics.notebooks.resources import tfg_simplified_logo


print(f"tf.__version__ {tf.__version__}")

tf.enable_eager_execution()
# tf.executing_eagerly()

# Loads the Tensorflow Graphics simplified logo.
vertices = tfg_simplified_logo.mesh['vertices'].astype(np.float32)
faces = tfg_simplified_logo.mesh['faces']
num_vertices = vertices.shape[0]


def pose_estimation_loss(y_true, y_pred):
    """Pose estimation loss used for training.

    This loss measures the average of squared distance between some vertices
    of the mesh in 'rest pose' and the transformed mesh to which the predicted
    inverse pose is applied. Comparing this loss with a regular L2 loss on the
    quaternion and translation values is left as exercise to the interested
    reader.

    Args:
    y_true: The ground-truth value.
    y_pred: The prediction we want to evaluate the loss for.

    Returns:
    A scalar value containing the loss described in the description above.
    """
    # y_true.shape : (batch, 7)
    y_true_q, y_true_t = tf.split(y_true, (4, 3), axis=-1)
    # y_pred.shape : (batch, 7)
    y_pred_q, y_pred_t = tf.split(y_pred, (4, 3), axis=-1)

    # vertices.shape: (num_vertices, 3)
    # corners.shape:(num_vertices, 1, 3)
    corners = tf.expand_dims(vertices, axis=1)

    # transformed_corners.shape: (num_vertices, batch, 3)
    # q and t shapes get pre-pre-padded with 1's following standard broadcast rules.
    transformed_corners = quaternion.rotate(corners, y_pred_q) + y_pred_t

    # recovered_corners.shape: (num_vertices, batch, 3)
    recovered_corners = quaternion.rotate(transformed_corners - y_true_t,
                                        quaternion.inverse(y_true_q))

    # vertex_error.shape: (num_vertices, batch)
    vertex_error = tf.reduce_sum((recovered_corners - corners)**2, axis=-1)

    return tf.reduce_mean(vertex_error)


def generate_training_data(num_samples):
    # random_angles.shape: (num_samples, 3)
    random_angles = np.random.uniform(-np.pi, np.pi, (num_samples, 3)).astype(np.float32)

    # random_quaternion.shape: (num_samples, 4)
    random_quaternion = quaternion.from_euler(random_angles)

    # random_translation.shape: (num_samples, 3)
    random_translation = np.random.uniform(-2.0, 2.0, (num_samples, 3)).astype(np.float32)

    # data.shape : (num_samples, num_vertices, 3)
    data = quaternion.rotate(
        vertices[tf.newaxis, :, :], random_quaternion[:, tf.newaxis, :]) + random_translation[:, tf.newaxis, :]

    # target.shape : (num_samples, 4+3)
    target = tf.concat((random_quaternion, random_translation), axis=-1)

    return np.array(data), np.array(target)


# Constructs the model.
model = keras.Sequential()
model.add(layers.Flatten(input_shape=(num_vertices, 3)))

model.add(layers.Dense(128, activation=tf.nn.relu))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation=tf.nn.relu))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation=tf.nn.relu))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation=tf.nn.relu))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(128, activation=tf.nn.relu))
# model.add(layers.BatchNormalization())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(256, activation=tf.nn.relu))
# model.add(layers.BatchNormalization())
model.add(layers.Dense(7))

optimizer = keras.optimizers.Adam()
model.compile(loss=pose_estimation_loss, optimizer=optimizer)
model.summary()

num_samples = 10000

data, target = generate_training_data(num_samples)

print(data.shape)   # (num_samples, num_vertices, 3): the vertices
print(target.shape)  # (num_samples, 4+3): the quaternion and translation


class ProgressTracker(keras.callbacks.Callback):

    def __init__(self, num_epochs, step=5):
        super().__init__()
        self.num_epochs = num_epochs
        self.current_epoch = 0.
        self.step = step
        self.last_percentage_report = 0

    def on_epoch_end(self, batch, logs={}):
        self.current_epoch += 1.
        training_percentage = int(self.current_epoch * 100.0 / self.num_epochs)
        if training_percentage - self.last_percentage_report >= self.step:
            print('Training ' + str(
                training_percentage) + '% complete. Training loss: ' + str(
                    logs.get('loss')) + ' | Validation loss: ' + str(
                        logs.get('val_loss')))
            self.last_percentage_report = training_percentage


reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    verbose=0,
    mode='auto',
    min_delta=0.0001,
    cooldown=0,
    min_lr=0)


# google internal 1
# Everything is now in place to train.
EPOCHS = 400
pt = ProgressTracker(EPOCHS)
history = model.fit(
    data,
    target,
    epochs=EPOCHS,
    validation_split=0.2,
    verbose=0,
    batch_size=32,
    callbacks=[reduce_lr_callback, pt])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim([0, 1])
plt.legend(['loss', 'val loss'], loc='upper left')
plt.xlabel('Train epoch')
_ = plt.ylabel('Error [mean square distance]')

plt.show()
