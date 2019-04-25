import numpy as np
import time
from tensorflow.examples.tutorials.mnist import input_data

from keras.models import Sequential 
from keras.layers import Dense, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import Adam, RMSprop

import matplotlib.pyplot as plt

input_shape = (74, 74, 1)

depth = 64
dropout = 0.4

discriminator = Sequential()
discriminator.add(
         Conv2D(depth*1, 5, strides=2, input_shape=input_shape, padding='same', activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(dropout))
discriminator.add(
         Conv2D(depth*2, 5, strides=2, padding='same',activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(dropout))
discriminator.add(
         Conv2D(depth*4, 5, strides=2, padding='same',activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(dropout))
discriminator.add(
         Conv2D(depth*8, 5, strides=1, padding='same', activation=LeakyReLU(alpha=0.2)))
discriminator.add(Dropout(dropout))
# Out: 1-dim probability
discriminator.add(Flatten())
discriminator.add(Dense(1))
discriminator.add(Activation('sigmoid'))
discriminator.summary()

generator = Sequential()
dropout = 0.4
depth = 64+64+64+64
dim = 7
# In: 100
# Out: dim x dim x depth
generator.add(Dense(dim*dim*depth, input_dim=100))
generator.add(BatchNormalization(momentum=0.9))
generator.add(Activation('relu'))
generator.add(Reshape((dim, dim, depth)))
generator.add(Dropout(dropout))
# In: dim x dim x depth
# Out: 2*dim x 2*dim x depth/2
generator.add(UpSampling2D())
generator.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(Activation('relu'))
generator.add(UpSampling2D())
generator.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(Activation('relu'))
generator.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
generator.add(BatchNormalization(momentum=0.9))
generator.add(Activation('relu'))
# Out: 28 x 28 x 1 grayscale image [0.0,1.0] per pix
generator.add(Conv2DTranspose(1, 5, padding='same'))
generator.add(Activation('sigmoid'))
generator.summary()

optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
DM = Sequential()
DM.add(discriminator)
DM.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

optimizer = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)
AM = Sequential()
AM.add(generator)
AM.add(discriminator)
AM.compile(loss='binary_crossentropy', optimizer=optimizer,metrics=['accuracy'])

images_train = x_train[np.random.randint(0, x_train.shape[0], size=batch_size), :, :, :]
noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
images_fake = generator.predict(noise)
x = np.concatenate((images_train, images_fake))
y = np.ones([2*batch_size, 1])
y[batch_size:, :] = 0
d_loss = discriminator.train_on_batch(x, y)
y = np.ones([batch_size, 1])
noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
a_loss = AM.train_on_batch(noise, y)
