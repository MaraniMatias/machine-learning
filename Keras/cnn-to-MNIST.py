#!/usr/bin/python3

from keras.datasets import mnist
import numpy as np

(train_imgs, train_labs), (test_imgs, test_labes) = mnist.load_data()

# normalize
train_imgs = train_imgs.astype(np.float32)/255.
test_imgs = test_imgs.astype(np.float32)/255.

aet = 60000
aev = 60000 * 10000

nrows = 28
ncols = 28

import matplotlib.pyplot as plt
# plt.matshow(train_imgs[2], cmap=plt.cm.gray_r)
# plt.show()

# print(train_labs[2])

# modeling input
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Reshape

# upsample is reverse of pooling
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
import sklearn.metrics as metritcs

"""
# ============================================================================
# simple model
model = Sequential()
# 128 vector de caracteristicas
model.add(Dense(128, input_shape=(nrows*ncols,)))
model.add(Activation("relu"))
model.add(Dense(nrows*ncols))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adadelta")

history = model.fit(train_imgs.reshape([-1, 784])[:aet],
                    train_imgs.reshape([-1, 784])[:aet],
                    epochs=100,
                    validation_data=(train_imgs.reshape([-1, 784])[aet:aev],
                                     train_imgs.reshape([-1, 784])[aet:aev])
                    )

plt.matshow(model.predict(train_imgs[2].reshape(
    [1, 784])).reshape([28, 28]), cmap=plt.cm.gray)
# ============================================================================
"""

# convolutions
model = Sequential()

model.add(Conv2D(24, (5, 5), input_shape=(1, nrows, ncols,),
                 padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Conv2D(24, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Conv2D(24, (5, 5), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))

model.add(Flatten())

model.add(Dense(32))
model.add(Activation('relu'))

# decoding
model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(216))
model.add(Activation('relu'))

model.add(Reshape(24, 3, 3))

model.add(Conv2D(24, (5, 5), padding='same'))
model.add(Activation('relu'))

model.add(UpSampling2D(size=(2, 2)))
model.add(ZeroPadding2D(padding=(0, 1, 0, 1)))

model.summary()

model.add(Reshape(24, 3, 3))
model.add(Conv2D(24, (5, 5), padding='same'))
model.add(Activation('relu'))
model.add(UpSampling2D(size=(2, 2)))

# reduce to 1 channel
model.add(Conv2D(1, (5, 5), padding='same', activation='sigmoid'))

model.compile(loss='binary_crossestropy', optimizer='adadelta')

history = model.fit(train_imgs.reshape([-1, 1, 28, 28])[:aet],
                    train_imgs.reshape([-1, 1, 28, 284])[:aet],
                    epochs=100,
                    validation_data=(train_imgs.reshape([-1, 1, 28, 28])[aet:aev],
                                     train_imgs.reshape([-1, 1, 28, 28])[aet:aev])
                    )
