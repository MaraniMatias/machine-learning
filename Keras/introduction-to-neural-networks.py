#!/usr/bin/python3

from keras.datasets import mnist
from keras.preprocessing.image import load_img, array_to_img
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = mnist.load_data()


# Jugando con el DataSet
#
# print(X_train.size)  # No se bien que mustra
# print(X_train.shape)  # Totan de imagenes, dimencion x, dimencion y
# print(y_train.shape)
# print(X_test.shape)
# print(y_test.shape)

# Mostrando el 5
# print(X_train.shape, ' is ', y_train[0])
# plt.imshow(X_train[0], cmap="gray")
# plt.show(block=True)


# Preprocessing the image data
#
image_height, image_width = 28, 28
# Redimencionar los 60k de ejemplos
X_train = X_train.reshape(60000, image_height * image_width)
# print(X_train.shape) # Totan de imagenes, dimencion en una linea
X_test = X_test.reshape(10000, image_height * image_width)
# print(X_test.shape)
# print(X_train[0])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0
# print(X_train[0])


#  Build a model
#
# Reprecentan 10 categorias
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
# print(y_train.shape)
# print(y_test.shape)

# Modelo
model = Sequential()
# Modelo con tres capas
# capa 1 con 512 neuronas
# model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dense(512, activation='relu', input_shape=(image_height * image_width,)))
model.add(Dense(512, activation='relu'))
# capa 3 con 10 neuronas y 10 salidas
model.add(Dense(10, activation='softmax'))


# Compile the model
# Creo que categorical_crossentropy es porque usamos clases
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
EPOCHS = 20  # epocas
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_test, y_test))


# What is the accuracy of the model?
#
# Plot the accuracy of the training model
plt.plot(history.history['acc'])
# Plot the accuracy of training and validation set
plt.plot(history.history['val_acc'])
# Accuracy of training and validation with loss
plt.plot(history.history['loss'])

plt.show()
# Evaluating the model
score = model.evaluate(X_test, y_test)
print(score)
