#!/usr/bin/python3

from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential
from keras.datasets import mnist
from keras.utils import to_categorical

import matplotlib.pyplot as plt

# Setting consts
BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 3

# Load the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# Pre-processing
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.0
X_test /= 255.0
y_train = to_categorical(y_train, NUM_CLASSES)
y_test = to_categorical(y_test, NUM_CLASSES)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# Use the model
cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(5, 5), input_shape=(
    28, 28, 1), padding='same', activation='relu'))
cnn.add(MaxPooling2D())
cnn.add(Conv2D(64, kernel_size=(5, 5), padding='same', activation='relu'))
cnn.add(MaxPooling2D())
cnn.add(Flatten())
cnn.add(Dense(1024, activation='relu'))
cnn.add(Dense(NUM_CLASSES, activation='softmax'))
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(cnn.summary())

# Train the model
#
history_cnn = cnn.fit(X_train, y_train, epochs=EPOCHS, verbose=1,
                      validation_data=(X_train, y_train))


# What is the accuracy of the model?
#
# Plot the accuracy of the training model
plt.plot(history_cnn.history['acc'])
# Plot the accuracy of training and validation set
plt.plot(history_cnn.history['val_acc'])
# Accuracy of training and validation with loss
plt.plot(history_cnn.history['loss'])
plt.show()


# Predict
# cnn.predict(X_train[0]) # No se si es asi

# Train the model add weights
#
# cnn.load_weights('weights/cnn-model5.h5')
# score = cnn.evaluate(X_test,y_test)
