#!/usr/bin/python3
"""
python3 keras.py
"""
# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report

from keras.callbacks import LearningRateScheduler, ModelCheckpoint
from keras.datasets import cifar10
from keras import backend as K
from keras.layers.convolutional import MaxPooling2D, Conv2D
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.utils import plot_model
from keras.optimizers import SGD, Adam
from trainingmonitor import TrainingMonitor
import numpy as np
import os
# set the matplotlib backend so figures can be saved in the background
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")

# show information on the process ID
print("[INFO] process ID: {}".format(os.getpid()))

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
if not os.path.exists(os.path.join(__location__, "output")):
    os.makedirs(os.path.join(__location__, "output"))
if not os.path.exists(os.path.join(__location__, "weights")):
    os.makedirs(os.path.join(__location__, "weights"))


def step_decay(epoch):
    # initialize the base initial learning rate, drop factor, and
    # epochs to drop every
    initAlpha = 0.01
    factor = 0.25
    # factor = 0.5
    dropEvery = 5
    # compute learning rate for the current epoch
    alpha = initAlpha * (factor ** np.floor((1 + epoch) / dropEvery))
    # return the learning rate
    return float(alpha)


class MiniVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        # if we are using "channels first", update the input shape
        # and channels dimension
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # first CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(32, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # second CONV => RELU => CONV => RELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        # first (and only) set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(512, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        # softmax classifier
        model.add(Dense(classes, activation="softmax"))

        # return the constructed network architecture
        return model


# load the training and testing data, then scale it into the range [0, 1]
print("[INFO] loading CIFAR-10 data...")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# convert the labels from integers to vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

print("[INFO] compiling model...")
figPath = os.path.sep.join(["output", "{}.png".format(os.getpid())])
jsonPath = os.path.sep.join(["output", "{}.json".format(os.getpid())])
# define the set of callbacks to be passed to the model during training
callbacks = [
    TrainingMonitor(figPath, jsonPath=jsonPath),
    LearningRateScheduler(step_decay),
    ModelCheckpoint(
        filepath="weights/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
        monitor='val_loss',
        verbose=1,
        save_weights_only=True,
        period=1
    )
]
# initialize the optimizer and model
opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)
# opt = Adam(lr=0.001, amsgrad=True)

model = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
# save image of build model
plot_model(
    model,
    to_file=os.path.join(__location__, "output", "model.png"),
    show_shapes=True
)

# train the network
print("[INFO] training network...")
H = model.fit(
    trainX,
    trainY,
    validation_data=(testX, testY),
    batch_size=64,
    epochs=40,
    verbose=1,
    callbacks=callbacks
)
# evaluate the network
print("[INFO] evaluating network...")
predictions = model.predict(testX, batch_size=64)
print(
    classification_report(
        testY.argmax(axis=1), predictions.argmax(axis=1), target_names=labelNames
    )
)
# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 40), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 40), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 40), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 40), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on CIFAR-10")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig(args["output"])
