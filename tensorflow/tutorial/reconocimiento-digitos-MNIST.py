import tensorflow as tf
import numpy as np

print("version de tensorflow:", tf.__version__)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import matplotlib.pyplot as plt
import time

# Los Ejemplos de entrenamiento estan en:
# mnist.train.images
print("Numero de ejemplos de entrenamiento:", mnist.train.images.shape[0])

# El conjunto de validacion es:
# mnist.validation
print("Numero de ejemplos de validacion:", mnist.validation.images.shape[0])


# El conjunto de prueba es:
# mnist.test
print("Numero de ejemplos de prueba:", mnist.test.images.shape[0])


# Cada digito es un vector de dimension 784 .
print("Tamanio de cada digito:", mnist.train.images.shape[1])


# Las etiquetas se encuentran en:
# mnist.train.labels
# mnist.validation.labels
# mnist.test.labels

print("Tamano de cada etiqueta:", mnist.train.labels.shape[1])
# Cada etiqueta es un one-hot-vector,ie. un vector con un solo uno, las demas entradas son cero
# [1,0,0,0,0,0,0,0,0,0]  representa el numero 0
# [0,1,0,0,0,0,0,0,0,0]  representa el numero 1
#   .
#   .
#   .

# Cada digito se almacena como un vector de 784 dimensiones. Para visualizarlo, primero lo redimensionamos a una imagen de 28x28.


def muestra_digito(x):
    """
        x: vector
            784 dimensiones
        Muestra el vector como una imagen de 28x28
    """
    plt.axis('off')
    plt.imshow(x.reshape((28, 28)), cmap=plt.cm.gray)
    plt.show()
    return


def vis_imagen(i, conjunto="train"):
    """
        i indice del conjunto especificado
        conjunto: cadena
            Cualquiera: train, validation, test

        Muestra el digito en el indice i  y su etiqueta
    """
    if(conjunto == "train"):
        muestra_digito(mnist.train.images[i, ])
        label = np.argwhere(mnist.train.labels[i])[0][0]
    elif(conjunto == "test"):
        muestra_digito(mnist.test.images[i, ])
        label = np.argwhere(mnist.test.labels[i])[0][0]
    else:
        muestra_digito(mnist.validation.images[i, ])
        label = np.argwhere(mnist.validation.labels[i])[0][0]
    print("Etiqueta " + str(label))
    return


vis_imagen(0, conjunto="train")
vis_imagen(132, conjunto="validation")
vis_imagen(32, conjunto="test")
vis_imagen(50000, conjunto="train")

#
#  RED NEURONAL
#

# Placeholders para los datos de entrenamiento
# En ellos se pasaran despues los datos de entrenamiento (x,y)
# x imagen, y etiqueta

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

# Variables del modelo

# Capa 1
W_1 = tf.Variable(tf.truncated_normal(shape=[784, 512], stddev=0.2))
b_1 = tf.Variable(tf.zeros([512]))

# Capa 2 de salida
W_2 = tf.Variable(tf.truncated_normal(shape=[512, 10], stddev=0.2))
b_2 = tf.Variable(tf.zeros([10]))

# Arquitectura de la red neural


def NN(x):
    """
        x: matriz
            su forma  debe ser (m, 784)

        regresa la activacion de la capa de salida
        matriz de (m, 10)
    """
    # Capa Escondida 1.
    z_1 = tf.matmul(x, W_1) + b_1  # Combinacion lineal
    a_1 = tf.nn.relu(z_1)  # Activacion (funcion no lineal)

    # Capa 2. Esta es la capa de salida
    z_2 = tf.matmul(a_1, W_2) + b_2  # Combinacion lineal

    return z_2

# Funcion de costo


y_ = NN(x)
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))

# Predicciones

train_pred = tf.nn.softmax(y_)  # predicciones en el conjunto de entrenamiento
# Nota: la funcion softmax calcula la probabilidad de cada etiqueta del 0 al 9.
# Para obtener la prediccion necesitamos usar las funcion tf.argmax(y_,1) o su version en python np.argmax(y_,1)
# Asi se elige el digito mas probable para la imagen
# Esto lo hace la funcion precision

y_valid = NN(mnist.validation.images)
# predicciones en el conjunto de validacion
valid_pred = tf.nn.softmax(y_valid)

# Optimizador

opt = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Sesion e inicializacion de varables

sess = tf.Session()  # Crea una session
sess.run(tf.global_variables_initializer())

# Precision


def precision(predicciones, etiquetas):
    return (100.0 * np.sum(np.argmax(predicciones, 1) == np.argmax(etiquetas, 1))
            / predicciones.shape[0])

# Entrenamiento


pasos = 5000

print("Entrenamiento:")
for i in range(pasos):
    batch = mnist.train.next_batch(100)
    _, costo, predicciones = sess.run(
        [opt, cross_entropy, train_pred],  feed_dict={x: batch[0], y: batch[1]})

    if (i % 500 == 0):
        print("Costo del minibatch hasta el paso %d: %f" % (i, costo))
        print("Precision en el conjunto de entrenamiento: %.1f%%" %
              precision(predicciones, batch[1]))
        print("Precision en el conjunto de validacion: %.1f%%" % precision(
            valid_pred.eval(session=sess), mnist.validation.labels))
        print("\n")

y_test = NN(mnist.test.images)
test_prediction = tf.nn.softmax(y_test)
print("Precision en el conjunto de PRUEBA: %.1f%%" %
      precision(test_prediction.eval(session=sess), mnist.test.labels))

indice = 251
p = tf.argmax(NN(mnist.test.images[indice:indice+1]).eval(session=sess), 1)
print("Prediccion:", sess.run(p)[0])
vis_imagen(indice, conjunto="test")


def remove_transparency(im, bg_colour=(255, 255, 255)):

    # Only process if image has transparency
    if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):

        # Need to convert to RGBA if LA format due to a bug in PIL
        alpha = im.convert('RGBA').split()[-1]

        # Create a new background image of our matt color.
        # Must be RGBA because paste requires both images have the same format

        bg = Image.new("RGBA", im.size, bg_colour + (255,))
        bg.paste(im, mask=alpha)
        return bg

    else:
        return im


#################################
#
# Usa tu propia imagen
#
################################
from PIL import Image
imagen = "numero2.png"
img = Image.open(imagen)
img = remove_transparency(img).convert('L')

if img.size != (28, 28):
    img.thumbnail((28, 28), Image.ANTIALIAS)

entrada = np.array(img, dtype=np.float32)
entrada = entrada.reshape((1, 784))
entrada = entrada/255.0

p = tf.argmax(NN(entrada).eval(session=sess), 1)
print("Imagen:{}".format(imagen))
img.show()
print("Prediccion:", sess.run(p)[0])
