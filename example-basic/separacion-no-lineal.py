#!/usr/bin/python3
import multiprocessing
import tensorflow as tf
import random
import scipy.io
print("TensorFalow", tf.__version__)

# Configurar TensorFlow para usar todos los CPUs de la PC
CPUs = multiprocessing.cpu_count()
config = tf.ConfigProto(device_count={"CPU": CPUs},
                        inter_op_parallelism_threads=1,
                        intra_op_parallelism_threads=1)

# Leer archivo con extencion mat
fileName = "./Clasificacion100.mat"
# Datos de entrenamiento, Contiene dos grupos de 100 elementos
fileArray = scipy.io.loadmat(fileName).get('P')
# Primero 100 Azul, los segundos 100 Rojo
# print('Azul', fileArray[0], '\n==========', '\nRojo',  fileArray[1])
inputArray = tf.concat([fileArray[0], fileArray[1]], 0, name="input_data")

# Array de entrenamiento para Azul 1 y para Rojo -1
trainAzul = tf.zeros([100])
trainRojo = tf.zeros([100]) + -1
trainData = tf.concat([trainAzul, trainRojo], 0, name="train_data")

# Num Total de ejemplos
Q = 200

# Inicializacion de pesos sipnaticos
#
n1 = 20  # Cantidad de neuronas en la capa 1, S^1 = n1
ep = 1  # Ventana entre los pesos sipnaticos
# Creo que son matrices de 'q' filas y 'm' columnas
W1 = tf.Variable(tf.random_uniform(
    [n1, 2], dtype=tf.float64, minval=-ep, maxval=ep,), name="Weight_1")
b1 = tf.Variable(tf.random_uniform(
    [n1, 1], dtype=tf.float64, minval=-ep, maxval=ep), name="Weight_sinap_1")
# La ultima capa contiene una neurona con n1 entradas
W2 = tf.Variable(tf.random_uniform(
    [n1], dtype=tf.float64, minval=-ep, maxval=ep), name="Weight_2")
b2 = random.random()

# Espesificar el alfa
ALFA = 0.01
EPOCAS = 1


if __name__ == "__main__":
    # Session and Variable Initialization
    #
    with tf.Session(config=config) as sess:
        # Inicializar variables de tensorflow
        sess.run(tf.global_variables_initializer())

        # print("Valores de entrada", inputArray.eval())
        # print("Datos de entrenamiento", trainData.eval())

        print("Entrenando...")
        # No usar logsig es muy lento, mejor sigmoid
        for epoca in range(EPOCAS):
            sumError = 0
            for i in [1]:  # range(Q):
                q = random.randint(0, Q-1)

                # Propagación de la entrada hacia la salida
                n_for_a1 = tf.multiply(W1, tf.gather(inputArray, q))
                a1 = tf.sigmoid(n_for_a1 + b1, name="a_1")

                n_for_a2 = tf.matmul(tf.reshape(W2, [1, 20]), a1)
                a2 = tf.sigmoid(tf.gather(n_for_a2 + b2, 0), name="a_2")

                # Retropropagación de la sensibilidades
                # e = T(q)-a2;
                # s2 = -2*diag((1-a2^2))*e;
                # s1 = diag((1-a1.^2))*W2'*s2;

                # Actualización de pesos sinapticos y polarizaciones
                # W2 = W2 - alfa*s2*a1';
                # b2 = b2 - alfa*s2;
                # W1 = W1 - alfa*s1*P(:,q)';
                # b1 = b1 - alfa*s1;

                # Sumando el error cuadratico
                # sumError = e^2 + sumError;
            # Error cuadratico medio
            # emedio(Epocas) = sum/Q
