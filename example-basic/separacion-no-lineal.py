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
trainAzul = tf.zeros([100], dtype=tf.float64)
trainRojo = tf.zeros([100], dtype=tf.float64) + -1
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
            print('Epoca:', epoca + 1, '/', EPOCAS)
            sumError = 0
            for i in range(Q):
                q = random.randint(0, Q-1)

                # Propagación de la entrada hacia la salida
                n_for_a1 = tf.multiply(W1, tf.gather(inputArray, q))
                a1 = tf.sigmoid(n_for_a1 + b1, name="a_1")

                print(epoca+1, ': W2', W2.eval())
                n_for_a2 = tf.matmul(tf.reshape(W2, [1, n1]), a1)
                a2 = tf.sigmoid(tf.gather(n_for_a2 + b2, 0), name="a_2")

                # Retropropagación de la sensibilidades
                e = tf.gather(trainData, q) - a2  # OK

                # s2 = -2*diag((1-a2^2))*e;
                diag2 = -2 * tf.matrix_diag(1 - tf.pow(a2, 2))
                print('diag2', diag2.eval())
                s2 = tf.matmul(diag2, tf.reshape(e, [2, 1]), name='sensivilidad_2')
                print('s2', s2.eval())
                s2 = tf.matrix_diag(
                    tf.concat([tf.gather(s2, 0), tf.gather(s2, 1)], 0), name='sensivilidad_2')
                print('s2', s2.eval())

                # s1 = diag((1-a1.^2))*W2'*s2;
                diag1 = tf.gather(1-tf.pow(a1, 2), 1)
                diag1 = tf.multiply(tf.reshape(diag1, [2, 1]), tf.reshape(W2, [1, n1]))
                s1 = tf.matmul(tf.reshape(diag1, [n1, 2]), s2, name='sensivilidad_1')

                # Actualización de pesos sinapticos y polarizaciones
                # W2 = W2 - alfa*s2*a1';
                W2 = W2 - tf.matmul(tf.multiply(s2, ALFA), tf.reshape(a1, [2, n1]))
                print(epoca+1, ': W2', W2.eval())
                # b2 = b2 - alfa*s2;
                b2 = b2 - tf.reduce_sum(ALFA*s2)
                # W1 = W1 - alfa*s1*P(:,q)';
                W1 = W1 - ALFA * s1 * tf.gather(trainData, q)
                # b1 = b1 - alfa*s1;
                b1 = b1 - ALFA * s1

                # Sumando el error cuadratico
                sumError = tf.reduce_sum(tf.pow(e, 2)) + sumError
            # Error cuadratico medio
            err = sumError/Q
            print("Error medio", err.eval())
