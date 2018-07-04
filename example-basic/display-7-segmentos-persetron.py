#!/usr/bin/python3
#
# Reconocer numero.
#
# Dado un display de 7 segmentos.
#
# a) Reconocer Pares
# b) Mayores a 5
# c) Reconocer impares
#
#     Representación del dígito.        Resultado deseados.
#  | # | a | b | c | d | e | f | g |    | ta | tb | tc |
#  |---|---|---|---|---|---|---|---|    |----|----|----|
#  | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |    | 1  | 0  | 0  |
#  | 1 |   | 1 | 1 | 0 | 0 | 0 | 0 |    | 0  | 0  | 0  |
#  | 2 | 1 | 1 | 0 | 1 | 1 | 0 | 1 |    | 1  | 0  | 1  |
#   ...                                  ...
#  | 7 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |    | 0  | 1  | 1  |
#   ...                                  ...
#
# ====================================================
import multiprocessing
import tensorflow as tf
import random
print("TensorFalow", tf.__version__)

# Configurar TensorFlow para usar todos los CPUs de la PC
CPUs = multiprocessing.cpu_count()
config = tf.ConfigProto(device_count={"CPU": CPUs},
                        inter_op_parallelism_threads=1,
                        intra_op_parallelism_threads=1)


display = tf.constant([
    # a  b  c  d  e  f  g
    [1, 1, 1, 1, 1, 1, 0],  # 0
    [0, 1, 1, 0, 0, 0, 0],  # 1
    [1, 1, 0, 1, 1, 0, 1],  # 2
    [1, 1, 1, 1, 0, 0, 1],  # 3
    [0, 1, 1, 0, 0, 1, 1],  # 4
    [1, 0, 1, 1, 0, 1, 1],  # 5
    [1, 0, 1, 1, 1, 1, 1],  # 6
    [1, 1, 1, 0, 0, 0, 0],  # 7
    [1, 1, 1, 1, 1, 1, 1],  # 8
    [1, 1, 1, 1, 0, 1, 1]   # 9
], dtype=tf.float32, name='Display')

# Resultados esperados para los puntos a, b, c
# Números pares
t_par = tf.constant([1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
                    dtype=tf.float32, name='Train_pares')
# Mayores a 5
t_may5 = tf.constant([0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                     dtype=tf.float32, name='Train_mayores_5')
# Números Primos
t_primo = tf.constant([0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
                      dtype=tf.float32, name='Train_primos')


# Función escalón
#   0 si x < 0
#   1 si >= 0
def funEsclon(x):
    return 0 if x < 0 else 1


if __name__ == "__main__":

    # Session and Variable Initialization
    #
    # with tf.Session() as sess:
    with tf.Session(config=config) as sess:
        try:
            # Valores a buscar
            #
            # - Peso
            W = tf.Variable(tf.random_uniform([7]), name="Weight")
            # - Peso sináptico
            # b = tf.random_uniform(shape=[1], name="Synaptic_weight")
            # b = random.uniform(-1, 1)
            b = random.random()
            # Epocas
            epocas = 10

            # Comenzar :D
            #
            # Inicializar variables de tensorflow
            sess.run(tf.global_variables_initializer())

            print(display.name, '\n', display.eval())
            # print("Elementos en display:", tf.size(display).eval())
            print(t_par.name, '\n', t_par.eval())

            trainData = t_par     # Con 10 epocas OK
            # trainData = t_may5      # Con 200 epocas OK
            # trainData = t_primo   # Con 100 epocas OK

            # Entrenar Neurona
            #
            # tf.multiply vs tf.matmul to calculate the dot product
            # https://stackoverflow.com/questions/47583501/tf-multiply-vs-tf-matmul-to-calculate-the-dot-product
            print("Entrenando...")
            for epoca in range(epocas):
                for q in range(10):
                    # print('>>', W.eval(), 'd', tf.gather(display, 0).eval())
                    n = tf.multiply(W, tf.gather(display, q))
                    # print(n.eval())
                    sum_n = tf.reduce_sum(n)
                    # print(sum_n.eval())
                    sum_n_b = sum_n.eval() + b  # Combinacion lineal
                    # print(sum_n_b)
                    e = tf.gather(trainData, q) - funEsclon(sum_n_b)  # Error
                    # print('e', e.eval())
                    # print(tf.multiply(e, tf.gather(display, q)).eval())
                    W = W + tf.multiply(e, tf.gather(display, q))
                    # print(W.eval())
                    b = b + e.eval()
                print(">> Epoca:", epoca, ' Error:', e.eval())

            print("Resultados del entrenamiento")
            print('W', W.eval())
            print('b', b)
            print('e', e.eval())

            # Comprobar
            #
            # Tiene que dar 1 en caso de ser Par.
            print('Test - Comprobar')
            # print("Es Par" if test(W, b, tf.gather(display, 1)) else "Es Impar")
            for q in range(10):
                evalue = tf.gather(display, q)
                real = tf.gather(trainData, q).eval()
                n = tf.reduce_sum(tf.multiply(W, tf.gather(display, q)))
                print("Evaluar:", evalue.eval(), "Valor real:", real,
                      'Ok' if funEsclon(n.eval() + b) == real else 'Error')

        except tf.errors.FailedPreconditionError as errorTry:
            print("Caught expected error: ", errorTry)
