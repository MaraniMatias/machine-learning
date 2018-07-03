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
t_prim = tf.constant([0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
                     dtype=tf.float32, name='Train_primos')


# Función escalón
#   0 si x < 0
#   1 si >= 0
def funEsclon(x):
    return 0 if x < 0 else 1


if __name__ == "__main__":
    # Start sesión del TF
    with tf.Session(config=config) as sess:
        try:
            # =========================
            # Entrenar Neurona
            # =========================

            # Valores a buscar
            #
            # - Peso
            W = tf.Variable(tf.random_uniform([7]), name="Weight")
            # - Peso sináptico
            # b = tf.random_uniform(shape=[1], name="Synaptic_weight")
            # b = random.uniform(0, 1)
            b = random.random()
            # - Error
            e = tf.zeros([10])

            # Comenzar :D
            #
            # Inicializar variables de tensorflow
            sess.run(tf.global_variables_initializer())

            print(display.name, '\n', display.eval())
            # print("Elementos en display:", tf.size(display).eval())
            print(t_par.name, '\n', t_par.eval())

            # Épocas
            #
            # tf.multiply vs tf.matmul to calculate the dot product
            # https://stackoverflow.com/questions/47583501/tf-multiply-vs-tf-matmul-to-calculate-the-dot-product
            for epocas in range(2):
                for q in range(10):
                    # print('>>', W.eval(), 'd', tf.gather(display, 0).eval())
                    n = tf.multiply(W, tf.gather(display, q))
                    # print(n.eval())
                    sum_n = tf.reduce_sum(n)
                    # print(sum_n.eval())
                    sum_n_b = sum_n.eval() + b
                    # print(sum_n_b)
                    e = tf.gather(t_par, q) - funEsclon(sum_n_b)
                    # print('e', e.eval())
                    # print(tf.multiply(e, tf.gather(display, q)).eval())
                    W = W + tf.multiply(e, tf.gather(display, q))
                    # print(W.eval())
                    b = b + e.eval()

            print('W', W.eval())
            print('b', b)
            print('e', e.eval())

            #
            # Comprobar
            #
            # Tiene que dar 1 en caso de ser Par.
            print(tf.gather(display, 2).eval())
            n = tf.multiply(W, tf.gather(display, 2))
            # print(n.eval())
            sum_n = tf.reduce_sum(n)
            # print(sum_n.eval())
            sum_n_b = sum_n.eval() + b
            print("Es Par:", funEsclon(sum_n_b))

        except tf.errors.FailedPreconditionError as errorTry:
            print("Caught expected error: ", errorTry)
