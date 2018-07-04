# https://colab.research.google.com/notebooks/mlcc/creating_and_manipulating_tensors.ipynb?hl=es-419#scrollTo=8Sef4d0SMMtk
import tensorflow as tf

#
#  Suma de vectores
#
with tf.Graph().as_default():
    # A scalar (0-D tensor).
    scalar = tf.zeros([])

    # A vector with 3 elements.
    vector = tf.zeros([3])

    # A matrix with 2 rows and 3 columns.
    matrix = tf.zeros([2, 3])

    with tf.Session() as sess:
        print('scalar has shape', scalar.get_shape(),
              'and value:\n', scalar.eval())
        print('vector has shape', vector.get_shape(),
              'and value:\n', vector.eval())
        print('matrix has shape', matrix.get_shape(),
              'and value:\n', matrix.eval())

#
#  Suma de vectores
#
with tf.Graph().as_default():
    # Create a six-element vector (1-D tensor).
    primes = tf.constant([2, 3, 5, 7, 11, 13], dtype=tf.int32)

    # Create a constant scalar with value 1.
    ones = tf.constant(1, dtype=tf.int32)

    # Add the two tensors. The resulting tensor is a six-element vector.
    just_beyond_primes = tf.add(primes, ones)

    with tf.Session() as sess:
        print(just_beyond_primes.eval())

#
#  Producto de arreglos
#
with tf.Graph().as_default():
    # Create a matrix (2-d tensor) with 3 rows and 4 columns.
    x = tf.constant([[5, 2, 4, 3], [5, 1, 6, -2], [-1, 3, -1, -2]],
                    dtype=tf.int32)

    # Create a matrix with 4 rows and 2 columns.
    y = tf.constant([[2, 2], [3, 5], [4, 5], [1, 6]], dtype=tf.int32)

    # Multiply `x` by `y`.
    # The resulting matrix will have 3 rows and 2 columns.
    matrix_multiply_result = tf.matmul(x, y)

    with tf.Session() as sess:
        print(matrix_multiply_result.eval())

#
#  Cambio de formas de tensores
#
with tf.Graph().as_default():
    # Create an 8x2 matrix (2-D tensor).
    matrix = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8],
                          [9, 10], [11, 12], [13, 14], [15, 16]], dtype=tf.int32)

    # Reshape the 8x2 matrix into a 2x8 matrix.
    reshaped_2x8_matrix = tf.reshape(matrix, [2, 8])

    # Reshape the 8x2 matrix into a 4x4 matrix
    reshaped_4x4_matrix = tf.reshape(matrix, [4, 4])

    with tf.Session() as sess:
        print("Original matrix (8x2):")
        print(matrix.eval())
        print("Reshaped matrix (2x8):")
        print(reshaped_2x8_matrix.eval())
        print("Reshaped matrix (4x4):")
        print(reshaped_4x4_matrix.eval())


with tf.Graph().as_default():
    # Create an 8x2 matrix (2-D tensor).
    matrix = tf.constant([[1, 2], [3, 4], [5, 6], [7, 8],
                          [9, 10], [11, 12], [13, 14], [15, 16]], dtype=tf.int32)

    # Reshape the 8x2 matrix into a 3-D 2x2x4 tensor.
    reshaped_2x2x4_tensor = tf.reshape(matrix, [2, 2, 4])

    # Reshape the 8x2 matrix into a 1-D 16-element tensor.
    one_dimensional_vector = tf.reshape(matrix, [16])

    with tf.Session() as sess:
        print("Original matrix (8x2):")
        print(matrix.eval())
        print("Reshaped 3-D tensor (2x2x4):")
        print(reshaped_2x2x4_tensor.eval())
        print("1-D vector:")
        print(one_dimensional_vector.eval())

#
print("Ejercicio n. 1: Cambia la forma de dos tensores para poder multiplicarlos.")
#
with tf.Graph().as_default():

    a = tf.constant([5, 3, 2, 7, 1, 4])
    b = tf.constant([4, 6, 3])

    reshaped_a = tf.reshape(a, [2, 3])
    reshaped_b = tf.reshape(b, [3, 1])

    a_x_b = tf.matmul(reshaped_a, reshaped_b)

    with tf.Session() as sess:
        print(reshaped_a.eval())
        print(reshaped_b.eval())
        print(a_x_b.eval())
#
#  Variables, Inicializacion y asignacion
#
g = tf.Graph()
with g.as_default():
    # Create a variable with the initial value 3.
    v = tf.Variable([3])

    # Create a variable of shape [1], with a random initial value,
    # sampled from a normal distribution with mean 1 and standard deviation 0.35.
    w = tf.Variable(tf.random_normal([1], mean=1.0, stddev=0.35))

with g.as_default():
    with tf.Session() as sess:
        try:
            v.eval()
        except tf.errors.FailedPreconditionError as e:
            print("Caught expected error: ", e)


with g.as_default():
    with tf.Session() as sess:
        initialization = tf.global_variables_initializer()
        sess.run(initialization)
        # Now, variables can be accessed normally, and have values assigned to them.
        print(v.eval())
        print(w.eval())

with g.as_default():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # These three prints will print the same value.
        print(w.eval())
        print(w.eval())
        print(w.eval())

with g.as_default():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # This should print the variable's initial value.
        print(v.eval())

        assignment = tf.assign(v, [7])
        # The variable has not been changed yet!
        print(v.eval())

        # Execute the assignment op.
        sess.run(assignment)
        # Now the variable is updated.
        print(v.eval())

# ### Ejercicio n.º 2: Simula 10 giros de dos dados.
#
# rea una simulación de dados, en la que se genere un tensor de 2-D de `10 × 3` con las siguientes características:
#
#  * Que las columnas `1` y `2` incluyan una tirada de uno de los dados.
#  * Que la columna `3` incluya la suma de las columnas `1` y `2` en la misma fila.
#
# or ejemplo, la primera fila puede tener los siguientes valores:
#
#  * La columna `1` incluye `4`
#  * La columna `2` incluye `3`
#  * La columna `3` incluye `7`
#
# onsulta la [documentación de TensorFlow](https://www.tensorflow.org/api_guides/python/array_ops) para resolver esta tarea.
#
with tf.Graph().as_default(), tf.Session() as sess:
    # Task 2: Simulate 10 throws of two dice. Store the results
    # in a 10x3 matrix.

    # We're going to place dice throws inside two separate
    # 10x1 matrices. We could have placed dice throws inside
    # a single 10x2 matrix, but adding different columns of
    # the same matrix is tricky. We also could have placed
    # dice throws inside two 1-D tensors (vectors); doing so
    # would require transposing the result.
    dice1 = tf.Variable(tf.random_uniform(
        [10, 1], minval=1, maxval=7, dtype=tf.int32))
    dice2 = tf.Variable(tf.random_uniform(
        [10, 1], minval=1, maxval=7, dtype=tf.int32))

    # We may add dice1 and dice2 since they share the same shape
    # and size.
    dice_sum = tf.add(dice1, dice2)

    # We've got three separate 10x1 matrices. To produce a single
    # 10x3 matrix, we'll concatenate them along dimension 1.
    resulting_matrix = tf.concat(
        values=[dice1, dice2, dice_sum], axis=1)

    # The variables haven't been initialized within the graph yet,
    # so let's remedy that.
    sess.run(tf.global_variables_initializer())

    print(resulting_matrix.eval())


with tf.Session() as sess:
    a = tf.constant([1, 2, 3, 4, 5, 6])
    b = tf.constant([7, 8, 9, 10, 11, 12, 1, 2, 3,  1, 1, 3], shape=[6, 2])
    a = tf.reshape(a, [1, 6])
    c = tf.matmul(a, b)
    print(a.eval())
    print(b.eval())
    print(c.eval())
