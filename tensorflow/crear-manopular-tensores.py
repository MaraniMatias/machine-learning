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
    print( 'scalar has shape', scalar.get_shape(), 'and value:\n', scalar.eval())
    print( 'vector has shape', vector.get_shape(), 'and value:\n', vector.eval())
    print( 'matrix has shape', matrix.get_shape(), 'and value:\n', matrix.eval())

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
    print( just_beyond_primes.eval())

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
    print( matrix_multiply_result.eval())

#
#  Cambio de formas de tensores
#
with tf.Graph().as_default():
  # Create an 8x2 matrix (2-D tensor).
  matrix = tf.constant([[1,2], [3,4], [5,6], [7,8],
                        [9,10], [11,12], [13, 14], [15,16]], dtype=tf.int32)

  # Reshape the 8x2 matrix into a 2x8 matrix.
  reshaped_2x8_matrix = tf.reshape(matrix, [2,8])

  # Reshape the 8x2 matrix into a 4x4 matrix
  reshaped_4x4_matrix = tf.reshape(matrix, [4,4])

  with tf.Session() as sess:
    print( "Original matrix (8x2):")
    print( matrix.eval())
    print( "Reshaped matrix (2x8):")
    print( reshaped_2x8_matrix.eval())
    print( "Reshaped matrix (4x4):")
    print( reshaped_4x4_matrix.eval())


with tf.Graph().as_default():
  # Create an 8x2 matrix (2-D tensor).
  matrix = tf.constant([[1,2], [3,4], [5,6], [7,8],
                        [9,10], [11,12], [13, 14], [15,16]], dtype=tf.int32)

  # Reshape the 8x2 matrix into a 3-D 2x2x4 tensor.
  reshaped_2x2x4_tensor = tf.reshape(matrix, [2,2,4])

  # Reshape the 8x2 matrix into a 1-D 16-element tensor.
  one_dimensional_vector = tf.reshape(matrix, [16])

  with tf.Session() as sess:
    print( "Original matrix (8x2):")
    print( matrix.eval())
    print( "Reshaped 3-D tensor (2x2x4):")
    print( reshaped_2x2x4_tensor.eval())
    print( "1-D vector:")
    print( one_dimensional_vector.eval())

#
print( "Ejercicio n. 1: Cambia la forma de dos tensores para poder multiplicarlos.")
#
with tf.Graph().as_default():

    a = tf.constant([5, 3, 2, 7, 1, 4])
    b = tf.constant([4, 6, 3])

    reshaped_a = tf.reshape(a, [2,3])
    reshaped_b = tf.reshape(b, [3,1])

    a_x_b = tf.matmul(reshaped_a,reshaped_b)

    with tf.Session() as sess:
        print reshaped_a.eval()
        print reshaped_b.eval()
        print a_x_b.eval()
#
#  Variables, Inicializacion y asignacion
#

