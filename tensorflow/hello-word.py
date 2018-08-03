# pip install https://anaconda.org/intel/tensorflow/1.6.0/download/tensorflow-1.6.0-cp36-cp36m-linux_x86_64.whl
import tensorflow as tf
print(tf.__version__)

c = tf.constant('Hello, world!')

with tf.Session() as sess:

    print(sess.run(c))

    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))

# Create a graph.
g = tf.Graph()

# Establish the graph as the "default" graph.
with g.as_default():
    # Assemble a graph consisting of the following three operations:
    #   * Two tf.constant operations to create the operands.
    #   * One tf.add operation to add the two operands.
    x = tf.constant(8, name="x_const")
    y = tf.constant(5, name="y_const")
    suma = tf.add(x, y, name="x_y_sum")

    # Now create a session.
    # The session will run the default graph.
    with tf.Session() as sess:
        print(suma.eval())
