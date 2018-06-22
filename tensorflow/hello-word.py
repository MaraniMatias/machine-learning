# pip install https://anaconda.org/intel/tensorflow/1.6.0/download/tensorflow-1.6.0-cp36-cp36m-linux_x86_64.whl
import tensorflow as tf

c = tf.constant('Hello, world!')

with tf.Session() as sess:

    print( sess.run(c) )

    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print(sess.run(hello))

