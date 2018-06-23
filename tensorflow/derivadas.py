# Psychofun/Tensorflow-Como-Calcular-Derivadas
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

### Crea una sesion
sess = tf.Session()

x = tf.placeholder('float32')

f = 3*x


derivada = tf.gradients(f, x)[0]

x_array = np.linspace(-3,3)
#print(x_array)
f_, f_derivada = sess.run([f,derivada],
                                 {x:x_array})

plt.plot(x_array, f_, label="f")
plt.plot(x_array, f_derivada, label="df")
plt.legend();

x = tf.placeholder('float32')

f = x**2

#la derivada de x^2
derivada = tf.gradients(f, x)[0]

x_array = np.linspace(-3,3)

f_, f_derivada = sess.run([f,derivada],
                                 {x:x_array})

plt.plot(x_array, f_,label="f")
plt.plot(x_array, f_derivada, label="df")
plt.legend();

x = tf.placeholder('float32')

f = x**4 + x**3 + 100

#la derivada de x^2
derivada = tf.gradients(f, x)[0]

x_array = np.linspace(-3,3)

f_, f_derivada = sess.run([f,derivada],
                                 {x:x_array})

plt.plot(x_array, f_,label="f")
plt.plot(x_array, f_derivada, label="df")
plt.legend();

x = tf.placeholder('float32')

f = tf.sin(x)

#la derivada de x^2
derivada = tf.gradients(f, x)[0]

x_array = np.linspace(-3,3)
f_, f_derivada = sess.run([f,derivada],
                                 {x:x_array})

plt.plot(x_array, f_, label="f")
plt.plot(x_array, f_derivada, label="df")
plt.legend();

x = tf.placeholder('float32')

f = tf.cos(x)

#la derivada de x^2
derivada = tf.gradients(f, x)[0]

x_array = np.linspace(-3,3)
f_, f_derivada = sess.run([f,derivada],
                                 {x:x_array})
plt.figure(figsize=(12,12))
plt.plot(x_array, f_, label="f = cos(x)")
plt.plot(x_array, f_derivada, label="df = -sin(x)")
plt.legend();
### Guarda la grafica como una imagen
plt.savefig("derivada.png")

x = tf.placeholder('float32')

f = tf.exp(x)

#la derivada de x^2
derivada = tf.gradients(f, x)[0]

x_array = np.linspace(-3,3)
f_, f_derivada = sess.run([f,derivada],
                                 {x:x_array})

plt.plot(x_array, f_, label="f")
plt.plot(x_array, f_derivada, label="df")
plt.legend();



x = tf.placeholder('float32')

f = tf.log(x)

#la derivada de x^2
derivada = tf.gradients(f, x)[0]

x_array = np.linspace(-3,3)
f_, f_derivada = sess.run([f,derivada],
                                 {x:x_array})

plt.plot(x_array, f_, label="f")
plt.plot(x_array, f_derivada, label="df")
plt.legend();

x = tf.placeholder('float32')

f = tf.sigmoid(x)

#la derivada de x^2
derivada = tf.gradients(f, x)[0]

x_array = np.linspace(-3,3)
f_, f_derivada = sess.run([f,derivada],
                                 {x:x_array})

plt.plot(x_array, f_, label="f")
plt.plot(x_array, f_derivada, label="df")
plt.legend();


x = tf.placeholder('float32')

f = tf.nn.softmax(x)

#la derivada de x^2
derivada = tf.gradients(f, x)[0]

x_array = np.linspace(-3,3)
f_, f_derivada = sess.run([f,derivada],
                                 {x:x_array})

plt.plot(x_array, f_, label="f")
plt.plot(x_array, f_derivada, label="df")
plt.legend();

x = tf.placeholder('float32')

f = tf.sigmoid(x**2)

#la derivada de x^2
derivada = tf.gradients(f, x)[0]

x_array = np.linspace(-3,3)
f_, f_derivada = sess.run([f,derivada],
                                 {x:x_array})

plt.plot(x_array, f_, label="f")
plt.plot(x_array, f_derivada, label="df")
plt.legend();











