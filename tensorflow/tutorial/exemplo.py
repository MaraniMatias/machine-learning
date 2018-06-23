import tensorflow as tf

a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0, dtype=tf.float32)
c = a + b
print c

sess = tf.Session()
print sess.run(c)

### Creacion de la  grafica
a = tf.placeholder(tf.float32) # Requiere el tipo
b = tf.placeholder(tf.float32)
c = a+ b
print(c)

print(sess.run(c,{a:[1,2],b:[2,3]}))

# Variable
W = tf.Variable([.3],tf.float32) #Necesita un valor, y el tipo
b = tf.Variable([-.3],tf.float32)
x = tf.placeholder(tf.float32)
modelo_lineal = W*x +b

#
# Inicializacion
#
# Las constantes se inicializan cuando llama a tf.constant, y su valor nunca puede cambiar.
# Por el contrario, las variables no se inicializan cuando llama a tf.Variable.
# Para inicializar todas las variables en un programa TensorFlow,
# debe llamar explicitamente a una operacion especial de la siguiente manera:

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(modelo_lineal, {x: [1, 2, 3, 4]}))

y = tf.placeholder(tf.float32)
diferencia_sqr = (modelo_lineal - y)**2

costo = tf.reduce_sum(diferencia_sqr)

print(sess.run(costo,{x:[1,2,3,4],y: [0,-1,-2,-3]}))

W_ = tf.assign(W,[-1.0])
b_ = tf.assign(b,[1.0])

sess.run([W_,b_])
print(sess.run(costo,{x:[1,2,3,4],y: [0,-1,-2,-3]} ))

#
# Optimizadores
#
opt = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = opt.minimize(costo)

sess.run(init) # resetea las variables a su valor original

### Entrenamiento
for i in range(1000):
    sess.run(train,{x:[1, 2, 3, 4], y:[0, -1, -2, -3]})

print(sess.run([W,b]))

print(sess.run(costo,{x:[1,2,3,4],y: [0,-1,-2,-3]}))

print(sess.run([W,b]))

#
# Resumen
#
import tensorflow as tf

# Parametros del modelo
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)

# Entrada x, prediccion  modelo_lineal, y etiquetas u objetivos
x = tf.placeholder(tf.float32)
modelo_lineal = W*x + b
y = tf.placeholder(tf.float32)

# Funcion de costo
loss = tf.reduce_sum(tf.square(modelo_lineal - y)) # sum of the squares

# Optimizador
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Datos de entrenamiento
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]

# Loop de entrenamiento
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
for i in range(1000):
  sess.run(train, {x: x_train, y: y_train})

#Precision
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

