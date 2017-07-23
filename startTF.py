import tensorflow as tf
#定义常量
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly

#这里打印常量，但是并不能打印出常量的数值，而是Tensor("Const:0", shape=(), dtype=float32) Tensor("Const_1:0", shape=(), dtype=float32)
print(node1, node2)

#必须借助session操作常量
sess = tf.Session()
print(sess.run([node1, node2])) #打印出[3.0,4.0]

#常量运算
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))

#先定义，以后赋值，操作
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b

#用字典类型赋值
print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))

#先赋值 在运算
add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b:4.5}))

#定义变量
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
#使用变量，必须使用特定的语句初始化，在运行sess.run 之前，变量是未初始化的
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))

#计算损失值
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))#23.66

#损失值较大，所以手动更新权重和偏差，尝试计算损失值
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))