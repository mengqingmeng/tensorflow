#condig=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784]) #输入图像
W = tf.Variable(tf.zeros([784, 10]))        #权重
b = tf.Variable(tf.zeros([10]))             #偏差
y = tf.nn.softmax(tf.matmul(x, W) + b)      #评分函数
y_ = tf.placeholder(tf.float32, [None, 10]) #标签
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))   #cost function 交叉熵函数
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy) #使用梯度下降算法（0.5的学习率），最小化交叉熵
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))      #比较预测和真实标签内容，返回结果为[1，0，1，1]
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  #求正确率
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


