#condig=utf-8
import tensorflow as tf
import numpy as np
from PIL import Image

x = tf.placeholder(tf.float32, shape=[None, 784]) #输入
y_ = tf.placeholder(tf.float32, shape=[None, 10]) #输出
W = tf.Variable(tf.zeros([784,10])) #权重 变量
b = tf.Variable(tf.zeros([10])) #偏差 变量
y = tf.matmul(x,W) + b

# 使用和保存模型代码中一样的方式来声明变量
v1 = tf.Variable(tf.random_normal([1, 2]), name="v1")
v2 = tf.Variable(tf.random_normal([2, 3]), name="v2")
saver = tf.train.Saver() # 声明tf.train.Saver类用于保存模型
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化变量
    saver.restore(sess, "save/model.ckpt") # 即将固化到硬盘中的Session从保存路径再读取出来
    print("v1:", sess.run(v1)) # 打印v1、v2的值和之前的进行对比
    print("v2:", sess.run(v2))
    print("Model Restored")
    #
    im = Image.open("test_model_image/0_0.png")
    im = im.convert("L")
    data = im.getdata()
    data = np.array(data).reshape(1,784)
    a = (data-np.min(data))/(np.max(data)-np.min(data))
    result = sess.run(y, feed_dict={x: a})
    print(result)


