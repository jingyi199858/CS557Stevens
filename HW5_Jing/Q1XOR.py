import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])

x = tf.placeholder(tf.float32, [None, 2])
y = tf.placeholder(tf.float32, [None, 1])

#initiate weight
w1 = tf.Variable(tf.random_uniform([2,2]))
w2 = tf.Variable(tf.random_uniform([2,1]))

#initiate bias
b1=tf.Variable(tf.zeros([2]))
b2=tf.Variable(tf.zeros([1]))

h=tf.nn.relu(tf.matmul(x,w1)+b1)
out=tf.matmul(h,w2)+b2

loss = tf.reduce_mean(tf.square(out - y))

#using adamoptimizer with laerning_rate of 0.01
train = tf.train.AdamOptimizer(0.01).minimize(loss)

cost_his = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        cost = sess.run(train, feed_dict={x: X, y: Y})
        loss_ = sess.run(loss, feed_dict={x: X, y: Y})
        cost_his.append(cost)
        if i%500==0 :
            print("step: %d, loss: %.3f"%(i, loss_))
    print("X: %r"%X)
    print("pred: %r"%sess.run(out, feed_dict={x: X}))