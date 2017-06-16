# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import tensorflow as tf
x = tf.constant(2, name='a')
y = tf.constant(3, name='b')
z = tf.add(x,y)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(z))
writer.close()


#$ tensorboard --logdir="./graphs" --port 6006
#Then open your browser and go to: http://localhost:6006/


my_const = tf.constant([1.0, 2.0], name="my_const")
with tf.Session() as sess:
    print sess.graph.as_graph_def()
# you will see value of my_const stored in the graph’s definition

#a variable vector
a = tf.Variable([12,12], name='vector')
b = tf.Variable(2, name = 'scalar')
# create variable c as a 2x2 matrix
c = tf.Variable([1,2],[3,4], name = 'matrix')

init = tf.global_varialbles_initializer()
with tf.Session as sess:
    sess.run(init)
    
    
#create a float32 placeholder
a = tf.placeholder(tf.float32, shape=[3])
# create a constant of type float 32-bit, shape is a vector of 3 elements
b = tf.constant([5, 5, 5], tf.float32)
# use the placeholder as you would a constant or a variable
c = a + b # Short for tf.add(a, b)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(c, {a:[1,2,3]}))