"""
Simple TensorFlow exercises
You should thoroughly test your code
"""

import tensorflow as tf

###############################################################################
# 1a: Create two random 0-d tensors x and y of any distribution.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond()
# I do the first problem for you
###############################################################################

x = tf.random_uniform([])  # Empty array as shape creates a scalar.
y = tf.random_uniform([])
out = tf.cond(tf.greater(x, y), lambda: tf.add(x, y), lambda: tf.sub(x, y))

###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from -1 and 1.
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################

# YOUR CODE
x = tf.random_uniform([],minval=-1,maxval=1)
y = tf.random_uniform([],minval=-1,maxval=1)

out = tf.case({tf.less(y,x): lambda: tf.add(x,y), 
               tf.less(y,x): lambda: tf.subtract(x,y)}, 
               lambda: tf.constant(0.0), exclusive=True)

with tf.Session() as sess:
  print (sess.run([x, y, out]))

###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]] 
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################

# YOUR CODE
x = tf.constant([[0, -2, -1], [0, 1, 2]])
y = tf.zeros_like(x)
out = tf.equal(x,y)

with tf.Session() as sess:
  print (sess.run([x, y, out]))

###############################################################################
# 1d: Create the tensor x of value 
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################

# YOUR CODE
x = tf.constant([29.05088806,  27.61298943,  31.19073486,  29.35532951,
  30.97266006,  26.67541885,  38.08450317,  20.74983215,
  34.94445419,  34.45999146,  29.06485367,  36.01657104,
  27.88236427,  20.56035233,  30.20379066,  29.51215172,
  33.71149445,  28.59134293,  36.05556488,  28.66994858])

high_index = tf.where(x > 30)
high_elem = tf.gather(x,(tf.where(x > 30)))

with tf.Session() as sess:
  print (sess.run([x, high_index, high_elem]))

###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################

# YOUR CODE
diag_ = tf.range(1,7)
x = tf.diag(diag_)
with tf.Session() as sess:
  print (sess.run([x]))

###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################

# YOUR CODE
x = tf.random_normal([10,10], stddev=1.0)
det = tf.matrix_determinant(x)

with tf.Session() as sess:
  print (sess.run([x,det]))
  
###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################

# YOUR CODE
x=tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])
unique_elem = tf.unique(x)[1]

with tf.Session() as sess:
  print (sess.run([x,unique_elem]))

###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution.
# Use tf.less() and tf.select() to return:
# - The mean squared error of (x - y) if the average of all elements in (x - y)
#   is negative, or
# - The sum of absolute value of all elements in the tensor (x - y) otherwise.
# Hint: see the Huber loss function in the lecture slides 3.
###############################################################################

# YOUR CODE
x = tf.random_normal([300], stddev=2.0)
y = tf.random_normal([300], stddev=3.0)
sub_mean = tf.reduce_mean(tf.subtract(x,y))
mse = tf.reduce_mean(tf.square(tf.subtract(x,y)))
sum_abs = tf.reduce_sum(tf.abs(tf.subtract(x,y)))
    
out = tf.where(tf.less(sub_mean,0), mse, sum_abs)

with tf.Session() as sess:
  print (sess.run([out, mse, sum_abs, sub_mean]))
  
  
  
  