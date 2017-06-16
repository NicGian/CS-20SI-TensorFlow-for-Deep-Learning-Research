x = tf.Variable(2.0, tf.float32)
y = 2.0 * (x ** 3) #y=2*x^3
z = 3.0 + y ** 2 #z=3+4*x^6
grad_z = tf.gradients(z, [x, y])   #dz/dx @ x=2 : 6*4*x^5 = 24*32 = 768    #dz/dx @ y=2*2^3=16 : 2*y= 32
with tf.Session() as sess:
sess.run(x.initializer)
print sess.run(grad_z) # >> [768.0, 32.0]
# 768 is the gradient of z with respect to x, 32 with respect to y