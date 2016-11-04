import tensorflow as tf
import numpy as np
from autocorrect import spell
# import pyttsx as tell
import cv2

# load hyper parameters

no_of_classes = 26

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[None,28*28], name ='x')
W = tf.Variable(tf.zeros([28*28, no_of_classes]), name = 'W')
b = tf.Variable(tf.zeros([no_of_classes]), name = 'b')
y = tf.nn.softmax(tf.nn.sigmoid(tf.add(tf.matmul(x, W), b)), name = 'y')

# print '23'

Win = np.loadtxt('W_bro')
Bin = np.loadtxt('b_bro')

print('w shape :' , Win.shape)
print('b shape :' , Bin.shape)


W = tf.convert_to_tensor(Win, dtype=tf.float32)
b = tf.convert_to_tensor(Bin, dtype=tf.float32)

init_op = tf.initialize_all_variables()
# saver.restore(sess, "/Users/aravinth_muthu/Desktop/OCR-Handwritten-Text-master/model.ckpt")

img = cv2.imread("2.png", 0)
img = np.array(img,dtype=np.float32)
print('Printing the image itself', img)
img = img.reshape(1,784)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# print(W.eval(session = sess),y.eval(session=sess))
prediction = sess.run(y, feed_dict={x : img})
print(y.eval(session=sess, feed_dict={x:img}))
# print('y Eval: ',y.eval(session = sess))
# print('Just y: ',y)
# print('y shape: ',y.eval(session = sess).shape)
# print('y shape2: ',tf.shape(y))

# prediction = tf.Variable(1, name = 'prediction')
# print sess.run(prediction)
# # engine = tell.init()
# engine.say()
# engine.runAndWait()

# for i in range(len(a)):
# 	a[i] = spell(a[i])