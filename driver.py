import tensorflow as tf
import numpy as np
from autocorrect import spell
import pyttsx as tell
import cv2

# load hyper parameters

no_of_classes = 26

x = tf.placeholder(tf.float32, shape=[1,28*28], name ='x')
W = tf.Variable(tf.zeros([28*28, no_of_classes]), name = 'W')
b = tf.Variable(tf.zeros([no_of_classes]), name = 'b')
prediction = tf.Variable(1, name = 'prediction')
y = tf.nn.softmax(tf.nn.sigmoid(tf.matmul(x, W) + b), name = 'y')

# print '23'

Win = np.loadtxt('W_bro')
Bin = np.loadtxt('b_bro')

print 'w shape :' , Win.shape
print 'b shape :' , Bin.shape


W = tf.convert_to_tensor(Win, dtype=tf.float32)
b = tf.convert_to_tensor(Bin, dtype=tf.float32)

print Win.shape

init_op = tf.initialize_all_variables()

# print sess.run(b)


# print sess.run(W)

# saver.restore(sess, "/Users/aravinth_muthu/Desktop/OCR-Handwritten-Text-master/model.ckpt")

img = cv2.imread("2.png", 0)
img = np.array(img,dtype=np.float32)
img = img.reshape(1,784)
x = img
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

print W.eval(session = sess),y.eval(session=sess)

print 'img shape' , img.shape

prediction = sess.run(y, feed_dict={x : x})
print y

# print sess.run(prediction)
# # engine = tell.init()
# engine.say()
# engine.runAndWait()

# for i in range(len(a)):
# 	a[i] = spell(a[i])