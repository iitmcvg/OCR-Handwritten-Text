import time
import os
import glob
import tensorflow as tf
import numpy as np

# path = 'by_class'
path = 'test'

t1 = time.time()
file_names=glob.glob(os.path.join(path,'*','train_*','*.[pP][nN][gG]'))
filename_queue = tf.train.string_input_producer(file_names)
t2 = time.time()
print('Time to list files: ', t2-t1)

file_classes=[int(ele.split('/')[1], base=16) for ele in file_names]
try:
    file_labels = [str(chr(i)) for i in file_classes]    #python 3
except:
    file_labels = [str(unichr(i)) for i in file_classes]    #python 2.7    

# file_labels=np.array(file_labels)
# tf.convert_to_tensor(file_labels.eval())
t3 = time.time()
print('Time to list labels: ', t3-t2)

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
my_img = tf.image.decode_png(value) # use png or jpg decoder based on your files.

init_op = tf.initialize_all_variables()

sess =  tf.Session()
sess.run(init_op)

# Start populating the filename queue.

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

for i in range(len(file_classes)): #length of your filename list
  image = my_img.eval(session = sess).ravel() #here is your image Tensor :)

coord.request_stop()
coord.join(threads)
t4 = time.time()
print('Time to read images: ',t4-t3)
# Takes about  seconds to read test folder on my 4GB PC :-D
# And the code works!!
#Input the images from mnist to tensorflow

file_labels = np.zeros((len(file_labels),10))

x = tf.placeholder(tf.float32, [None, 128*128*3],name="x")
W = tf.Variable(tf.zeros([128*128*3, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

#Training the NN
sess.run(train_step, feed_dict={x: image.reshape(1,image.shape[0]), y_: file_labels})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(correct_prediction)
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))