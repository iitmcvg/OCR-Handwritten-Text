import time
import os
import glob
import tensorflow as tf
import numpy as np
import cv2

# path = 'by_class'
path = 'test'

t1 = time.time()
# file_names = tf.train.match_filenames_once(path+'*/train_*/*.png')
# Originally done like this:
file_names=glob.glob(os.path.join(path,'*','train_*','*.[pP][nN][gG]'))
# filename_queue = tf.train.string_input_producer(file_names)
no_of_files=len(file_names)
t2 = time.time()
#print(file_names[0])
print('Time to list files: ', t2-t1)

print('No of files: ',no_of_files)



unique_classes = [int(ele.split('/')[1], base=16) for ele in glob.glob(os.path.join(path,'*/'))]
no_of_classes = len(unique_classes)

labels=[int(ele.split('/')[1], base=16) for ele in file_names]

try:
    label_names = [str(chr(i)) for i in labels]    #python 3
except:
    label_names = [str(unichr(i)) for i in labels]    #python 2.7    

label_encoding = dict()
for idx in range(len(unique_classes)):
    try:
        label_encoding[str(chr(unique_classes[idx]))] = idx
    except:
        label_encoding[str(unichr(unique_classes[idx]))] = idx

print('No of classes: ', no_of_classes)
print('Class encoding: ', label_encoding)



labels_oneHotEncoded = np.zeros((len(file_names),no_of_classes))
for k in range(no_of_files):
	labels_oneHotEncoded[k,label_encoding[label_names[k]]]=1

# tf.convert_to_tensor(label_names.eval())
t3 = time.time()
print('Time to list labels: ', t3-t2)
#labels_oneHotEncoded=tf.cast(labels_oneHotEncoded,tf.float32, name=None)			#converting it to float32


# reader = tf.WholeFileReader()
# key, value = reader.read(filename_queue)
# my_img = tf.image.decode_png(value, channels=1,dtype=float32) # use png or jpg decoder based on your files.

# init_op = tf.initialize_all_variables()

sess =  tf.Session()
# sess.run(init_op)

# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

images = []
# for i in range(len(labels)): #length of your filename list
#   images.append(my_img.eval(session = sess).ravel() ) #here is your image Tensor :)
  #images.append(my_img.eval(session = sess)) #here is your image Tensor :)

for i in range(no_of_files):
	# images.append(cv2.imread(file_names[i],0))
	a=(cv2.imread(file_names[i],0))
	a=np.array(a,dtype=np.float32)
	images.append(a)

for i in range(no_of_files):
	images[i]=images[i].reshape(-1)
#images=tf.to_float(images, name='ToFloat')						#Converting it to float32
#images=tf.reshape(images, [len(images), 128*128])      
#images=tf.cast(images, tf.float32, name=None)
images = np.array(images)
coord.request_stop()
coord.join(threads)

t4 = time.time()
print('Time to read images: ',t4-t3)
# Takes about  seconds to read test folder on my 4GB PC :-D
# And the code works!!




print('labels : ',labels_oneHotEncoded)
print('column size : ',images[1].shape)
print('no. of images :', len(images))
cv2.namedWindow('Input',0)
images=images*1.0/255.0;
# for i in range(128*128) :
#   if(images[0][i]<1):
#       print images[0][i]
print('non zero :',np.count_nonzero(images[0])) 

# images2=tf.convert_to_tensor(images)

# while(True) :
#   for i in range (10) :
#       cv2.imshow('Input',images[60*i].reshape(128,128))
#       if cv2.waitKey(100) & 0xFF == ord('q'):
#           break
#   if cv2.waitKey(100) & 0xFF == ord('e'):
#       break       
x = tf.placeholder(tf.float32, shape=[None, 128*128])
y_ = tf.placeholder(tf.float32, shape=[None, 4])
W = tf.Variable(tf.zeros([128*128, 4]))
b = tf.Variable(tf.zeros([4]))

# init = tf.initialize_all_variables()
y = tf.matmul(x,W) + b

# sess.run(tf.initialize_all_variables())

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,128,128,1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


# for i in range(20000):
#     batch = images.train.next_batch(97)
#     if i%100 == 0:
#         train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
#         print("step %d, training accuracy %g"%(i, train_accuracy))
#     train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())

train_accuracy = accuracy.eval(session=sess,feed_dict={x:images, y_: labels_oneHotEncoded, keep_prob: 1.0})
train_step.run(session=sess,feed_dict={x: images, y_: labels_oneHotEncoded, keep_prob: 0.5})
print("step %d, training accuracy %g"%(0, train_accuracy))

print("test accuracy %g"%accuracy.eval(feed_dict={x: images, y_: labels_oneHotEncoded, keep_prob: 1.0}))

# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


#Training the NN
#sess.run(train_step, feed_dict={x: images, y_: labels_oneHotEncoded})



#print(sess.run(accuracy, feed_dict={x: images, y_: labels_oneHotEncoded}))

sess = tf.Session()
sess.run(init)


#print(correct_prediction)

# print "\nReached"
# print(correct_prediction)
# print "\nReached....."

# while(True) :
# 	cv2.imshow('Input',x[0].reshape(128,128))
# 	if cv2.waitKey(100) & 0xFF == ord('q'):
#  			break
print W[0]
print x[0]
#print(len(label_names[1,:]))
#print(np.nonzero(label_names))
