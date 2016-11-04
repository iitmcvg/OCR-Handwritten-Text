import time
import os
import glob
import tensorflow as tf
import numpy as np
import cv2
import random

# path = 'by_class'
# path = 'final'
batch_size=100
path = 'by_class'

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
ab = range(no_of_files)
random.shuffle(ab)

test_size = int(0.2*no_of_files)
train_size = no_of_files-test_size
#images=tf.to_float(images, name='ToFloat')						#Converting it to float32
#images=tf.reshape(images, [len(images), 128*128])      
#images=tf.cast(images, tf.float32, name=None)
images = np.array(images)
test_images = images[ab[:test_size]]*1./255.
test_labels = labels_oneHotEncoded[ab[:test_size]] 

images=images[ab[test_size:]]
labels_oneHotEncoded=labels_oneHotEncoded[ab[test_size:]]

coord.request_stop()
coord.join(threads)

t4 = time.time()
print('Time to read images: ',t4-t3)
# Takes about  seconds to read test folder on my 4GB PC :-D
# And the code works!!

# x = tf.placeholder(tf.float32, shape=[None, 28*28])
# y_ = tf.placeholder(tf.float32, shape=[None, no_of_classes])

# W = tf.Variable(tf.zeros([28*28, 200]))
# b = tf.Variable(tf.zeros([200]))
# h1 = tf.nn.sigmoid(tf.matmul(x,W) + b)
# W2 = tf.Variable(tf.zeros([200, 50]))
# b2 = tf.Variable(tf.zeros([50]))
# h2 = tf.nn.sigmoid((tf.matmul(h1, W2) + b2))
# W3 = tf.Variable(tf.zeros([50, no_of_classes]))
# b3 = tf.Variable(tf.zeros([no_of_classes]))
# y = tf.nn.softmax((tf.matmul(h2, W3) + b3))
x = tf.placeholder(tf.float32, shape=[None, 28*28])
y_ = tf.placeholder(tf.float32, shape=[None, no_of_classes])

W = tf.Variable(tf.random_normal([28*28, 200], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
b = tf.Variable(tf.random_normal([200], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
h1 = tf.nn.sigmoid(tf.matmul(x,W) + b)
W2 = tf.Variable(tf.random_normal([200, 50], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
b2 = tf.Variable(tf.random_normal([50], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
h2 = tf.nn.sigmoid((tf.matmul(h1, W2) + b2))
W3 = tf.Variable(tf.random_normal([50, no_of_classes], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
b3 = tf.Variable(tf.random_normal([no_of_classes], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None))
y = tf.nn.softmax((tf.matmul(h2, W3) + b3))


# W = tf.Variable(tf.zeros([28*28, no_of_classes]))
# b = tf.Variable(tf.zeros([no_of_classes]))
# y = tf.nn.softmax(tf.nn.sigmoid(tf.matmul(x, W) + b))

saver=tf.train.Saver([b])

print('labels : ',labels_oneHotEncoded)
print('column size : ',len(images[0]))
print('no. of images :', len(images))
cv2.namedWindow('Input',0)
images=images*1.0/255.0;
# for i in range(128*128) :
# 	if(images[0][i]<1):
# 		print images[0][i]
print('non zero :',np.count_nonzero(images[0])) 

# images2=tf.convert_to_tensor(images)

# while(True) :
# 	for i in range (10) :
# 		cv2.imshow('Input',images[60*i].reshape(128,128))
# 		if cv2.waitKey(100) & 0xFF == ord('q'):
# 			break
# 	if cv2.waitKey(100) & 0xFF == ord('e'):
# 		break		
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1,use_locking=False).minimize(cross_entropy)


# global_step = tf.Variable(0, trainable=False)
# starter_learning_rate = 0.5
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
#                                            10, 0.95, staircase=True)
# train_step = (tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step))

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
# Passing global_step to minimize() will increment it at each step.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for i in range(10000):
    rand_idx = random.sample(range(train_size), batch_size)
    batch_x, batch_y = images[rand_idx], labels_oneHotEncoded[rand_idx]
    
    #Training the NN
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
    if(i%100==0):
        print('Iteration {:} done'.format(i))
        print('training accuracy')
        print(sess.run(accuracy, feed_dict={x: images, y_: labels_oneHotEncoded}))
        print('test accuracy')
        print(sess.run(accuracy, feed_dict={x: test_images, y_: test_labels}))

#Training the NN

#print(correct_prediction)

# print "\nReached"
# print(correct_prediction)
# print "\nReached....."
# print('training accuracy')
# print(sess.run(accuracy, feed_dict={x: images, y_: labels_oneHotEncoded}))
# print('test accuracy')
# print(sess.run(accuracy, feed_dict={x: test_images, y_: test_labels}))

# while(True) :
# 	cv2.imshow('Input',x[0].reshape(128,128))
# 	if cv2.waitKey(100) & 0xFF == ord('q'):
#  			break
print correct_prediction.get_shape()
print(sess.run(W))
# print x[0]
np.savetxt('testW.out', sess.run(W), delimiter=',')
np.savetxt('testb.out', sess.run(b), delimiter=',')
np.savetxt('testW2.out', sess.run(W2), delimiter=',')
np.savetxt('testb2.out', sess.run(b2), delimiter=',')
np.savetxt('testW3.out', sess.run(W3), delimiter=',')
np.savetxt('testb3.out', sess.run(b3), delimiter=',')
save_path=saver.save(sess,"/home/ganga/Documents/model.ckpt")
#print(len(label_names[1,:]))
#print(np.nonzero(label_names))
