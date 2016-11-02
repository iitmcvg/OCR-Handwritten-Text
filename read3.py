import time
import os
import glob
import tensorflow as tf
import numpy as np
import cv2

# path = 'by_class'
# path = 'final'
path = 'data_final_31k'

t1 = time.time()
# file_names = tf.train.match_filenames_once(path+'*/train_*/*.png')
# Originally done like this:
file_names=glob.glob(os.path.join(path,'*','*.[pP][nN][gG]'))
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

x = tf.placeholder(tf.float32, shape=[None, 28*28])
y_ = tf.placeholder(tf.float32, shape=[None, no_of_classes])

# W = tf.Variable(tf.zeros([28*28, 100]))
# b = tf.Variable(tf.zeros([100]))
# h = tf.nn.sigmoid(tf.matmul(x,W) + b)
# W2 = tf.Variable(tf.zeros([100, 4]))
# b2 = tf.Variable(tf.zeros([4]))
# y = tf.nn.softmax((tf.matmul(h, W2) + b2))

W = tf.Variable(tf.zeros([28*28, no_of_classes]))
b = tf.Variable(tf.zeros([no_of_classes]))
y = tf.nn.softmax(tf.nn.sigmoid(tf.matmul(x, W) + b))

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

#train_step = tf.train.GradientDescentOptimizer(1,use_locking=False).minimize(cross_entropy)


global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 1.0
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           10, 0.8, staircase=True)


init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)
# Passing global_step to minimize() will increment it at each step.
train_step = (tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step))

#Training the NN
for i in range(50):
    sess.run(train_step, feed_dict={x: images, y_: labels_oneHotEncoded})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#print(correct_prediction)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# print "\nReached"
# print(correct_prediction)
# print "\nReached....."
print(sess.run(accuracy, feed_dict={x: images, y_: labels_oneHotEncoded}))

# while(True) :
# 	cv2.imshow('Input',x[0].reshape(128,128))
# 	if cv2.waitKey(100) & 0xFF == ord('q'):
#  			break
print correct_prediction.get_shape()
print(sess.run(W))
print x[0]
#print(len(label_names[1,:]))
#print(np.nonzero(label_names))
