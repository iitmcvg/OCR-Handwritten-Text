import time
import os
import glob
import tensorflow as tf

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
  image = my_img.eval(session = sess) #here is your image Tensor :)

coord.request_stop()
coord.join(threads)
t4 = time.time()
print('Time to read images: ',t4-t3)
