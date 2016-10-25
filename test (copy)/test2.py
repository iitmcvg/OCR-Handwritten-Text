import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

filename_queue = tf.train.string_input_producer(['./hsf_4/hsf_4_00000.png']) #  list of files to read

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_png(value) # use png or jpg decoder based on your files.

init_op = tf.initialize_all_variables()

sess =  tf.Session()
sess.run(init_op)

# Start populating the filename queue.

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord, sess=sess)

for i in range(1): #length of your filename list
  image = my_img.eval(session = sess) #here is your image Tensor :) 

print(image)
# plt.imshow(np.array(image))
# plt.show()

coord.request_stop()
coord.join(threads)