import time
import cv2
import os
import glob

path = 'by_class'

t1 = time.time()
file_names=glob.glob(os.path.join('hsf_4','*.[pP][nN][gG]'))
t2 = time.time()
print('Time to list files: ', t2-t1)

file_classes=[ele.split('/')[1] for ele in file_names]
t3 = time.time()
print('Time to list labels: ', t3-t2)


# for i in range(len(file_names)):
# 	print(file_names[i], file_classes[i])

images = [cv2.imread(file) for file in file_names]
t4 = time.time()
print('Time to read images: ',t4-t3)

for image in images:
    cv2.imshow('temp', image)
    if cv2.waitKey()==ord('q'):
        break