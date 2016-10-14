import cv2
import os
import glob

path = 'by_class'

file_names=glob.glob(os.path.join(path,'*','train_*','*.[pP][nN][gG]'))
for ele in file_names:
	print(ele)