import os
import glob
import cv2
import numpy as np

path = 'by_class'
new_path = 'resized'

file_names=glob.glob(os.path.join(path,'*','train_*','*.png'))
no_of_files=len(file_names)

unique_folders = glob.glob(os.path.join(path,'*','train_*/'))

if not os.path.isdir(new_path):
    os.mkdir(new_path)
# print(file_names,'yolo')
for ele in unique_folders:
    j,k,l = (ele.split('/')[1:])
    new_loc = os.path.join(new_path,j,k,l)
    if(not os.path.isdir(new_loc)):
        os.mkdir("/".join(new_loc.split('/')[:2]))
        os.mkdir(new_loc)

for i in range(no_of_files):
    img = cv2.imread(file_names[i],0)
    img_=img.copy()
    im2,contours,hier=cv2.findContours(img,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    areas=[cv2.contourArea(c) for c in contours]
    areas=filter(lambda x: x<0.9*img.shape[0]*img.shape[1],areas)
    if(len(areas)!=0):
        idx=np.argmax(areas)
        x,y,w,h=cv2.boundingRect(contours[idx])
        im=img_[y:y+h,x:x+w]
        im2=cv2.resize(im,(28,28))
        j,k,l = (file_names[i].split('/')[1:])
        new_loc = os.path.join(new_path,j,k,l)
        print(file_names[i], new_loc)
        # cv2.imshow(new_loc, temp)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(new_loc, im2)

