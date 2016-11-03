import os
import glob
import cv2

path = 'test'
new_path = 'resized'

file_names=glob.glob(os.path.join(path,'*','train_*','*.png'))
no_of_files=len(file_names)

unique_folders = glob.glob(os.path.join(path,'*','train_*/'))

if not os.path.isdir(new_path):
    os.mkdir(new_path)

for ele in unique_folders:
    j,k,l = (ele.split('/')[1:])
    new_loc = os.path.join(new_path,j,k,l)
    if(not os.path.isdir(new_loc)):
        os.mkdir("/".join(new_loc.split('/')[:2]))
        os.mkdir(new_loc)

for i in range(no_of_files):
    temp = cv2.resize(cv2.imread(file_names[i],0), (28,28))
    j,k,l = (file_names[i].split('/')[1:])
    new_loc = os.path.join(new_path,j,k,l)
    print(file_names[i], new_loc)
    # cv2.imshow(new_loc, temp)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(new_loc, temp)