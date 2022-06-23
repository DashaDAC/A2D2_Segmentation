from collections import Counter
import os
from PIL import Image

pixels_list = []
dir_path = []

path = '/home/darya/a2d2/camera_lidar_semantic'
dir_files = os.listdir(path)
dir_path.append([i for i in dir_files if i.startswith('2')])
print(dir_path)
len_dir = len(dir_path[0])
print(len_dir)
for j in range(len_dir):
    mask_dir = path + '/' + str(dir_path[0][j]) + '/label/cam_front_center'
    print(mask_dir)
    mask_files = os.listdir(mask_dir)
    print(mask_files)

    len_list = len(mask_files)
    print(len_list)
    for i in range(len_list):

        img_mask = Image.open(mask_dir + '/' + mask_files[i])
        pixels = list(set(img_mask.getdata()))
        print(i)
        pixels_list = pixels_list + pixels
    print(pixels_list)
count_list = Counter(pixels_list)
print(count_list)
