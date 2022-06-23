import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

pixels_list = []
dir_path = []

col_road = 0
col_dash = 0
col_solid = 0
col_car = 0
col_non_dstr = 0
col_traf_sign = 0
col_poles = 0
col_traf_signal = 0
col_bus = 0
col_truck = 0
len_dir = 0
len_all = 0

path = ['/home/darya/cityscapes/gtFine/train', '/home/darya/cityscapes/gtFine/val']
# dir_files = [os.listdir(path[0]), os.listdir(path[1])]
# print(dir_files[1])
# dir_path.append([i for i in dir_files if i.startswith('2')])
# print(dir_path)
len_d = [7, 3]
for i in range(2):
    path1 = path[i]
    print(path1)
    dir_files = os.listdir(path1)
    print(dir_files)
    len_dir = len(dir_files)
    for j in range(len_d[i]):
        mask_path = []
        print(str(path1) + '/' + str(dir_files[j]))
        mask_dir = str(path1) + '/' + str(dir_files[j])
        print("mask dir", mask_dir)
        mask_files = os.listdir(mask_dir)
        print("mask files", mask_files)
        col_mask_files = len(mask_files)
        print(col_mask_files)
        mask_path.append([i for i in mask_files if i.endswith('labelIds.png')])
        print("mask path", mask_path)

        len_list = int(col_mask_files/4)
        len_all = len_all + len_list
        for i in range(len_list):

            print(mask_dir + '/' + mask_path[0][i])
            img_mask = Image.open(mask_dir + '/' + mask_path[0][i])
            # print(img_mask)
            pixels = list(img_mask.getdata())
            road = pixels.count(7)
            truck = pixels.count(27)
            bus = pixels.count(28)
            traf_sign = pixels.count(20)
            traf_signal = pixels.count(19)
            car = pixels.count(26)
            poles = pixels.count(17)
            col_traf_sign = traf_sign + col_traf_sign
            col_truck = col_truck + truck
            col_car = col_car + car
            col_road = col_road+road
            col_bus = col_bus + bus
            col_poles = poles + col_poles
            col_traf_signal = col_traf_signal + traf_signal
            print(i)
        print(pixels_list)
print("col road", col_road)
print("col truck", col_truck)
print("col bus", col_bus)
print("col car", col_car)
print('col traf sign', col_traf_sign)
print('col traf signal', col_traf_signal)
print('col poles', col_poles)
print(len_all)
