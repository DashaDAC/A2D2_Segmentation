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

path = '/home/darya/a2d2/camera_lidar_semantic'
dir_files = os.listdir(path)
dir_path.append([i for i in dir_files if i.startswith('2')])
print(dir_path)
len_dir = len(dir_path)
for j in range(6):
    mask_dir = path + '/' + str(dir_path[0][j]) + '/label/cam_front_center'
    print(mask_dir)
    mask_files = os.listdir(mask_dir)
    print(mask_files)

    len_list = len(mask_files)
    print(len_list)
    for i in range(len_list):

        img_mask = Image.open(mask_dir + '/' + mask_files[i])
        # pixels = sum(img_mask, [])
        # print(pixels)
        pixels = list(img_mask.getdata())
        road = pixels.count((255, 0, 255))
        d_line = pixels.count((255, 193, 37))
        s_line = pixels.count((128, 0, 255))
        non_drav_street = pixels.count((139, 99, 108))
        traf_sign = pixels.count((0, 255, 255)) + pixels.count((30, 220, 220)) +pixels.count((60, 157, 199))
        traf_signal = pixels.count((30, 28, 158)) + pixels.count((60, 28, 100)) + pixels.count((0, 128, 255))
        car = pixels.count((255, 0, 0)) + pixels.count((200, 0, 0)) +pixels.count((150, 0, 0)) + pixels.count((128, 0, 0))
        poles = pixels.count((255, 246, 143))
        col_traf_sign = traf_sign + col_traf_sign
        col_non_dstr = col_non_dstr + non_drav_street
        col_car = col_car + car
        col_road = col_road+road
        col_dash = col_dash + d_line
        col_solid = col_solid + s_line
        col_poles = poles + col_poles
        col_traf_signal = col_traf_signal + traf_signal
        print(col_road)
        print(col_solid)
        print(col_dash)
        print(i)
        # print(pixels)
        # pixels_list = pixels_list + pixels
    print(pixels_list)
# count_list = Counter(pixels_list)
# print(count_list)
print("col road", col_road)
print("col solid", col_solid)
print("col dash", col_dash)
print("col car", col_car)
print("col non dr street ", col_non_dstr)
print('col traf sign', col_traf_sign)
print('col traf signal', col_traf_signal)
print('col poles', col_poles)

