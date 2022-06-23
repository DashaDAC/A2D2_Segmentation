import json
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os

DIR_MASK_LIST_ = '/home/darya/cityscapes'
dir_files = os.listdir(DIR_MASK_LIST_)
print('folders: ', dir_files)
count_files = []
files_json = []
dir_f = []

for j in range(len(dir_files)):
    PATH_MASK_LIST = '/home/darya/cityscapes/' + str(dir_files[j])
    files = os.listdir(PATH_MASK_LIST)
    files_json.append([i for i in files if i.endswith('.json')])
    count_files.append(len(files_json[j]))
print('json', files_json)
print(count_files)
print(sum(count_files))
sum_files = sum(count_files)

app_label_list = []
all_list = []
uni_all_label = []
uni_app_label_list = []

for j in range(len(dir_files)):
    dir_f = files_json[j]
    print(j)
    for i in range(count_files[j]):
        with open("example_/" + str(dir_files[j]) + str('/') + str(dir_f[i])) as cs_json:
            json_data = json.load(cs_json)

        label_list = []
        label_list_uni = []
        objects = json_data['objects']

        for label in objects:
            element = label['label']
            label_list.append(element)
            label_list_uni.append(element)
        uni_class_json = list(set(label_list_uni))
        all_list = all_list + app_label_list
        app_label_list = uni_class_json
all_class = Counter(all_list)
print('All class', all_class)
class_CS = ['pole', 'car', 'traffic sign', 'traffic light', 'road', 'bus', 'truck']
col_class_CS = []

for i in range(len(class_CS)):
    val = all_class[str(class_CS[i])]
    col_class_CS.append(val)
print(type(col_class_CS))
col_class_CS = np.array(col_class_CS)
print('Counter', np.around(col_class_CS/sum_files*100, decimals = 2))

colors = ("#326da8", "#326da8", "#326da8", "#326da8", "#ff2929", "#326da8", "#326da8", "#326da8", "#326da8")
index = np.arange(7)
fig, ax = plt.subplots()
ax.barh(index, np.around(col_class_CS/sum_files*100, decimals = 2), color = colors)

plt.yticks(index, ['pole', 'car', 'traffic sign', 'traffic light', 'road', 'bus', 'truck'])
ax.set_yticklabels(['pole', 'car', 'traffic sign', 'traffic light', 'road', 'bus', 'truck'],
                   fontsize = 15)
plt.title("Cityscapes. Percentage of images containing some classes", size = 22,  position=(0.47, 5))

fig.set_figwidth(12)
fig.set_figheight(8)
plt.axvline(100, color='k', linestyle='dashed', linewidth=1)
plt.savefig('hist.png' , bbox_inches="tight")
for key, value in all_class.items():
    print(value)




