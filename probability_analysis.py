# from python_gdal import *
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

Folder_name = r"E:/HSI/code/predicts_mat/"
Sub_Folder_list = os.listdir(Folder_name)

print(Sub_Folder_list)

cnn_1d = os.listdir(Folder_name + Sub_Folder_list[0])
cnn_2d = os.listdir(Folder_name + Sub_Folder_list[1])
mlp = os.listdir(Folder_name + Sub_Folder_list[2])

# print(cnn_1d)
print(cnn_2d)
cnn_1d_IP_dict = sio.loadmat(Folder_name + Sub_Folder_list[0] + '/' + cnn_1d[3])
cnn_1d_IP = cnn_1d_IP_dict[list(cnn_1d_IP_dict.keys())[-1]]
print(cnn_1d_IP.shape)
cnn_2d_dict = sio.loadmat(Folder_name + Sub_Folder_list[1] + '/' + cnn_2d[11])
cnn_2d_IP_13 = cnn_2d_dict[list(cnn_2d_dict.keys())[-1]]
print(cnn_2d_IP_13.shape)
shape = (610, 340)

fig = plt.figure(num=1, figsize=(8, 6))
ax1 = plt.subplot2grid((2, 2), (0, 0))

prob_img = np.max(cnn_1d_IP, axis=1).reshape(shape)
sn.heatmap(prob_img, annot=False, cmap="Greys_r", xticklabels=False, yticklabels=False)
ax1.imshow(prob_img, cmap='gray')
ax1.axis('off')
# ax1.set_title('cnn 1d max probability')

ax2 = plt.subplot2grid((2, 2), (0, 1))

conf = np.max(cnn_1d_IP, axis=1)
conf = [x for x in conf if x < 0.5]
ax2.hist(conf, bins=10, range=(0, 0.5), facecolor='red', alpha=0.5)
ax2.yaxis.tick_right()
ax2.spines['top'].set_visible(False)
# ax2.spines['right'].set_visible(False)
# ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)


ax3 = plt.subplot2grid((2, 2), (1, 0))

prob_img = np.max(cnn_2d_IP_13, axis=1).reshape(shape)
sn.heatmap(prob_img, annot=False, cmap="Greys_r", xticklabels=False, yticklabels=False)
ax3.imshow(prob_img, cmap='gray')
ax3.axis('off')
# ax3.set_title('cnn 2d max probability')

ax4 = plt.subplot2grid((2, 2), (1, 1))

conf = np.max(cnn_2d_IP_13, axis=1)
conf = [x for x in conf if x < 0.5]
ax4.hist(conf, bins=10, range=(0, 0.5), facecolor='red', alpha=0.5)
ax4.yaxis.tick_right()
ax4.spines['top'].set_visible(False)
# ax4.spines['right'].set_visible(False)
# ax4.spines['bottom'].set_visible(False)
ax4.spines['left'].set_visible(False)

plt.show()
