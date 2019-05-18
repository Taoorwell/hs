from python_gdal import *

cnn_1d_path = r'F:/paper/HSI/code/predicts_mat/cnn_1d/'
cnn_2d_path = r'F:/paper/HSI/code/predicts_mat/cnn_2d/'

cnn_1d_list = os.listdir(cnn_1d_path)
cnn_2d_list = os.listdir(cnn_2d_path)

print(cnn_1d_list[-1])
print(cnn_2d_list[3])


def get_predictions_from_mat(files):
    pre_dict = sio.loadmat(files)
    prediction = pre_dict[list(pre_dict.keys())[-1]]
    return prediction


cnn_1d_pre = get_predictions_from_mat(cnn_1d_path+cnn_1d_list[-1])

cnn_2d_pre = get_predictions_from_mat(cnn_2d_path+cnn_2d_list[3])

print(cnn_1d_pre.shape)
print(cnn_2d_pre.shape)

fusion_pre = []
# for p1, p2 in zip(cnn_1d_pre, cnn_2d_pre):
#     c1 = np.argmax(p1)
#     c2 = np.argmax(p2)
#     if c1 == c2:
#         fusion_pre.append(p2)
#     else:
#         m1 = np.max(p1)
#         m2 = np.max(p2)
#         if m1 < m2:
#             fusion_pre.append(p2)
#         else:
#             fusion_pre.append(p1)
for p1, p2 in zip(cnn_1d_pre, cnn_2d_pre):
    m1 = np.max(p1)
    m2 = np.max(p2)
    if m1 < m2:
        fusion_pre.append(p2)
    else:
        fusion_pre.append(p1)
print(len(fusion_pre))
fusion_pre = np.stack(fusion_pre, axis=0)
print(fusion_pre.shape)

# display result!!
fig = plt.figure(num=1, figsize=(6, 4), dpi=100)
plt.subplot(131)
write_whole_image_classification_result(predict=fusion_pre, shape=(610, 340))
# plt.show()
plt.subplot(132)
write_whole_image_predicts_prob(predict=fusion_pre,shape=(610, 340))

plt.subplot(133)
cof = write_whole_image_predicts_prob1(predict=fusion_pre, shape=(610, 340))
print(cof)

plt.show()
