from python_gdal import *
import numpy as np
import sys
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import cross_val_score
# import tensorflow as tf
import time
import matplotlib.pyplot as plt
from models_keras import *

# # Data Preparation
MAIN_FOLDER = r'E:/HSI/'
IP_DATA_PATH = 'IP/Indian_pines_corrected'
IP_TRAIN_PATH = 'IP/Indian_pines_gt'
PAVIA_DATA_PATH = "Pavia/Pavia"
PAVIA_TRAIN_PATH = "Pavia/Pavia_gt"
PAVIA_U_DATA_PATH = "Pavia/PaviaU"
PAVIA_U_TRAIN_PATH = "Pavia/PaviaU_gt"
SALINAS_DATA_PATH = "Salinas/Salinas"
SALINAS_TRAIN_PATH = 'Salinas/Salinas_gt'
KSC_DATA_PATH = "KSC/KSC"
KSC_TRAIN_PATH = "KSC/KSC_gt"
IEEEHSI_2018_DATA_PATH = "2018IEEEHSI/FullHSIDataset/20170218_UH_CASI_S4_NAD83.pix"
IEEEHSI_2018_TRAIN_PATH = "2018IEEEHSI/GT"
# convert_to_color(train_data_path=MAIN_FOLDER+PAVIA_U_TRAIN_PATH)
c = [16, 9, 9, 13]
# m = 29
input_shape1 = [(200,), (102,), (103,), (176,)]
input_shape2 = [(200, 1), (102, 1), (103, 1), (176, 1)]
lists2 = [200, 200, 200, 200, 200, 200, 200, 200, 200]
# lists = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200]
lists1 = [33, 200, 200, 181, 200, 200, 20, 200, 14, 200, 200, 200, 143, 200, 200, 75]
lists3 = [200, 150, 150, 150, 100, 150, 50, 200, 200, 200, 200, 200, 200]
lists = [lists1, lists2, lists2, lists3]
image_shape = [(145, 145), (1096, 715), (610, 340), (512, 614)]
DATA_PATH = [IP_DATA_PATH, PAVIA_DATA_PATH, PAVIA_U_DATA_PATH, KSC_DATA_PATH]
TRAIN_PATH = [IP_TRAIN_PATH, PAVIA_TRAIN_PATH, PAVIA_U_TRAIN_PATH, KSC_TRAIN_PATH]
# name = ["mlp_IP.h5", "mlp_P.h5", "mlp_PU.h5", "mlp_KSC.h5"]
name = ["cnn_2d_pca_IP.h5", "cnn_2d_pca_P.h5", "cnn_2d_pca_PU.h5", "cnn_2d_pca_KSC.h5"]
model1 = ["cnn_1d_IP.h5", "cnn_1d_P.h5", "cnn_1d_PU.h5", "cnn_1d_KSC.h5"]
model2 = ["29-cnn_2d_IP.h5", "29-cnn_2d_P.h5", "29-cnn_2d_PU.h5", "29-cnn_2d_KSC.h5"]
# m = 29
# i = 2
path = "E:/HSI/code/new_model_2/"
save_path = "E:/HSI/code/predicts_mat/"
model_list = os.listdir(path)
model_path = [path + x for x in model_list]
# print(model_path)
for j in model_path:
    print(j)
    model = load_model(j)
    l1 = j.split('/')[-1].split('_')[-1].split('.')[0]
    print(l1)
    if l1 == 'IP':
        i = 0
    elif l1 == 'P':
        i = 1
    elif l1 == 'PU':
        i = 2
    else:
        i = 3
    m = int(j.split('/')[-1].split('-')[0])
    predicts = write_out_whole_predicts(model, MAIN_FOLDER+DATA_PATH[i], bsize=3200, m=m)
    filename = j.split('/')[-1].split('.')[0] + '.mat'
    save_array_to_mat(predicts, filename=save_path + filename)
    print(filename + ' save successful')

# (1096, 715), (145, 145), (512, 217), (512, 614)
# # #
# train_samples_1, train_labels_1 = get_train_sample(data_path=MAIN_FOLDER+DATA_PATH[i],
#                                                    train_data_path=MAIN_FOLDER+TRAIN_PATH[i],
#                                                    c=c[i], d=3, lists=lists[i], m=1)
# train_samples_2, train_labels_2 = get_train_sample(data_path=MAIN_FOLDER+DATA_PATH[i],
#                                                    train_data_path=MAIN_FOLDER+TRAIN_PATH[i],
#                                                    c=c[i], d=4, lists=lists[i], m=m)
# train_labels = one_hot_encode(c=c[i], labels=train_labels_1)
#
# model = feature_fusion_model(model1="cnn_1d_PU.h5", model2="model/29-cnn_2d_PU.h5",
#                              train_samples_1=train_samples_1, train_samples_2=train_samples_2,
#                              train_labels=train_labels, c=c[i])
# save_model(model, "29-feature_fusion_model.h5")

# model = load_model("29-feature_fusion_model.h5")
# # fusion_feature, y_test = get_fusion_features_from_test(model1=model1[i], model2="model/"+model2[i],
# #                                                        data_path=MAIN_FOLDER+DATA_PATH[i],
# #                                                        train_data_path=MAIN_FOLDER+TRAIN_PATH[i],
# #                                                        c=c[i], m=m, lists=lists[i])
# # pre = model.predict(fusion_feature)
# # print_plot_cm(y_test, pre)
# f3 = get_fusion_features_from_whole(model1=model1[i], model2="model/"+model2[i],
#                                     data_path=MAIN_FOLDER+DATA_PATH[i], m=m)
# pre = model.predict(f3)
# write_whole_image_classification_result(pre, shape=image_shape[i])
# write_whole_image_predicts_prob(pre, shape=image_shape[i])

# model = load_model("new_model_0/cnn_1d_PU.h5")

# # # # # # # # # # # # # # print(train_samples[0])
# x_train, x_valid, y_train, y_valid = train_test_split(train_samples, train_labels,
#                                                       test_size=0.1, shuffle=True,
#                                                       stratify=train_labels)
# y_train, y_valid = one_hot_encode(c=c[i], labels=y_train), one_hot_encode(c=c[i], labels=y_valid)
# # # # # # # # # # # # # # #
# model = cnn_2d_pca(input_shape=(m, m, n), c=c[i], lr=0.001, rate1=0.2, rate2=0.2, l=0.0001)
# model.summary()
# print(name[i].split('.')[0] + " Training Start!")
# network = model.fit(x_train, y_train, batch_size=30, epochs=50, verbose=1,
#                     validation_data=(x_valid, y_valid))
# print(name[i].split('.')[0] + " Training Finish!")
# plot_history(network)
# save_model(model, name[i])
# #
# model = load_model('43-cnn_2d_pca_IP.h5')
# model = load_model('model/3-43-cnn_2d_pca_IP.h5')
# model = load_model('model/29-cnn_2d_PU.h5')
# model = load_model('9-19-cnn_2d_pca_PU.h5')

# # # model.summary()
# # # print(model.history['loss'])
# # # config = model.get_config()
# # # print(config['output_layers'])
# # # model.summary()
# print(name[i].split('.')[0] + " Classification Report!")
# oa, kappa = get_test_predict(model=model, data_path=MAIN_FOLDER+DATA_PATH[i],
#                              train_data_path=MAIN_FOLDER+TRAIN_PATH[i], c=c[i], lists=lists[i],
#                              bsize=3200, m=9, n=9)
# write_train_region_image(model=model, data_path=MAIN_FOLDER+DATA_PATH[i],
#                          train_data_path=MAIN_FOLDER+TRAIN_PATH[i], shape=image_shape[i],
#                          bsize=16000, m=19, pca=True)
# print(image_shape[i])
# predicts = write_out_whole_predicts(model=model, data_path=MAIN_FOLDER+DATA_PATH[i],
#                                     bsize=32000)
# print(predicts.shape)
# predicts = predicts.reshape(image_shape[i][0]*image_shape[i][1], c[i])
# print(predicts)
# save_array_to_mat(predicts, "mat/cnn_1d_PU.mat")
# max_prob = np.max(predicts, axis=1)
# print(max_prob[:3])
# mean_prob = np.mean(predicts, axis=1)
# print(mean_prob[:3])
# min_prob = np.min(predicts, axis=1)
# print(min_prob[:3])
# sum_prob = np.sum(predicts, axis=1)
# print(sum_prob[:3]).
# predicts = sio.loadmat("mat/cnn_1d_PU.mat")
# predicts = predicts["pre"]
# write_whole_image_classification_result(predicts, image_shape[i])
# write_whole_image_predicts_prob(predicts, image_shape[i])


# mat_path = [save_path + x for x in os.listdir(save_path)]
# print(mat_path)
# for j in mat_path:
#     pred = sio.loadmat(j)
#     pre = pred['pre']
#     print(pre.shape)
