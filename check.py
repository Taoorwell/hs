from python_gdal import *
# from models_keras import *
import numpy as np
import pandas as pd
import shapefile as shp
from shapfile_plot import *
file_path = r"E:/temp/GF2T.dat"
vector_path = r"E:/temp/samples/"
centroid_path = r"E:/temp/centroid_position.txt"
polygone_100 = r"E:/temp/pci_seg/100.shp"
lists = [10, 30, 20, 30, 30, 30, 30]

sf = shp.Reader(polygone_100)

df = read_shapefile(sf)
print(df.head())

print(sf.shape())
print("The amount of shape files: {}".format(len(sf)))
print("The shapefile fields: {} ".format(sf.fields))
print("The xth shape attribution: {}".format(sf.records()[0]))

plot_map(sf)
plt.show()
###########################################################
# OCNN
# centroid = pd.read_csv(centroid_path)
# R = centroid["R"]
# C = centroid["C"]
# # print(R, C)
# model = load_model(r"E:/temp/cnn_33_model.h5")
#
# rows, cols, n_bands, bands_data, geo_transform, proj = get_raster_info(raster_data_path=file_path)
# bands_data = norma_data(bands_data,norma_methods="min-max")
# centroid_index = []
# n = 16
# for i, j in zip(R, C):
#     index = (i+16, j+16)
#     centroid_index.append(index)
# print(centroid_index)
# samples = []
# bands_data = np.pad(bands_data, ((n, n), (n, n), (0, 0)), 'constant', constant_values=0)
# print(bands_data.shape)
# for i, j in enumerate(centroid_index):
#     k1 = j[0] - n
#     k2 = j[0] + n +1
#     k3 = j[1] - n
#     k4 = j[1] + n + 1
#     block = bands_data[k1:k2, k3:k4]
#     samples.append(block)
# pre = np.stack(samples)
# predict = model.predict(pre)
# predicts = np.argmax(predict, axis=-1) + 1
# centroid["pr"] = predicts
# centroid.to_csv(r"E:/temp/pre.csv")
# print(centroid)
###########################################################################

###########################################################################
# samples = get_samples_info(labels_samples=train_labels)
# {1.0: 15, 2.0: 40, 3.0: 25, 4.0: 39, 5.0: 35, 6.0: 36, 7.0: 36} overall: 226

# train_samples, train_labels = get_train_sample(data_path=file_path, train_data_path=vector_path,
#                                                c=7, lists=lists, d=4, norma_methods="min-max",
#                                                m=33)
#
#
# #
# train_labels = one_hot_encode(c=7, labels=train_labels)
# #
# print(train_samples.shape, train_labels.shape)

# model = mlp(input_shape=(4,), c=7, lr=0.01, rate1=0.1, rate2=0.1, l=0.00001)
# model.summary()

# model = cnn_2d(input_shape=(33, 33, 4), c=7, lr=0.001)
# model.summary()
# model.fit(train_samples, train_labels, batch_size=10, epochs=100)
# model.save(r"E:/temp/cnn_33_model.h5")
################################################################################
# MLP
# model = load_model(r"E:/temp/mlp_model.h5")
# model.summary()
# _, _, _, bands_data, _, _ = get_raster_info(raster_data_path=file_path)
# #
# bands_data = norma_data(bands_data, norma_methods="min-max")
# #
# pre = bands_data.reshape((bands_data.shape[0]*bands_data.shape[1], bands_data.shape[2]))
# #
# predicts = model.predict(pre)
#
# save_array_to_mat(predicts, filename=r"E:/temp/mlp.mat")
# write_whole_image_classification_result(predicts, shape=(801, 801))
###################################################################################
# CNN -32
# model = load_model(r"E:/temp/cnn_33_model.h5")
# model.summary()

# OA, KAPPA = get_test_predict(model=model, data_path=file_path, train_data_path=vector_path,
#                              c=7, lists=lists, bsize=100, norma_methods="min-max", m=33)
# print(OA, KAPPA)
# MLP 0.8478, 0.8199 32 32
# CNN_33 0.9347 0.9226 32 64 64

# predicts = write_out_whole_predicts(model=model, data_path=file_path, bsize=32000, norma_methods="min-max",
#                                     m=33)

# save_array_to_mat(predicts, filename=r"E:/temp/CNN_33.mat")
# write_whole_image_classification_result(predicts, shape=(801, 801))
