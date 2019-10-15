import tensorflow as tf
from python_gdal import *
# from models_keras import *
import numpy as np
import pandas as pd
import shapefile as shp
# from shapfile_plot import *
file_path = r"G:/GF/JL/images/GF2_4314_GS_2.dat"
vector_path = r"G:/GF/JL/vector/shp"
mat_images_path = r'G:/GF/JL/images/mat/GF_2.mat'
mat_labels_path = r'G:/GF/JL/images/mat/GF_2_LABEL.mat'
mat_region_path = r'G:/GF/JL/images/mat/GF_2_REGION.mat'
# {1.0: 10445, 2.0: 10510, 3.0: 10867, 4.0: 7974, 5.0: 9886, 6.0: 7411, 7.0: 10581, 8.0: 10131}
lists = [1445, 1510, 1867, 1974, 1886, 1411, 1581, 1131]
train_samples, train_labels = get_train_sample(data_path=mat_images_path, train_data_path=mat_labels_path,
                                               c=8, lists=lists, d=4, norma_methods='z-score', m=33)
# print(train_samples.shape, len(train_labels))
# rows, cols, n_bands, bands_data, geo_transform, proj = get_raster_info(file_path)
# labeled_pixels, is_train = vectors_to_raster(vector_path=vector_path, rows=rows, cols=cols,
#                                              geo_transform=geo_transform,
#                                              projection=proj)
# save_array_to_mat(bands_data, filename=r'G:/GF/JL/images/mat/GF_2.mat')
# save_array_to_mat(labeled_pixels, filename=r'G:/GF/JL/images/mat/GF_2_LABEL.mat')
# bands_data, is_train, train_labels = get_mat_info(mat_data_path=mat_images_path,
#                                                   train_mat_data_path=mat_labels_path)
# test_region = vectors_to_raster1(vector_path=r"G:/GF/JL/vector/test_boundary_polygon.shp",
#                                  rows=rows, cols=cols, geo_transform=geo_transform, projection=proj)
# print(test_region.shape)
# print(get_samples_info(test_region))
# save_array_to_mat(test_region, filename='G:/GF/JL/images/mat/GF_2_REGION.mat')
# print(bands_data.shape, len(train_labels))
# print(get_samples_info(train_labels))
# print(rows, cols, n_bands, bands_data, geo_transform, proj)
# bands_data, is_train, training_labels = get_prep_data(data_path=file_path,
#                                                       train_data_path=vector_path,
#                                                       norma_method='min-max')
# print("Image size:{}, Number of samples:{}".format(bands_data.shape, len(training_labels)))
# print(is_train[0], is_train[1])
# samples = get_samples_info(training_labels)
# print(samples)

# print(training_labels)
# print(is_train[0].shape)
# plt.imshow(bands_data[:, :, 3])
# plt.show()

# centroid_path = r"E:/temp/centroid_position.txt"
# polygone_100 = r"E:/temp/pci_seg/100.shp"
# lists = [10, 30, 20, 30, 30, 30, 30]
#
# sf = shp.Reader(polygone_100)
#
# df = read_shapefile(sf)
# print(df.head())
#
# print(sf.shape())
# print("The amount of shape files: {}".format(len(sf)))
# print("The shapefile fields: {} ".format(sf.fields))
# print("The xth shape attribution: {}".format(sf.records()[0]))
# plot_map(sf)
# plt.show()
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
train_labels = one_hot_encode(c=8, labels=train_labels)
# #
print(train_samples.shape, train_labels.shape)

# model1 = tf.keras.models.Sequential([tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
#                                     tf.keras.layers.Dropout(0.1),
#                                     tf.keras.layers.Dense(32, activation='relu'),
#                                     tf.keras.layers.Dropout(0.1),
#                                     tf.keras.layers.Dense(8, activation='softmax')])
#
# model1.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
# model1.summary()
# model1.fit(train_samples, train_labels, batch_size=30, epochs=100)


model2 = tf.keras.models.Sequential([tf.keras.layers.Conv2D(12, (3, 3), padding='same', input_shape=(33, 33, 4)),
                                     tf.keras.layers.BatchNormalization(),
                                     tf.keras.layers.Activation(activation='relu'),
                                     tf.keras.layers.MaxPool2D(2, padding='same'),
                                     tf.keras.layers.Conv2D(24, (3, 3), padding='same'),
                                     tf.keras.layers.BatchNormalization(),
                                     tf.keras.layers.Activation(activation='relu'),
                                     tf.keras.layers.MaxPool2D(2, padding='same'),
                                     tf.keras.layers.Conv2D(48, (3, 3), padding='same'),
                                     tf.keras.layers.BatchNormalization(),
                                     tf.keras.layers.Activation(activation='relu'),
                                     tf.keras.layers.MaxPool2D(2, padding='same'),
                                     tf.keras.layers.Flatten(),
                                     tf.keras.layers.Dense(32, activation='relu'),
                                     tf.keras.layers.Dropout(0.1),
                                     tf.keras.layers.Dense(8, activation='softmax')])

model2.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

model2.summary()
model2.fit(train_samples, train_labels, batch_size=30, epochs=100)

# # SAVE MODEL!!!!!!!!!!!!!!!
model2.save(r"G:/GF/JL/model/CNN_33.h5")
print("Model Saved!!!")
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
