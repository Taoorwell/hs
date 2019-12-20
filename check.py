import tensorflow as tf
from python_gdal import *


# pwd = r"H:/GF/JL/"
pwd = r"D:/JL/"
# pwd = r"G:/GF/JL/"
# pwd = r"/content/drive/'My Drive'/Drive_Data"
file_path = pwd + r"images/GF2_4314_GS_2.dat"
vector_path = pwd + r"vector/new_shp"
vector = pwd + r"vector/test_samples_0411.shp"

segments_path = pwd + r"images/WORKSPACE/GF2_4314_GS_3.shp"
# centroid_path = r"G:/GF/JL/sg/40_10_05_centroid_new.shp"

mat_images_path = pwd + r'images/mat/GF_2.mat'

mat_labels_path = pwd + r'images/mat/GF_2_LABEL_1.mat'
mat_labels_path_1 = pwd + r'images/mat/GF_2_LABEL_2.mat'
mat_region_path = pwd + r'images/mat/GF_2_REGION.mat'
model_path = pwd + r"model/"


# train_samples, train_labels = get_train_sample(data_path=mat_images_path, train_data_path=mat_labels_path,
#                                                c=7, norma_methods="z-score", m=1)
# print(train_samples.shape, train_labels.shape)
# # bands_data, is_train, training_labels = get_prep_data(data_path=mat_images_path,
# #                                                       train_data_path=mat_labels_path,
# #                                                       norma_method='z-score')
# # print(bands_data.shape, is_train, len(training_labels))
#
# # rows, cols, n_bands, bands_data, geo_transform, proj = get_raster_info(raster_data_path=file_path)
# # labeled_pixels, is_train = vectors_to_raster(vector_data_path=vector, cols=cols, rows=rows,
# #                                              geo_transform=geo_transform,
# #                                              projection=proj)
# # save_array_to_mat(labeled_pixels, filename=pwd + 'images/mat/GF_2_LABEL_2')
# # print(labeled_pixels.shape)
#
# # print(bands_data.shape)
# # bands_data, is_train, training_labels = get_prep_data(data_path=file_path, train_data_path=vector,
# #                                                       norma_method="z-score")
# # print(len(is_train), training_labels.shape)
# # bands_data, is_train, training_labels = get_mat_info(mat_data_path=mat_images_path,
# #                                                      train_mat_data_path=mat_labels_path)
# # print(get_samples_info(training_labels))
# # model = tf.keras.models.load_model(model_path + 'MLP.h5')
# # model.summary()
#
# # {1.0: 10445, 2.0: 10510, 3.0: 10867, 4.0: 7974, 5.0: 9886, 6.0: 7411, 7.0: 10581, 8.0: 10131}
# # (541546.573, 1.0, -0.0, 2957730.452, -0.0, -1.0)
# # lists = [400, 400, 400, 400, 400, 400, 400, 400]
# #
# # model_list = ["CNN_33.h5", "CNN_65.h5", "CNN_49.h5"]
# # m = 33
# # samples = gpd.read_file(pwd+"vector/test_samples_0711.shp")
# # classes = np.unique(samples['CLASS_ID'])
# # single = samples[samples['CLASS_ID'] == classes[0]]
# # print(r'{}.shp'.format(classes[0]))
# # print(single)
# # split_vector(vector_path=pwd+"vector/test_samples_0711.shp", save_path=pwd+"vector/new_shp/")
# # rows, cols, n_bands, bands_data, geo_transform, proj = get_raster_info(
# #     raster_data_path=pwd + "images/GF2_4314_GS_2.dat")
# # labeled_pixels, is_train = vectors_to_raster(vector_data_path=pwd+"images/WORKSPACE/GF2_4314_GS_3_PRE_65.shp",
# #                                              cols=cols, rows=rows, geo_transform=geo_transform,
# #                                              projection=proj)
# # plot_predicts(labeled_pixels)
# # plt.show()
# # # train_samples = labeled_pixels[is_train]
# # # print(train_samples)
# # print(labeled_pixels.shape)
# # print(get_samples_info(labeled_piename=pwd+r"images/mat/GF_2_LABEL.mat")
# #
# # # print(labeled_pixels.shape)
# # # print(xels[is_train]))
# # save_array_to_mat(labeled_pixels, filget_samples_info(labeled_pixels[is_train]))
# # # split train data and test data for
# ##################################################
# # train_vector = gpd.read_file(pwd + "vector/train_samples_0411.shp")
# # print(train_vector)
# # id = [int(x) for x in train_vector['CLASS_ID']]
# # print(id)
# # train_vector['id'] = id
# # print(train_vector.columns)
# # train_vector.to_file(filename=pwd + "train_samples_0411_id.shp")
#
# # train_vector_1 = train_vector[train_vector['CLASS_ID'] == '1']
# # print(train_vector_1)
#
# # rows, cols, n_bands, bands_data, geo_transform, proj = get_raster_info(
# #     raster_data_path=pwd + "images/GF2_4314_GS_2.dat")
# # layer = data_source.GetLayer(0)
#
# # model2 = tf.keras.models.load_model(model_path+model_list[0])
# # model2.summary()
# # print("MODEL LOADING SUCCESS!!!")
# #
# # segments = get_predicts_segments(segments_path=segments_path, image_mat_path=mat_images_path,
# #                                  norma_methods='min-max', m=33, model=model2)
# # segments.to_file(filename=pwd + r"images/WORKSPACE/GF2_4314_GS_3_PRE_33.shp")
# #
# # print("FILE EXPORT SUCCESS! CHECK")
# ######################################################
# # results = get_mat(model_path + 'MLP_pre.mat')
# # print(np.unique(results[:, :, 1]))
# # probs = np.max(results, axis=-1)
# # sn.heatmap(probs, annot=False, cmap="Greys_r", xticklabels=False, yticklabels=False)
# # plt.show()
# # plt.subplots(121)
#
# # plt.subplot(122)
# # train_samples, train_labels = get_train_sample(data_path=mat_images_path, train_data_path=mat_labels_path,
# #                                                c=8, lists=lists, seed=20, norma_methods='z-score')
# # print(train_samples.shape, train_labels.shape, train_samples[:10])
# # model1 = tf.keras.models.load_model(model_path + 'MLP.h5')
# # model2 = tf.keras.models.load_model(model_path + 'CNN_33.h5')
# # get_test_predict(model=model2, data_path=mat_images_path, train_data_path=mat_labels_path, seed=10,
# #                  c=8, lists=lists, bsize=10000, norma_methods='min-max', m=33)
# # for i in model_list:
# #     print("Loading Model:" + i)
# #     model = tf.keras.models.load_model(model_path + i)
# #     print("Model Loading Success, Model Inference...")
# #     if len(model.input.shape) == 2:
# #         predicts = write_region_predicts(model, data_path=mat_images_path,
# #                                          train_data_path=mat_region_path, bsize=10000,
# #                                          norma_methods='min-max')
# #     else:
# #         predicts = write_region_predicts(model, data_path=mat_images_path,
# #                                          train_data_path=mat_region_path, bsize=10000,
# #                                          norma_methods='min-max', m=int(i.split('_')[-1].split(".")[0]))
# #     print("Model Inference Finished, Result output.....")
# #     write_region_image_classification_result_probs(predicts, train_data_path=mat_region_path,
# #                                                    shape=(7500, 5000, 2),
# #                                                    filename=model_path+i.split(".")[0]+'_pre.mat')
#
# # split_vector(vector_path=vector, save_path=pwd + 'vector/new_shp/')
# # segments = get_centroid_index(segments_path=segments_path)
#
# # plot_region_image_classification_result_prob(predict_mat_path=model_path+'CNN_65_pre.mat')
# # print(segments['R'])
# # rows, cols, n_bands, bands_data, geo_transform, proj = get_raster_info(raster_data_path=file_path)
# # bands_data = norma_data(bands_data, norma_methods='z-score')
# # bands_data_dict = sio.loadmat(mat_images_path)
# # bands_data = bands_data_dict[list(bands_data_dict.keys())[-1]]
# # bands_data = norma_data(bands_data, norma_methods='min-max')
#
# # model2 = tf.keras.models.load_model(r'G:/GF/JL/model/CNN_33.h5')
# # model2.summary()
# # n = 16
#
# # samples = []
# # for x, y in tqdm(zip(segments['R'], segments['C'])):
# #    print(x, y)
# #    k1 = x - n
# #    k2 = x + n + 1
# #    k3 = y - n
# #    k4 = y + n + 1
# #    block = bands_data[k1:k2, k3:k4]
# #    samples.append(block)
# # pre = model2.predict(np.stack(samples))
# # predicts = np.argmax(pre, axis=-1) + 1
#
# # labeled_pixels, is_train = vectors_to_raster(vector_path=r"G:/GF/JL/vector/new_shp_1", rows=rows, cols=cols,
# #                                             geo_transform=geo_transform,
# #                                             projection=proj)
#
# # train_samples, train_labels = get_train_sample(data_path=mat_images_path, train_data_path=mat_labels_path, c=8,
# #                                                lists=lists, d=2, norma_methods='z-score')
#
# # rows, cols, n_bands, bands_data, geo_transform, proj = get_raster_info(file_path)
# # labeled_pixels, is_train = vectors_to_raster(vector_path=vector_path, rows=rows, cols=cols,
# #                                              geo_transform=geo_transform,
# #                                              projection=proj)
#
#
# # bands_data, is_train, train_labels = get_mat_info(mat_data_path=mat_images_path,
# #                                                   train_mat_data_path=mat_labels_path)
#
# # print(bands_data.shape, len(train_labels))
# # print(get_samples_info(train_labels))
# # print(rows, cols, n_bands, bands_data, geo_transform, proj)
# # bands_data, is_train, training_labels = get_prep_data(data_path=file_path,
# #                                                       train_data_path=vector_path,
# #                                                       norma_method='min-max')
#
# ###########################################################
# # OCNN
#
# # model = load_model(r"E:/temp/cnn_33_model.h5")
# #
# # rows, cols, n_bands, bands_data, geo_transform, proj = get_raster_info(raster_data_path=file_path)
# # bands_data = norma_data(bands_data,norma_methods="min-max")
#
# ###########################################################################
#
# ###########################################################################
#
# # train_samples, train_labels = get_train_sample(data_path=mat_images_path,
# #                                                train_data_path=mat_labels_path,
# #                                                c=7, lists=lists, seed=10,
# #                                                norma_methods='min-max', m=m)
# #
# # print(train_samples.shape, train_labels.shape)
#
# model1 = tf.keras.models.Sequential([tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
#                                     tf.keras.layers.Dropout(0.1),
#                                     tf.keras.layers.Dense(32, activation='relu'),
#                                     tf.keras.layers.Dropout(0.1),
#                                     tf.keras.layers.Dense(7, activation='softmax')])
#
# model1.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# model1.summary()
# model1.fit(train_samples, train_labels, batch_size=30, epochs=1000)
# model1.save(pwd + r"model/MLP_1.h5")
# #
# model = tf.keras.models.load_model(pwd + 'model/MLP_1.h5')
# predicts = write_region_predicts(model, data_path=mat_images_path, train_data_path=mat_region_path,
#                                  bsize=10000, norma_methods='z-score', m=1)
# write_region_image_classification_result_probs(predicts, train_data_path=mat_region_path, shape=(7500, 5000, 2),
#                                                filename=pwd + "model/MLP_PRE.mat")
plot_region_image_classification_result_prob(predict_mat_path=pwd + "model/MLP_PRE.mat")
# oa, kappa = get_test_predict(model1, data_path=mat_images_path, test_data_path=mat_labels_path_1, bsize=10000,
#                              norma_methods="z-score")

# train_samples, train_labels = get_train_sample(data_path=mat_images_path, train_data_path=mat_labels_path,
#                                                 c=8, lists=lists, d=4, norma_methods='min-max', m=49)
# #
# train_labels = one_hot_encode(c=8, labels=train_labels)
# print(train_samples.shape, train_labels.shape)
# # #
# model2 = tf.keras.models.Sequential([tf.keras.layers.Conv2D(12, (3, 3), padding='same', input_shape=(m, m, 4)),
#                                      tf.keras.layers.BatchNormalization(),
#                                      tf.keras.layers.Activation(activation='relu'),
#                                      tf.keras.layers.MaxPool2D(2, padding='same'),
#                                      tf.keras.layers.Conv2D(24, (3, 3), padding='same'),
#                                      tf.keras.layers.BatchNormalization(),
#                                      tf.keras.layers.Activation(activation='relu'),
#                                      tf.keras.layers.MaxPool2D(2, padding='same'),
#                                      tf.keras.layers.Conv2D(48, (3, 3), padding='same'),
#                                      tf.keras.layers.BatchNormalization(),
#                                      tf.keras.layers.Activation(activation='relu'),
#                                      tf.keras.layers.MaxPool2D(2, padding='same'),
#                                      tf.keras.layers.Flatten(),
#                                      tf.keras.layers.Dense(32, activation='relu'),
#                                      tf.keras.layers.Dropout(0.1),
#                                      tf.keras.layers.Dense(7, activation='softmax')])
# model2.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
# # model2.summary()
# # #
# model2.fit(train_samples, train_labels, batch_size=30, epochs=100)
# model2 = tf.keras.models.load_model(r"D:/JL/model/CNN_33.h5")
#
# oa, kappa = get_test_predict(model=model2,
#                              data_path=mat_images_path,
#                              test_data_path=mat_labels_path,
#                              bsize=10000, norma_methods='min-max', m=m)
# #
# # # SAVE MODEL!!!!!!!!!!!!!!!
# model2.save(r"D:/JL/model/CNN_33.h5")
# print("Model Saved!!!")
################################################################################
# MLP
# model1 = tf.keras.models.load_model(pwd + r"model/MLP.h5")
# model1.summary()

# model2 = tf.keras.models.load_model(pwd + r'model/CNN_33.h5')
# model2.summary()

# region_bands_data = get_mat(pwd + r"images/mat/GF_2_REGION.mat")
# is_train = np.nonzero(region_bands_data)

# print(region_bands_data.shape)
# predicts = write_train_region_predicts(model=model2, data_path=mat_images_path,
#                                       train_data_path=mat_region_path, bsize=6400,
#                                       norma_methods='z-score', m=33)

# write_region_image_classification_result(predicts, train_data_path=mat_region_path, shape=(7500, 5000))
# _, _, _, bands_data, _, _ = get_raster_info(raster_data_path=file_path)
#
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
