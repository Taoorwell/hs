import tensorflow as tf
import time
from python_gdal import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# pwd = r"H:/GF/JL/"
pwd = r"D:/JL/"
# pwd = r"G:/GF/JL/"
# pwd = r"/content/drive/'My Drive'/Drive_Data"
file_path = pwd + r"images/GF2_4314_GS_2.dat"
vector = pwd + r"vector/train_samples_0411.shp"  # test shapefile

segments_path = pwd + r"images/WORKSPACE/"  # segments shapefiles

mat_images_path = pwd + r'images/mat/GF_2.mat'  # 7500*5000 size images whose form is .mat
mat_labels_path = pwd + r'images/mat/GF_2_LABEL_1.mat'  # train label
mat_labels_path_1 = pwd + r'images/mat/GF_2_LABEL_2.mat'  # test label
mat_region_path = pwd + r'images/mat/GF_2_REGION.mat'  # whole region
model_path = pwd + r"model/"
predicts_mat_path = 'D:/JL/model/cpu/mat'
m = [35, 45, 55, 65, 75, 85, 95, 105]
s = [20, 30, 40, 50, 60, 70, 80]
# m = [55, 85, 95]
c = 7

# segments = gpd.read_file(filename=pwd + r"images/WORKSPACE/New Workspace/GF2_4314_GS_3008053_predicts.shp")
# print(segments.columns)
# predicts, index = vectors_to_raster(vector_data_path=pwd +
#                                     r"images/WORKSPACE/New Workspace/GF2_4314_GS_3008053_predicts.shp",
#                                     raster_data_path=file_path, field='predicts')
# oa, kappa = get_test_segments(data_path=mat_images_path, test_data_path=mat_labels_path_1, predicts=predicts,
#                               norma_methods="min-max")
#
# plot_segments_predicts_prob(segment_path=pwd + r"images/WORKSPACE/New Workspace/GF2_4314_GS_3008053_predicts.shp",
#                             raster_data_path=file_path, field='predicts')

# predicts, index = vectors_to_raster(vector_data_path=pwd + r"images/WORKSPACE/GF2_4314_GS_3_predicts1.shp",
#                                     raster_data_path=file_path, field='predicts')
# oa, kappa = get_test_segments(data_path=mat_images_path, test_data_path=mat_labels_path_1,
#                               predicts=predicts, norma_methods='min-max')
# plot_segments_predicts_prob(segment_path=pwd + r"images/WORKSPACE/GF2_4314_GS_3_predicts1.shp",
#                             raster_data_path=file_path)
# M, T = [], []
# for k in m:
#     model = tf.keras.models.load_model(model_path+r"gpu/GF_MODEL/valid/CNN_{}".format(k))
#     model.summary()
#     for i in range(20, 90, 10):
#         t, _ = get_predicts_segments(segments_path=segments_path+r"GF2_4314_GS_{}0805.shp".format(i),
#                                      image_mat_path=mat_images_path,
#                                      norma_methods="min-max", m=k, model=model)
#         M.append(m)
#         T.append(t)
# df = pd.DataFrame({"M": M,
#                    "T": T})
# df.to_excel(r'D:/JL/model/cpu/cpu_pre.xlsx')


# bands_data = get_mat(mat_data_path=mat_region_path)
# print(get_samples_info(bands_data))
# label_pixels, index = vectors_to_raster(vector_data_path=segments_path_2, raster_data_path=file_path,
#                                         field="predicts")
# get_test_segments(data_path=mat_images_path, test_data_path=mat_labels_path_1,predicts=label_pixels)
# segments = gpd.read_file(filename=filename)
# fig, ax = plt.subplots(1, 1)
# segments.plot(column='predicts', ax=ax, legend=True)
# ax.set_xticks([])
# ax.set_yticks([])
# plt.show()
# model = tf.keras.models.load_model(model_path + "CNN_{}.h5".format(m))
# model = tf.keras.models.load_model(model_path + "MLP_1.h5")
#
# model.summary()
# write_region_predicts(model=model, image_data_path=mat_images_path, region_data_path=mat_region_path,
#                       bsize=10000, filename=pwd + r"model/MLP_1_PRE.mat")

# plot_region_image_classification_result_prob(predict_mat_path=pwd+r"model/CNN_35.mat")
#
# get_predicts_segments(segments_path=segments_path, image_mat_path=mat_images_path, raster_data_path=file_path,
#                       test_data_path=mat_labels_path_1, norma_methods='z-score', m=m, model=model,
#                       filename=model_path+"sg/GF2_SEGMENTATION_81_45.shp")


# print(get_samples_info(prob[index]))
# filename = pwd + r"model/CNN_{}".format(m)
# predicts = get_mat(mat_data_path=pwd + "model/MLP_PRE.mat")
# print(predicts.shape)
# get_test_segments(data_path=mat_images_path, test_data_path=mat_labels_path_1, predicts=predicts[:, :, 0])
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
#
# model1.save(pwd + r"model/2/MLP03.h5")

# #
# #
train_samples, train_labels = get_train_sample(data_path=mat_images_path, train_data_path=mat_labels_path,
                                               c=c, norma_methods="min-max", m=1)
print(train_samples.shape, train_labels.shape)
# # #
model1 = tf.keras.models.Sequential([tf.keras.layers.Dense(32, activation='relu', input_shape=(4,)),
                                     tf.keras.layers.Dropout(0.1),
                                     tf.keras.layers.Dense(16, activation='relu'),
                                     tf.keras.layers.Dropout(0.1),
                                     tf.keras.layers.Dense(7, activation='softmax')])
# #
model1.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
model1.summary()
t1 = time.clock()
model1.fit(train_samples, train_labels, batch_size=30, epochs=500)
t2 = time.clock()
print("Training Time Consuming: {}".format(t2-t1))

bands_data, is_train, _ = get_mat_info(mat_data_path=mat_images_path, train_mat_data_path=mat_region_path)
bands_data = norma_data(bands_data, norma_methods="min-max")
index = np.array(is_train).transpose((1, 0))
samples = []
t3 = time.clock()
for i in index:
    sample = bands_data[i[0], i[1]]
    samples.append(sample)
samples = np.stack(samples)
predicts = model1.predict(samples)
t4 = time.clock()
print("Predicting Time Consuming: {} ".format(t4-t3))

# model1 = tf.keras.models.load_model()
# write_region_predicts(model1, image_data_path=mat_images_path, region_data_path=mat_region_path,
#                       bsize=10000, filename=pwd+r"model/2/MLP_PRE", norma_methods="min-max")

# #
# model = tf.keras.models.load_model(pwd + 'model/MLP_1.h5')
# predicts = write_region_predicts(model, data_path=mat_images_path, train_data_path=mat_region_path,
#                                  bsize=10000, norma_methods='z-score', m=1)
# write_region_image_classification_result_probs(predicts, train_data_path=mat_region_path, shape=(7500, 5000, 2),
#                                                filename=pwd + "model/MLP_PRE.mat")
# plot_region_image_classification_result_prob(predict_mat_path=pwd + "model/MLP_PRE.mat")
# oa, kappa = get_test_predict(model, data_path=mat_images_path, test_data_path=mat_labels_path_1, bsize=10000,
                             # norma_methods="z-score")

# train_samples, train_labels = get_train_sample(data_path=mat_images_path, train_data_path=mat_labels_path,
#                                                c=c, norma_methods='z-score', m=m)
#
# print(train_samples.shape, train_labels.shape)
#
# OA = []
# K = []
# M = []
# T = []
# l = os.listdir(model_path + r"2/")
# print(l)
# for i in l:
#     model = tf.keras.models.load_model(model_path + r"2/" + i)
#     oa, kappa = get_test_predict(model, data_path=mat_images_path, test_data_path=mat_labels_path_1,
#                                  bsize=10000, norma_methods='min-max', m=int(i.split('_')[-2]))
#     M.append(i.split('_')[-1])
#     T.append(i.split('_')[-2])
#     OA.append(oa)
#     K.append(kappa)
# df = pd.DataFrame({"M": M, "T": T, "OA": OA, "K":K})
# df.to_excel(pwd)
# M, T = [], []
# # I = []
# # T = []
# for i in m:
#     train_samples, train_labels = get_train_sample(data_path=mat_images_path, train_data_path=mat_labels_path,
#                                                    c=c, norma_methods="min-max", m=i)
#     print("Training Samples Shape: {}".format(train_samples.shape))
#     model2 = tf.keras.models.Sequential([tf.keras.layers.Conv2D(8, (5, 5), padding='valid',
#                                                                 input_shape=(i, i, 4)),
#                                         tf.keras.layers.BatchNormalization(),
#                                         tf.keras.layers.Activation(activation='relu'),
#                                         tf.keras.layers.MaxPool2D(2, padding='same'),
#                                         tf.keras.layers.Conv2D(12, (3, 3), padding='valid'),
#                                         tf.keras.layers.BatchNormalization(),
#                                         tf.keras.layers.Activation(activation='relu'),
#                                         tf.keras.layers.MaxPool2D(2, padding='same'),
#                                         tf.keras.layers.Conv2D(12, (3, 3), padding='valid'),
#                                         tf.keras.layers.BatchNormalization(),
#                                         tf.keras.layers.Activation(activation='relu'),
#                                         tf.keras.layers.MaxPool2D(2, padding='same'),
#                                         tf.keras.layers.Conv2D(12, (3, 3), padding='valid'),
#                                         tf.keras.layers.BatchNormalization(),
#                                         tf.keras.layers.Activation(activation='relu'),
#                                         tf.keras.layers.MaxPool2D(2, padding='same'),
#                                         tf.keras.layers.Flatten(),
#                                         tf.keras.layers.Dense(24, activation='relu'),
#                                         tf.keras.layers.Dropout(0.1),
#                                         tf.keras.layers.Dense(c, activation='softmax')])
#     model2.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy',
#                    metrics=['accuracy'])
#     model2.summary()
#     t1 = time.clock()
#     model2.fit(train_samples, train_labels, batch_size=30, epochs=100)
#     t2 = time.clock()
#     t = t2 - t1
#     print("Train CNN-{} Finished, Time Consuming:{}".format(i, t))
#     M.append(i)
#     T.append(t)
#     del model2
# print("Finish Training and Saving Model")
# df = pd.DataFrame({"M": M,
#                    "T": T})
# df.to_excel(pwd + r"model/cpu/time.xlsx")
# bands_data, is_train, training_labels = get_prep_data(data_path=file_path, train_data_path=vector)
# print(bands_data.shape, len(training_labels))
# print(get_samples_info(training_labels))
# model2 = tf.keras.models.load_model(pwd + r"model/CNN_{}.h5".format(m))
# model2.summary()
# t1 = time.clock()
# segments = get_predicts_segments(segments_path=segments_path, image_mat_path=mat_images_path,
#                                  norma_methods="z-score", m=m, model=model2)
# t2 = time.clock()
# print("Predict time: {}".format(t2-t1))
# segments.to_file(filename=pwd + r"model/sg/GF2_SEGMENTATION.shp")


# predicts, is_train = vectors_to_raster(vector_data_path=pwd + r"model/sg/GF2_SEGMENTATION.shp",
#                                        raster_data_path=file_path,
#                                        field='predicts')
# print(predicts.shape)
# plot_predicts(predicts)
# plt.show()
# oa, kappa = get_test_segments(data_path=mat_images_path, test_data_path=mat_labels_path_1, predicts=predicts)


# print(t1)
# predicts = write_region_predicts(model2, data_path=mat_images_path, train_data_path=mat_region_path, bsize=50000,
#                                  norma_methods="z-score", m=m)
# t2 = time.clock()
# print(t2)
# print("Predict time: {}".format(t2-t1))
# write_region_image_classification_result_probs(predicts, train_data_path=mat_region_path, shape=(7500, 5000, 2),
#                                                filename=filename)
#
# plot_region_image_classification_result_prob(predict_mat_path=filename)

# oa, kappa = get_test_predict(model=model2,
#                              data_path=mat_images_path,
#                              test_data_path=mat_labels_path_1,
#                              bsize=10000, norma_methods='z-score', m=m)
#
# # SAVE MODEL!!!!!!!!!!!!!!!
# model2.save(r"D:/JL/model/CNN_{}.h5".format(m))
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
# L = os.listdir(pwd + r"model/gpu/GF_MODEL/valid/")
#
# M, S, N, T, OA, KAPPA = [], [], [], [], [], []
# for l in L:
#     for s in range(20, 90, 10):
#         print(l, s)
#         model = tf.keras.models.load_model(pwd + r"model/gpu/GF_MODEL/valid/" + l)
#         m = int(l.split("_")[-2])
#         segments = gpd.read_file(segments_path)
#         t1 = time.clock()
#         x = segments.centroid.x
#         y = segments.centroid.y
#         segments['R'] = [int(a) for a in (2957730.452 - y)]
#         segments["C"] = [int(b) for b in (x - 541546.573)]
#         l = len(segments['R'])
#         bands_data = get_mat(mat_data_path=mat_images_path)
#         bands_data = norma_data(bands_data, norma_methods='min-max')
#         n = int((m - 1) / 2)
#         samples = []
#         pres = []
#         q = l // 20000
#         p = 0
#         for x, y in tqdm(zip(segments['R'], segments['C'])):
#             k1 = x - n
#             k2 = x + n + 1
#             k3 = y - n
#             k4 = y + n + 1
#             block = bands_data[k1:k2, k3:k4]
#             samples.append(block)
#             if len(samples) == 20000 or (len(samples) + p * 20000 == l):
#                 print("    Starting Predicts Segments")
#                 pre = model.predict(np.stack(samples))
#                 pres.append(pre)
#                 samples = []
#                 p = p + 1
#         press = np.concatenate(pres)
#         t2 = time.clock()
#         t = t2 - t1
#         M.append(m)
#         S.append(s)
#         T.append(t)
#         del pres, press, samples, segments
#         oa, kappa, t, number = get_predicts_segments(segments_path=pwd +
#                                                      r"images/WORKSPACE/New Workspace/GF2_4314_GS_{}0805.shp".format(s),
#                                                      image_mat_path=mat_images_path, raster_data_path=file_path,
#                                                      test_data_path=mat_labels_path_1, norma_methods='min-max',
#                                                      m=int(l.split('_')[-2]),
#                                                      model=model,
#                                                      filename=pwd +
#                                                      r"images/WORKSPACE/New Workspace/GF2_4314_GS_{}_{}0805_result.shp"
#                                                      .format(int(l.split("_")[-2]), s))
#         M.append(int(l.split('_')[-2]))
#         S.append(s)
#         N.append(number)
#         T.append(t)
#         OA.append(oa)
#         KAPPA.append(kappa)
# df = pd.DataFrame({"M": M, "S": S, "N": N, "T": T, "OA": OA, "KAPPA": KAPPA})
# df.to_excel(pwd + r"model/gpu/2.xlsx")

# model.summary()
# get_predicts_segments(segments_path=segments_path, image_mat_path=mat_images_path, raster_data_path=file_path,
#                       test_data_path=mat_labels_path_1, norma_methods="min-max", m=45, model=model,
#                       filename=pwd + r"images/WORKSPACE/New Workspace/GF2_4314_GS_200805_results.shp")
# oa, kappa = get_test_predict(model=model, data_path=mat_images_path, test_data_path=mat_labels_path_1,
#                              bsize=10000, norma_methods="min-max", m=105)
# M, T = [], []
# for l in L:
#     print(l)
#     model = tf.keras.models.load_model(pwd + r"model/gpu/GF_MODEL/valid/" + l)
#     m = int(l.split("_")[-2])
#     bsize = (3*1024*1024*1024)//(m*m*4*4)
#     t = write_region_predicts(model, image_data_path=mat_images_path, region_data_path=mat_region_path,
#                               bsize=bsize, filename=predicts_mat_path+"CNN_{}_REGION_PRE.mat".format(m),
#                               norma_methods='nin-max', m=m)
#     M.append(m)
#     T.append(t)
# df1 = pd.DataFrame({"M": M, "T": T})
# df1.to_excel(pwd + r'D:/JL/model/cpu/cpu_region_predicts.xlsx')
# for k in m:
#     train_samples, train_labels = get_train_sample(data_path=mat_images_path, train_data_path=mat_labels_path,
#                                                    c=c, norma_methods="min-max", m=k)
#     print("Train Samples Siz：{}".format(train_samples.shape))
#
