from python_gdal import *
from models_keras import *
file_path = r"E:/temp/GF2T.dat"
vector_path = r"E:/temp/samples/"
lists = [10, 30, 20, 30, 30, 30, 30]
# samples = get_samples_info(labels_samples=train_labels)
# {1.0: 15, 2.0: 40, 3.0: 25, 4.0: 39, 5.0: 35, 6.0: 36, 7.0: 36} overall: 226

# train_samples, train_labels = get_train_sample(data_path=file_path, train_data_path=vector_path,
#                                                c=7, lists=lists, d=2, norma_methods="min-max")
#
#
# train_labels = one_hot_encode(c=7, labels=train_labels)
#
# print(train_samples.shape, train_labels.shape)

# model = mlp(input_shape=(4,), c=7, lr=0.01, rate1=0.1, rate2=0.1, l=0.00001)
# model.summary()

# model.fit(train_samples, train_labels, batch_size=10, epochs=5000)
# model.save(r"E:/temp/mlp_model.h5")
model = load_model(r"E:/temp/mlp_model.h5")
# model.summary()

# OA, KAPPA = get_test_predict(model=model, data_path=file_path, train_data_path=vector_path,
#                              c=7, lists=lists, bsize=10, norma_methods="min-max")
# print(OA, KAPPA)
# 0.8478, 0.8199

_, _, _, bands_data, _, _ = get_raster_info(raster_data_path=file_path)

bands_data = norma_data(bands_data, norma_methods="min-max")

pre = bands_data.reshape((bands_data.shape[0]*bands_data.shape[1], bands_data.shape[2]))

predicts = model.predict(pre)

write_whole_image_classification_result(predicts, shape=(801, 801))
