#!/usr/bin/env python
# coding: utf-8
# # This .py file finish the purpose on reading vector or .mat data format
# which labeling by hand for extracting train raster pixels
# # and pixels labeling accordingly, is for preparation before feeding data into models.
# # And then, load needed classification files and get classification results!


# # Load packages
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import os
from osgeo import gdal
import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, f1_score, cohen_kappa_score, accuracy_score
from imblearn.over_sampling import RandomOverSampler, BorderlineSMOTE, SMOTE
from models_keras import *
# Functions of Gdal
# get raster data info, included rows, cols, n_bands, bands_data(read by band and shape is (W,H,C)),
# projection and geo transformation.


def get_raster_info(raster_data_path):
    raster_dataset = gdal.Open(raster_data_path, gdal.GA_ReadOnly)
    geo_transform = raster_dataset.GetGeoTransform()
    proj = raster_dataset.GetProjectionRef()
    bands_data = []
    for b in range(1, raster_dataset.RasterCount + 1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())
    bands_data = np.dstack(bands_data)
    bands_data = bands_data[:, :, :]
    rows, cols, n_bands = bands_data.shape
    return rows, cols, n_bands, bands_data, geo_transform, proj


# read shapefiles of label, And rasterize layer with according label values.used together with below func.
def create_mask_from_vector(vector_data_path, cols, rows, geo_transform,
                            projection, target_value=1):
    """ Rasterize the given vector(wrapper for gdal.RasterizeLayer)"""
    data_source = gdal.OpenEx(vector_data_path, gdal.OF_VECTOR)
    layer = data_source.GetLayer(0)
    driver = gdal.GetDriverByName('MEM')
    target_ds = driver.Create('', cols, rows, 1, gdal.GDT_UInt16)
    target_ds.SetGeoTransform(geo_transform)
    target_ds.SetProjection(projection)
    gdal.RasterizeLayer(target_ds, [1], layer, burn_values=[target_value])
    return target_ds


# return label_pixel including geo information and index.
def vectors_to_raster(vector_path, rows, cols, geo_transform, projection):
    """Rasterize the vectors in given directory in a single image."""
    files = [f for f in os.listdir(vector_path) if f.endswith('.shp')]
    classes = [f.split('.')[0] for f in files]
    shapefiles = [os.path.join(vector_path, f) for f in files]
    labeled_pixels = np.zeros((rows, cols))
    for i, path in zip(classes, shapefiles):
        label = int(i)
        ds = create_mask_from_vector(path, cols, rows, geo_transform,
                                     projection, target_value=label)
        band = ds.GetRasterBand(1)
        labeled_pixels += band.ReadAsArray()
        ds = None
    is_train = np.nonzero(labeled_pixels)
    return labeled_pixels, is_train


# # due to hyperspectral images datasets on web is .mat format, using scipy.sio read .mat data
# # return is dict and bands data and labels ndarray is last key value.
# # get is_train from labels ndarry and generate training labels and bands data and is_train.
def get_mat_info(mat_data_path, train_mat_data_path):
    bands_data_dict = sio.loadmat(mat_data_path)
    bands_data = bands_data_dict[list(bands_data_dict.keys())[-1]]
    labeled_pixel_dict = sio.loadmat(train_mat_data_path)
    labeled_pixel = labeled_pixel_dict[list(labeled_pixel_dict.keys())[-1]]
    is_train = np.nonzero(labeled_pixel)
    training_labels = labeled_pixel[is_train]
    return bands_data, is_train, training_labels


def save_array_to_mat(array, filename):
    dict = {"pre": array}
    sio.savemat(filename, dict)


# # pca data for decreasing dimensions, data is bands data after normalization.
def pca_data(data, n=3):
    x = data.reshape(-1, data.shape[-1])
    pca = PCA(n_components=n).fit(x)
    x_p = pca.transform(x)
    x_p = x_p.reshape(data.shape[0], data.shape[1], n)
    return x_p


# According to label_pixel's geo and index to obtain training samples accordingly.
# m is block size which for CNN input.
# # increase various data format inputs, images data: .tif and .mat, labels data: .shp and .mat
# when m == 1,Function return training_samples is shape of (numbers samples, bands) array,
# training_labels return is shape of (numbers samples,) array.
# when m == block_size, Function returns training_samples is array of shape is (numbers samples,m,m,bands)
# training_labels is same above!
def get_prep_data(data_path, train_data_path, pca=False, norma_method="z-score", n=3):
    if data_path.endswith('.pix'):
        rows, cols, n_bands, bands_data, geo_transform, proj = get_raster_info(data_path)
        try:
            labeled_pixels, is_train = vectors_to_raster(train_data_path, rows, cols, geo_transform, proj)
            training_labels = labeled_pixels[is_train]
        except NotADirectoryError:
            rows, cols, n_bands, band_data, geo_transform, proj = get_raster_info(train_data_path)
            band_data = band_data.reshape(rows, cols)
            is_train = np.nonzero(band_data)
            training_labels = band_data[is_train]
    else:
        bands_data, is_train, training_labels = get_mat_info(data_path, train_data_path)

    bands_data = norma_data(bands_data, norma_methods=norma_method)
    if pca is True:
        bands_data = pca_data(bands_data, n=n)
    return bands_data, is_train, training_labels


def custom_train_index(is_train, training_labels, c, lists):
    np.random.seed(10)
    index = np.array(is_train).transpose((1, 0))
    x_train_index, x_test_index, y_train, y_test = [], [], [], []
    for i, n in zip(range(1, c+1, 1), lists):
        i_index = [j for j, x in enumerate(training_labels) if x == i]
        i_index_random = np.random.choice(i_index, n, replace=False)
        i_index_rest = [k for k in i_index if k not in i_index_random]
        i_train_index = index[i_index_random]
        i_train_labels = np.ones(len(i_index_random))* i
        i_test_index = index[i_index_rest]
        i_test_labels = np.ones(len(i_test_index))* i
        x_train_index.append(i_train_index)
        x_test_index.append(i_test_index)
        y_train.append(i_train_labels)
        y_test.append(i_test_labels)
    x_train_index = np.concatenate(x_train_index, axis=0)
    x_test_index = np.concatenate(x_test_index, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    y_test = np.concatenate(y_test, axis=0)
    x_train_index, _, y_train, _ = train_test_split(x_train_index, y_train, test_size=0, shuffle=True)
    x_test_index, _, y_test, _ = train_test_split(x_test_index, y_test, test_size=0, shuffle=True)
    return x_train_index, x_test_index, y_train, y_test


def get_train_sample(data_path, train_data_path, c, lists, d, norma_methods='z-score', pca=False, m=1, n=3):
    bands_data, is_train, training_labels = get_prep_data(data_path, train_data_path,
                                                          norma_method=norma_methods,
                                                          pca=pca, n=n)
    x_train_index, _, train_labels, _ = custom_train_index(is_train, training_labels,
                                                           c=c, lists=lists)
    samples = []
    if m == 1:
        for i in x_train_index:
            sample = bands_data[i[0], i[1]]
            samples.append(sample)
        train_samples = np.stack(samples)
        if d == 3:
            train_samples = train_samples.reshape((train_samples.shape[0], train_samples.shape[1], -1))

    else:
        n = int((m - 1) / 2)
        x_train_nindex = x_train_index + n
        bands_data = np.pad(bands_data, ((n, n), (n, n), (0, 0)), 'constant', constant_values=0)
        for j in x_train_nindex:
            k1 = j[0] - n
            k2 = j[0] + n + 1
            k3 = j[1] - n
            k4 = j[1] + n + 1
            block = bands_data[k1:k2, k3:k4]
            samples.append(block)
        train_samples = np.stack(samples, axis=0)
        if d == 5:
            train_samples = train_samples.reshape((train_samples.shape[0], train_samples.shape[1],
                                                   train_samples.shape[2], train_samples.shape[3], -1))
    return train_samples, train_labels


def get_test_predict(model, data_path, train_data_path, c, lists, bsize, norma_methods='z-score', pca=False, m=1, n=3):
    bands_data, is_train, training_labels = get_prep_data(data_path, train_data_path,
                                                          norma_method=norma_methods,
                                                          pca=pca, n=n)
    _, x_test_index, _, y_test = custom_train_index(is_train, training_labels, c=c, lists=lists)
    samples = []
    predicts = []
    if m == 1:
        for i in x_test_index:
            sample = bands_data[i[0], i[1]]
            samples.append(sample)
        samples = np.stack(samples)
        if len(model.input.shape) == 3:
            samples = samples.reshape((samples.shape[0], samples.shape[1], -1))
        predicts = model.predict(samples)

    else:
        n = int((m - 1) / 2)
        x_test_nindex = x_test_index + n
        bands_data = np.pad(bands_data, ((n, n), (n, n), (0, 0)), 'constant', constant_values=0)
        for i, j in enumerate(x_test_nindex):
            k1 = j[0] - n
            k2 = j[0] + n + 1
            k3 = j[1] - n
            k4 = j[1] + n + 1
            block = bands_data[k1:k2, k3:k4]
            samples.append(block)
            if len(samples) == bsize or i == x_test_nindex.shape[0] - 1:
                # print("Batches Predictions...")
                pre = np.stack(samples)
                if len(model.input.shape) == 5:
                    pre = pre.reshape((pre.shape[0], pre.shape[1], pre.shape[2], pre.shape[3], -1))
                predict = model.predict(pre)
                predicts.append(predict)
                samples = []
        predicts = np.concatenate(predicts)
    print("Batches Predictions Finish!!!")
    OA, KAPPA = print_plot_cm(y_test, predicts)
    return OA, KAPPA
    # return predicts


def write_out_whole_predicts(model, data_path, bsize, norma_methods='z-score', pca=False, m=1, n=3):
    bands_data_dict = sio.loadmat(data_path)
    bands_data = bands_data_dict[list(bands_data_dict.keys())[-1]]
    bands_data = norma_data(bands_data, norma_methods=norma_methods)
    if pca is True:
        bands_data = pca_data(bands_data, n=n)
    if m == 1:
        if len(model.input.shape) == 2:
            pre = bands_data.reshape((bands_data.shape[0]*bands_data.shape[1], bands_data.shape[2]))
        else:
            pre = bands_data.reshape((bands_data.shape[0]*bands_data.shape[1], bands_data.shape[2], -1))
        predicts = model.predict(pre)
    else:
        n = int((m - 1) / 2)
        bands_data = np.pad(bands_data, ((n, n), (n, n), (0, 0)), 'constant', constant_values=0)
        cols = bands_data.shape[1]-2*n
        rows = bands_data.shape[0]-2*n
        result = []
        predicts = []
        for i in range(0, rows, 1):
            for j in range(0, cols, 1):
                data = bands_data[i: i + m, j: j + m, :]
                result.append(data)
                if len(result) == bsize or i == int(rows-1):
                    # print("Batches Predictions...")
                    pre = np.stack(result)
                    if len(model.input.shape) == 5:
                        pre = pre.reshape((pre.shape[0], pre.shape[1], pre.shape[2], pre.shape[3], -1))
                    predict = model.predict(pre)
                    predicts.append(predict)
                    result = []
        predicts = np.concatenate(predicts)
        print("Batches Predictions Finish!!!")
    return predicts
    # write_classification_result2(predicts, shape)
    # write_classification_prob(predicts, shape)


def write_train_region_predicts(model, data_path, train_data_path,
                                bsize, norma_methods='z-score', pca=False, m=1, n=3):
    bands_data, is_train, _ = get_mat_info(data_path, train_data_path)
    bands_data = norma_data(bands_data, norma_methods=norma_methods)
    if pca is True:
        bands_data = pca_data(bands_data, n=n)
    index = np.array(is_train).transpose((1, 0))
    samples = []
    if m == 1:
        for i in index:
            sample = bands_data[i[0], i[1]]
            samples.append(sample)
        samples = np.stack(samples)
        if len(model.input.shape) == 3:
            samples = samples.reshape((samples.shape[0], samples.shape[1], -1))
        predicts = model.predict(samples)
    else:
        predicts = []
        n = int((m - 1) / 2)
        index = index + n
        bands_data = np.pad(bands_data, ((n, n), (n, n), (0, 0)), 'constant', constant_values=0)
        for i, j in enumerate(index):
            k1 = j[0] - n
            k2 = j[0] + n + 1
            k3 = j[1] - n
            k4 = j[1] + n + 1
            block = bands_data[k1:k2, k3:k4]
            samples.append(block)
            if len(samples) == bsize or i == index.shape[0] - 1:
                # print("Batches Predictions...")
                pre = np.stack(samples)
                if len(model.input.shape) == 5:
                    pre = pre.reshape((pre.shape[0], pre.shape[1], pre.shape[2], pre.shape[3], -1))
                pre = model.predict(pre)
                predicts.append(pre)
                samples = []
        predicts = np.concatenate(predicts)
        print("Batches Predictions Finish!!!")
    return predicts
    # write_classification_result3(predicts, train_data_path, shape)


def get_fusion_features_from_test(model1, model2, data_path, train_data_path, c, lists, m):
    extractor_from_model1, extractor_from_model2 = feature_extractor(model1, model2)
    bands_data, is_train, train_labels = get_prep_data(data_path, train_data_path)
    _, test_index, _, y_test = custom_train_index(is_train, train_labels, c, lists)
    features1 = []
    for i in test_index:
        feature = bands_data[i[0], i[1]]
        features1.append(feature)
    features1 = np.stack(features1)
    features1 = features1.reshape((features1.shape[0], features1.shape[1], -1))
    features1 = extractor_from_model1([features1])[0]
    features2 = []
    samples = []
    n = int((m - 1) / 2)
    x_test_nindex = test_index + n
    bands_data = np.pad(bands_data, ((n, n), (n, n), (0, 0)), 'constant', constant_values=0)
    for i, j in enumerate(x_test_nindex):
        k1 = j[0] - n
        k2 = j[0] + n + 1
        k3 = j[1] - n
        k4 = j[1] + n + 1
        block = bands_data[k1:k2, k3:k4]
        samples.append(block)
        if len(samples) == 3200 or i == x_test_nindex.shape[0] - 1:
            print("Batches Features...")
            pre = np.stack(samples)
            feature = extractor_from_model2([pre])[0]
            features2.append(feature)
            samples = []
    features2 = np.concatenate(features2)
    fusion_features = np.concatenate([features1, features2], axis=1)
    return fusion_features, y_test


def get_fusion_features_from_whole(model1, model2, data_path, m):
    extractor_from_model1, extractor_from_model2 = feature_extractor(model1, model2)
    bands_data_dict = sio.loadmat(data_path)
    bands_data_1 = bands_data_dict[list(bands_data_dict.keys())[-1]]
    bands_data_1 = norma_data(bands_data_1)
    features1 = bands_data_1.reshape((bands_data_1.shape[0] * bands_data_1.shape[1], bands_data_1.shape[2], -1))
    f1 = extractor_from_model1([features1])[0]

    n = int((m - 1) / 2)
    bands_data_1 = np.pad(bands_data_1, ((n, n), (n, n), (0, 0)), 'constant', constant_values=0)
    cols = bands_data_1.shape[1] - 2 * n
    rows = bands_data_1.shape[0] - 2 * n
    result1 = []
    f2 = []
    for g in range(0, rows, 1):
        for h in range(0, cols, 1):
            data = bands_data_1[g: g + m, h: h + m, :]
            result1.append(data)
            if len(result1) == 1600 or g == int(rows - 1):
                print("Batches Features...")
                pre1 = np.stack(result1)
                fe = extractor_from_model2([pre1])[0]
                f2.append(fe)
                result1 = []
    f2 = np.concatenate(f2)

    f3 = np.concatenate([f1, f2], axis=1)
    return f3


def write_classification_result_tif(fname, classification, original_raster_data_path, m=1):
    """Create a GeoTIFF file with the given data."""
    rows, cols, n_bands, bands_data, geo_transform, proj = get_raster_info(raster_data_path=original_raster_data_path)
    driver = gdal.GetDriverByName('GTiff')
    classification = classification.reshape((rows, cols))
    dataset = driver.Create(fname, cols, rows, 1, gdal.GDT_Byte)
    dataset.SetGeoTransform(geo_transform)
    dataset.SetProjection(proj)
    band = dataset.GetRasterBand(1)
    band.WriteArray(classification)
    dataset = None  # close the file


# # write out whole image to RGB
# # predict value form model, original image shape
def write_whole_image_classification_result(predict, shape):
    if predict.ndim == 2:
        predict = np.argmax(predict, axis=-1) + 1
    arr_2d = np.reshape(predict, shape)
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    plt.imshow(arr_3d)
    plt.show()


def write_whole_image_predicts_prob(predict, shape):
    max_prob = np.max(predict, axis=1)
    mean_prob = np.mean(predict, axis=1)
    sec_max = []
    for p in predict:
        p = sorted(p)
        second = p[-2]
        sec_max.append(second)
    conf1 = max_prob - sec_max
    conf2 = max_prob - mean_prob
    low_confidence0 = [x for x in max_prob if x < 0.5]
    low_confidence1 = [x for x in conf1 if x < 0.5]
    low_confidence2 = [x for x in conf2 if x < 0.5]
    cof1 = len(low_confidence0)/(shape[0]*shape[1])
    cof2 = len(low_confidence1)/(shape[0]*shape[1])
    cof3 = len(low_confidence2)/(shape[0]*shape[1])
    # print("cof1:{}, cof2:{}, cof3: {}".format(cof1, cof2, cof3))
    # prob_img = np.reshape(conf, shape)
    # fig = plt.figure()
    # fig.add_subplot(121)
    # plt.xlabel("Confidences")
    # plt.ylabel("Numbers")
    # plt.hist(conf, bins=10, range=(0, 1), facecolor='red', alpha=0.5)
    #
    # fig.add_subplot(122)
    # sn.heatmap(prob_img, annot=False, cmap="Greys_r", xticklabels=False, yticklabels=False)
    # plt.imshow(prob_img, cmap='gray')
    #
    # plt.show()
    return cof1, cof2, cof3


# # write out labeled data given to RGB
# # parameter: predict value from model, train_mat_data_path for is_train, original images shape
def write_region_image_classification_result(predict, train_data_path, shape):
    labeled_pixel_dict = sio.loadmat(train_data_path)
    labeled_pixel = labeled_pixel_dict[list(labeled_pixel_dict.keys())[-1]]
    is_train = np.nonzero(labeled_pixel)
    if predict.ndim == 2:
        labels = np.argmax(predict, axis=-1) + 1
    else:
        labels = predict
    label = np.zeros(shape)
    for i, j, k in zip(is_train[0], is_train[1], labels):
        label[i, j] = k
    arr_2d = label
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    plt.imshow(arr_3d)
    plt.show()


def print_plot_cm(y_true, y_pred):
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=-1) + 1
    if y_true.ndim == 2:
        y_true = np.argmax(y_true, axis=-1) + 1
    OA = accuracy_score(y_true, y_pred)
    KAPPA = cohen_kappa_score(y_true, y_pred)
    # print("Overall Accuracy:{:.4%}".format(OA))
    # print("Kappa: ", KAPPA)
    # print(classification_report(y_true, y_pred, digits=4))
    # labels = sorted(list(set(y_true)))
    # cm_data = confusion_matrix(y_true, y_pred, labels=labels)
    # df_cm = pd.DataFrame(cm_data, index=labels, columns=labels)
    # plt.figure(figsize=(10, 7))
    # sn.heatmap(df_cm, annot=True, cmap=plt.cm.Blues, fmt='0000')
    # plt.show()
    return OA, KAPPA


# # statics samples classes info form labels. return a dict.
def get_samples_info(labels_samples):
    unique, counts = np.unique(labels_samples, return_counts=True)
    return dict(zip(unique, counts))


# # one-hot encoding for labels, return shape of (c,1), value is form  1 to c
# # parameter: c, number of classes, labels, train valid test label data, form is [1, 2, 1, 0, .....]
def one_hot_encode(c, labels):
    return np.eye(c)[[int(e-1) for e in labels]]


# # norma bands data(whole image), calculator each bands max mean min std value to obtain norma info.
# # norma bands data according to norma info, method including z-score and max-min.
# # parameter: bands data, and norma method.
def norma_data(data, norma_methods="z-score"):
    norma_info = []
    for i in range(data.shape[-1]):
        array = data.transpose(2, 0, 1)[i, :, :]
        min = np.min(array)
        max = np.max(array)
        mean = np.mean(array)
        std = np.std(array)
        list = [min, max, mean, std]
        norma_info.append(list)
    norma_info = np.stack(norma_info, axis=0)
    new_data = []
    for i, j in zip(range(norma_info.shape[0]), range(data.shape[-1])):
        norma_info1 = norma_info[i, :]
        array = data[:, :, j]
        if norma_methods == "z-score":
            new_array = (array-norma_info1[2])/norma_info1[3]
        else:
            new_array = (2*(array-norma_info1[0])/(norma_info1[1]-norma_info1[0]))-1
        new_data.append(new_array)
    new_data = np.stack(new_data, axis=-1)
    # # for save half memory!
    new_data = np.float32(new_data)
    return new_data


# As for labeling, one pixel might be labeled more twice.So we delete those pixel by index.
def delete_error_category(training_labels, training_samples):
    category = np.unique(training_labels)
    for i in category[20:]:
        index = np.argwhere(training_labels == i)
        training_labels = np.delete(training_labels, index)
        training_samples = np.delete(training_samples, index, axis=0)
    return training_samples, training_labels


# # palette is color map for rgb convert. preference setting.
# # including 16 types color, can increase or decrease.
palette = {0: (255, 255, 255),
           1: (0, 0, 128),
           2: (0, 128, 0),
           3: (128, 0, 0),
           4: (0, 0, 255),
           5: (0, 255, 0),
           6: (255, 0, 0),
           7: (0, 255, 255),
           8: (255, 255, 0),
           9: (0, 128, 128),
           10: (128, 128, 0),
           11: (255, 128, 128),
           12: (128, 128, 255),
           13: (128, 255, 128),
           14: (255, 128, 255),
           15: (165, 42, 42),
           16: (175, 238, 238)}


# # read train_data form .mat for converting to rgb color.
def convert_to_color(train_data_path):
    train_mat_dict = sio.loadmat(train_data_path)
    arr_2d = train_mat_dict[list(train_mat_dict.keys())[-1]]
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    plt.imshow(arr_3d)
    plt.show()


# # plot model training history, context of history included train and valid loss and accuracy.
# # parameter network, network == model.fit().
def plot_history(network):
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(network.history["loss"])
    plt.plot(network.history["val_loss"])
    plt.legend(["Training", "Validation"])

    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(network.history["acc"])
    plt.plot(network.history["val_acc"])
    plt.legend(["Training", "Validation"])
    plt.show()

# # increase plotting result function more.....


# # Over-sampling ways
# # random oversampling ways
# def ros(x_train, y_train):
#     over = RandomOverSampler()
#     x_train_over, y_train_over = over.fit_sample(x_train, y_train)
#     return x_train_over, y_train_over
#
#
# # SMOTE Ways
# def smote(x_train, y_train):
#     over1 = SMOTE()
#     x_train_over1, y_train_over1 = over1.fit_sample(x_train, y_train)
#     return x_train_over1, y_train_over1


def txt2xls(txt_path, xls_path, column):
    df = pd.read_csv(txt_path, sep='\t', header=None)
    new_df = df.iloc[np.arange(2, len(df), 3)]
    new_df = new_df[0].str.split(',', expand=True)
    l = []
    for i in range(0, len(column)):
        nd = new_df[i]
        nd = nd.str.split(':', expand=True)
        nd = nd.drop(0, axis=1)
        nd.rename(columns={1: column[i]}, inplace=True)
        l.append(nd)
    ne_d = pd.concat(l, axis=1)
    ne_d[column[1:]] = ne_d[column[1:]].apply(pd.to_numeric)
    ne_d.to_excel(xls_path, index=False)
