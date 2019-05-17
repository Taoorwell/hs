from python_gdal import *

MAIN_FOLDER = r'E:/HSI/'
IP_DATA_PATH = 'IP/Indian_pines_corrected'
IP_TRAIN_PATH = 'IP/Indian_pines_gt'
PAVIA_DATA_PATH = "Pavia/Pavia"
PAVIA_TRAIN_PATH = "Pavia/Pavia_gt"
PAVIA_U_DATA_PATH = "Pavia/PaviaU"
PAVIA_U_TRAIN_PATH = "Pavia/PaviaU_gt"
KSC_DATA_PATH = "KSC/KSC"
KSC_TRAIN_PATH = "KSC/KSC_gt"

DATA_PATH = [IP_DATA_PATH, PAVIA_DATA_PATH, PAVIA_U_DATA_PATH, KSC_DATA_PATH]
TRAIN_PATH = [IP_TRAIN_PATH, PAVIA_TRAIN_PATH, PAVIA_U_TRAIN_PATH, KSC_TRAIN_PATH]
image_shape = [(145, 145), (1096, 715), (610, 340), (512, 614)]
bands_composite = [[28, 19, 10], [10, 27, 46]]

i = 3
##################################
# # plot original images!
# # get bands data from .mat files!


def display_original_images(mat_data_path, train_mat_data_path, bands):
    bands_data_dict = sio.loadmat(mat_data_path)
    bands_data = bands_data_dict[list(bands_data_dict.keys())[-1]]
    bands_data = norma_data(bands_data, norma_methods='min-max')
    labeled_pixel_dict = sio.loadmat(train_mat_data_path)
    labeled_pixel = labeled_pixel_dict[list(labeled_pixel_dict.keys())[-1]]
    arr_2d = labeled_pixel
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i
    fig = plt.figure(num=0, figsize=(6, 4), dpi=300)
    plt.subplot(122)
    plt.imshow(arr_3d)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(121)
    band_data = np.stack([bands_data[:, :, bands[0]], bands_data[:, :, bands[1]],
                         bands_data[:, :, bands[2]]], axis=-1)
    plt.imshow((band_data*255).astype(np.uint8))
    # plt.tick_params('off')
    plt.xticks([])
    plt.yticks([])
    plt.show()


# display_original_images(mat_data_path=MAIN_FOLDER+DATA_PATH[i],
#                         train_mat_data_path=MAIN_FOLDER+TRAIN_PATH[i],
#                         bands=bands_composite[0])
###################################################################################

###################################################################################
# # PLOT PREDICTIONS IMAGES & MLP & 1D CNN
# mlp_path = r'E:/HSI/code/predicts_mat/mlp/'
# mlp_list = os.listdir(mlp_path)
# # print(mlp_list)
#
# cnn_1d_path = r'E:/HSI/code/predicts_mat/cnn_1d/'
# cnn_1d_list = os.listdir(cnn_1d_path)
# print(cnn_1d_list)

# k = [0, -1, 1, 2]
# for i, j, k1 in zip([mlp_path + x for x in mlp_list], [cnn_1d_path + x for x in cnn_1d_list], k):
#     prediction_1 = sio.loadmat(i)
#     prediction_2 = sio.loadmat(j)
#     predictions_1 = prediction_1[list(prediction_1.keys())[-1]]
#     predictions_2 = prediction_2[list(prediction_2.keys())[-1]]
#     fig = plt.figure(num=1, figsize=(6, 4), dpi=300)
#     ax1 = plt.subplot2grid((1, 4), (0, 0))
#     write_region_image_classification_result(predictions_1, train_data_path=MAIN_FOLDER+TRAIN_PATH[k1],
#                                              shape=image_shape[k1])
#     ax1.set_xlabel('MLP_R')
#
#     ax2 = plt.subplot2grid((1, 4), (0, 1))
#     write_region_image_classification_result(predictions_2, train_data_path=MAIN_FOLDER+TRAIN_PATH[k1],
#                                              shape=image_shape[k1])
#     ax2.set_xlabel('1D CNN_R')
#
#     ax3 = plt.subplot2grid((1, 4), (0, 2))
#     write_whole_image_classification_result(predictions_1, shape=image_shape[k1])
#     ax3.set_xlabel('MLP_W')
#
#     ax4 = plt.subplot2grid((1, 4), (0, 3))
#     write_whole_image_classification_result(predictions_2, shape=image_shape[k1])
#     ax4.set_xlabel('1D CNN_W')
#     plt.show()
###############################################################################

##############################################################################
# # # # PLOT PREDICTION IMAGES & 2D CNN
# cnn_2d_path = r'E:/HSI/code/predicts_mat/cnn_2d/'
# cnn_2d_list = os.listdir(cnn_2d_path)
# # print(cnn_2d_list)
# IP, KSC, PC, PU = [], [], [], []
# for f in cnn_2d_list:
#     data = f.split(".")[0].split("_")[-1]
#     if data == 'IP':
#         IP.append(f)
#     elif data == 'KSC':
#         KSC.append(f)
#     elif data == 'P':
#         PC.append(f)
#     elif data == 'PU':
#         PU.append(f)
# # print(IP, '\n', KSC)
# L = [IP, PC, PU, KSC]
# # print(len(IP), len(KSC), len(PU), len(PC))
# for l in L:
#     fig = plt.figure(num=1, figsize=(12, 4), dpi=300)
#     for i in range(0, 2):
#         for j in range(0, 10):
#             prediction = sio.loadmat(cnn_2d_path+l[j])
#             prediction = prediction[list(prediction.keys())[-1]]
#             print(i, j)
#             ax = plt.subplot2grid((2, 10), (i, j))
#             if i == 0:
#                 write_region_image_classification_result(prediction,
#                                                          train_data_path=MAIN_FOLDER+TRAIN_PATH[L.index(l)],
#                                                          shape=image_shape[L.index(l)])
#                 ax.set_xlabel(str(l[j].split('-')[0]) + 'R')
#             else:
#                 write_whole_image_classification_result(prediction, shape=image_shape[L.index(l)])
#                 ax.set_xlabel(str(l[j].split('-')[0]) + 'W')
#     plt.show()
##############################################################################################################

##############################################################################################################
# # # PLOT PROBABILITY OF PREDICTION IMAGES
cnn_2d_path = r'E:/HSI/code/predicts_mat/cnn_2d/'
cnn_2d_list = os.listdir(cnn_2d_path)
# print(cnn_2d_list)
IP, KSC, PC, PU = [], [], [], []
for f in cnn_2d_list:
    data = f.split(".")[0].split("_")[-1]
    if data == 'IP':
        IP.append(f)
    elif data == 'KSC':
        KSC.append(f)
    elif data == 'P':
        PC.append(f)
    elif data == 'PU':
        PU.append(f)
# print(IP, '\n', KSC)
L = [IP, PC, PU, KSC]
# print(len(IP), len(KSC), len(PU), len(PC))
for l in L:
    fig = plt.figure(num=1, figsize=(12, 4), dpi=300)
    for i in range(0, 2):
        for j in range(0, 10):
            prediction = sio.loadmat(cnn_2d_path+l[j])
            prediction = prediction[list(prediction.keys())[-1]]
            print(i, j)
            ax = plt.subplot2grid((2, 10), (i, j))
            if i == 0:
                write_whole_image_predicts_prob(prediction, shape=image_shape[L.index(l)])
                # ax.axis('off')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel(str(l[j].split('-')[0]), fontsize=3)
            else:
                cof1 = write_whole_image_predicts_prob1(prediction, shape=image_shape[L.index(l)])
                ax.set_xlabel(str(l[j].split('-')[0]) + ' {:.4f}'.format(cof1), fontsize=3)
                ax.set_xticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
                ax.tick_params(axis='both', which='major', labelsize=3)
                # ax.set_yticks([])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                if j != 0:
                    ax.spines['left'].set_visible(False)
                    ax.set_yticks([])
    plt.show()

