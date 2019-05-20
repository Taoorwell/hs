# from python_gdal import *
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn
import pandas as pd

cnn_1d_mat_path = r'F:/paper/HSI/code/predicts_mat/cnn_1d/'
cnn_2d_mat_path = r'F:/paper/HSI/code/predicts_mat/cnn_2d/'
features_fusion_mat_path = r'F:/paper/HSI/code/predicts_mat/features_fusion/'
mlp_mat_path = r'F:/paper/HSI/code/predicts_mat/mlp/'

cnn_1d_mat_list = os.listdir(cnn_1d_mat_path)
cnn_1d_list = [cnn_1d_mat_path+f for f in cnn_1d_mat_list]
cnn_2d_mat_list = os.listdir(cnn_2d_mat_path)
cnn_2d_list = [cnn_2d_mat_path+f for f in cnn_2d_mat_list]
features_fusion_mat_list = os.listdir(features_fusion_mat_path)
features_fusion_list = [features_fusion_mat_path+f for f in features_fusion_mat_list]
mlp_mat_list = os.listdir(mlp_mat_path)
mlp_list = [mlp_mat_path+f for f in mlp_mat_list]


# print(len(cnn_1d_list))
# print(len(cnn_2d_list))
# print(len(features_fusion_list))
# print(len(mlp_list))


def get_confidence_distribution(pre, shape):
    pre = sio.loadmat(pre)
    pre = pre[list(pre.keys())[-1]]
    max_prob = np.max(pre, axis=1)
    low_conf = len([x for x in max_prob if x < 0.5]) / (shape[0] * shape[1])
    high_conf0 = len([x for x in max_prob if x >= 0.5]) / (shape[0] * shape[1])
    high_conf1 = len([x for x in max_prob if x >= 0.6]) / (shape[0] * shape[1])
    high_conf2 = len([x for x in max_prob if x >= 0.7]) / (shape[0] * shape[1])
    high_conf3 = len([x for x in max_prob if x >= 0.8]) / (shape[0] * shape[1])
    high_conf4 = len([x for x in max_prob if x >= 0.9]) / (shape[0] * shape[1])
    return low_conf, high_conf0, high_conf1, high_conf2, high_conf3, high_conf4


# DATA, MODEL, M, C00, C0, C1, C2, C3, C4 = [], [], [], [], [], [], [], [], []
#################################
# # MLP
# for f in mlp_list:
#     d = f.split('/')[-1].split('.')[0].split('_')[-1]
#     m = f.split('/')[-2]
#     print(d)
#     print(m)
#     if d == 'IP':
#         shape = (145, 145)
#     elif d == 'KSC':
#         shape = (512, 614)
#     elif d == 'PU':
#         shape = (610, 340)
#     else:
#         shape = (1096, 715)
#     low_conf, high_conf0, high_conf1, high_conf2, high_conf3, high_conf4 = get_confidence_distribution(f, shape)
#     DATA.append(d)
#     MODEL.append(m)
#     C00.append(low_conf)
#     C0.append(high_conf0)
#     C1.append(high_conf1)
#     C2.append(high_conf2)
#     C3.append(high_conf3)
#     C4.append(high_conf4)
# df = pd.DataFrame(data={'DATA': DATA, 'MODEL': MODEL, 'C00': C00, 'C0': C0,
#                         'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4})
# print(df)
# df.to_csv(r'F:/paper/HSI/code/out_files/mlp_cof.csv', index=False, header=True)
#################################
###############################
# CNN_1D
# for f in cnn_1d_list:
#     d = f.split('/')[-1].split('.')[0].split('_')[-1]
#     m = f.split('/')[-2]
#     print(d)
#     print(m)
#     if d == 'IP':
#         shape = (145, 145)
#     elif d == 'KSC':
#         shape = (512, 614)
#     elif d == 'PU':
#         shape = (610, 340)
#     else:
#         shape = (1096, 715)
#     low_conf, high_conf0, high_conf1, high_conf2, high_conf3, high_conf4 = get_confidence_distribution(f, shape)
#     DATA.append(d)
#     MODEL.append(m)
#     C00.append(low_conf)
#     C0.append(high_conf0)
#     C1.append(high_conf1)
#     C2.append(high_conf2)
#     C3.append(high_conf3)
#     C4.append(high_conf4)
# df = pd.DataFrame(data={'DATA': DATA, 'MODEL': MODEL, 'C00': C00, 'C0': C0,
#                         'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4})
# print(df)
# df.to_csv(r'F:/paper/HSI/code/out_files/cnn_1d_cof.csv', index=False, header=True)

##################################

#################################
# CNN 2D
# for f in cnn_2d_list:
#     m_ = int(f.split('/')[-1].split('-')[0])
#     d = f.split('/')[-1].split('.')[0].split('_')[-1]
#     m = f.split('/')[-2]
#     print(d)
#     print(m)
#     if d == 'IP':
#         shape = (145, 145)
#     elif d == 'KSC':
#         shape = (512, 614)
#     elif d == 'PU':
#         shape = (610, 340)
#     else:
#         shape = (1096, 715)
#     low_conf, high_conf0, high_conf1, high_conf2, high_conf3, high_conf4 = get_confidence_distribution(f, shape)
#     M.append(m_)
#     DATA.append(d)
#     MODEL.append(m)
#     C00.append(low_conf)
#     C0.append(high_conf0)
#     C1.append(high_conf1)
#     C2.append(high_conf2)
#     C3.append(high_conf3)
#     C4.append(high_conf4)
# df = pd.DataFrame(data={'DATA': DATA, 'MODEL': MODEL, 'C00': C00, 'C0': C0,
#                         'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'M': M})
# print(df)
# df.to_csv(r'F:/paper/HSI/code/out_files/cnn_2d_cof.csv', index=False)
#####################################################################
# features fusion
# for f in features_fusion_list:
#     m_ = int(f.split('/')[-1].split('-')[0])
#     d = f.split('/')[-1].split('.')[0].split('_')[-1]
#     m = f.split('/')[-2]
#     print(d)
#     print(m)
#     if d == 'IP':
#         shape = (145, 145)
#     elif d == 'KSC':
#         shape = (512, 614)
#     elif d == 'PU':
#         shape = (610, 340)
#     else:
#         shape = (1096, 715)
#     low_conf, high_conf0, high_conf1, high_conf2, high_conf3, high_conf4 = get_confidence_distribution(f, shape)
#     M.append(m_)
#     DATA.append(d)
#     MODEL.append(m)
#     C00.append(low_conf)
#     C0.append(high_conf0)
#     C1.append(high_conf1)
#     C2.append(high_conf2)
#     C3.append(high_conf3)
#     C4.append(high_conf4)
# df = pd.DataFrame(data={'DATA': DATA, 'MODEL': MODEL, 'C00': C00, 'C0': C0,
#                         'C1': C1, 'C2': C2, 'C3': C3, 'C4': C4, 'M': M})
# print(df)
# df.to_csv(r'F:/paper/HSI/code/out_files/ff_cof.csv', index=False, header=True)
##############################################################################
###########################################################################
# # new_df for excel
main_path = r'F:/paper/HSI/code/out_files/'
# df_cnn_1d = pd.read_csv(main_path+'cnn_1d_cof.csv')
# # print(df_cnn_1d)
# df_cnn_2d = pd.read_csv(main_path+'cnn_2d_cof.csv')
# # print(df_cnn_2d)
# df_ff = pd.read_csv(main_path+'ff_cof.csv')
# # print(df_ff)
# df_mlp = pd.read_csv(main_path+'mlp_cof.csv')
# # print(df_mlp)
# new_df = pd.concat((df_mlp, df_cnn_1d, df_cnn_2d, df_ff), axis=0)
# print(new_df)
# new_df.to_excel(r'F:/paper/HSI/code/out_files/cof.xlsx', index=False, header=True)
############################################################3

###################################################
# plot
# # select
df = pd.read_excel(main_path+'cof.xlsx')
# print(df)
df_PU_cnn_2d = df[(df['DATA'] == 'PU') & (df['MODEL'] == 'cnn_2d')].sort_values('M')
df_PU_FF = df[(df['DATA'] == 'PU') & (df['MODEL'] == 'features_fusion')].sort_values('M')

df_PC_cnn_2d = df[(df['DATA'] == 'P') & (df['MODEL'] == 'cnn_2d')].sort_values('M')
df_PC_FF = df[(df['DATA'] == 'P') & (df['MODEL'] == 'features_fusion')].sort_values('M')
# print(df_PU_cnn_2d)
fig = plt.figure(num=0, figsize=(6, 4), dpi=100)
plt.subplot(221)
plt.plot(df_PU_cnn_2d['M'], df_PU_cnn_2d['C00'], c='r', marker='o', label='C < 0.5')
# plt.plot(df_PU_cnn_2d['M'], df_PU_cnn_2d['C0'], c='r', marker='o', label='C > 0.5')
# plt.plot(df_PU_cnn_2d['M'], df_PU_cnn_2d['C1'], c='c', marker='+', label='C > 0.6')
# plt.plot(df_PU_cnn_2d['M'], df_PU_cnn_2d['C2'], c='blue', marker='1', label='C > 0.7')
# plt.plot(df_PU_cnn_2d['M'], df_PU_cnn_2d['C3'], c='orange', marker='v', label='C > 0.8')
# plt.plot(df_PU_cnn_2d['M'], df_PU_cnn_2d['C4'], c='green', marker='*', label='C > 0.9')
plt.xticks(np.arange(5, 42, 4))
plt.ylabel('Confidences')
plt.xlabel('M')
# plt.legend()

plt.subplot(222)
plt.plot(df_PU_FF['M'], df_PU_FF['C00'], c='r', marker='o', label='C < 0.5')
# plt.plot(df_PU_FF['M'], df_PU_FF['C0'], c='r', marker='o', label='C > 0.5')
# plt.plot(df_PU_FF['M'], df_PU_FF['C1'], c='c', marker='+', label='C > 0.6')
# plt.plot(df_PU_FF['M'], df_PU_FF['C2'], c='blue', marker='1', label='C > 0.7')
# plt.plot(df_PU_FF['M'], df_PU_FF['C3'], c='orange', marker='v', label='C > 0.8')
# plt.plot(df_PU_FF['M'], df_PU_FF['C4'], c='green', marker='*', label='C > 0.9')
plt.xticks(np.arange(5, 42, 4))
# plt.ylabel('Confidences')
plt.xlabel('M')
plt.legend()

plt.subplot(223)
plt.plot(df_PC_cnn_2d['M'], df_PC_cnn_2d['C00'], c='r', marker='o', label='C < 0.5')
# plt.plot(df_PC_cnn_2d['M'], df_PC_cnn_2d['C0'], c='r', marker='o', label='C > 0.5')
# plt.plot(df_PC_cnn_2d['M'], df_PC_cnn_2d['C1'], c='c', marker='+', label='C > 0.6')
# plt.plot(df_PC_cnn_2d['M'], df_PC_cnn_2d['C2'], c='blue', marker='1', label='C > 0.7')
# plt.plot(df_PC_cnn_2d['M'], df_PC_cnn_2d['C3'], c='orange', marker='v', label='C > 0.8')
# plt.plot(df_PC_cnn_2d['M'], df_PC_cnn_2d['C4'], c='green', marker='*', label='C > 0.9')
plt.xticks(np.arange(5, 42, 4))
plt.ylabel('Confidences')
plt.xlabel('M')
# plt.legend()

plt.subplot(224)
plt.plot(df_PC_FF['M'], df_PC_FF['C00'], c='r', marker='o', label='C < 0.5')
# plt.plot(df_PC_FF['M'], df_PC_FF['C0'], c='r', marker='o', label='C > 0.5')
# plt.plot(df_PC_FF['M'], df_PC_FF['C1'], c='c', marker='+', label='C > 0.6')
# plt.plot(df_PC_FF['M'], df_PC_FF['C2'], c='blue', marker='1', label='C > 0.7')
# plt.plot(df_PC_FF['M'], df_PC_FF['C3'], c='orange', marker='v', label='C > 0.8')
# plt.plot(df_PC_FF['M'], df_PC_FF['C4'], c='green', marker='*', label='C > 0.9')
plt.xticks(np.arange(5, 42, 4))
# plt.ylabel('Confidences')
plt.xlabel('M')
plt.legend()


plt.show()
