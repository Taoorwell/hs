import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d

file_path = "D:/JL/model/Results.xlsx"
df = pd.read_excel(file_path, sheet_name="CNN")
df1 = pd.read_excel(file_path, sheet_name="OCNN")
df2 = pd.read_excel(file_path, sheet_name="MLP")
df3 = pd.read_excel(file_path, sheet_name='SVM-OBJECT')
df4 = pd.read_excel(file_path, sheet_name='Time')
# print(df)
fig = plt.figure(num=0, figsize=(8, 4))

########
# Bar plot on Time!!!
# predict and train time of CNN
# barWidth = 0.25
# bar1 = list(df["T-train-gpu"])
# # bar3 = list(df["T-pre-gpu"]/60)
#
# # print(bar1)
# # bar11 = list(df_1d["KAPPA"])
# bar2 = list(df["T-train-cpu"])
# # bar4 = list(df["T-pre-cpu"]/60)
#
# # print(bar2)
# # bar22 = list(df_mlp["KAPPA"])
# # # print(bar1, bar2)
# r1 = np.arange(0, 8, 1)
# r2 = [x + barWidth for x in r1]
# # print(r1, r2)
# # r3 = [x + barWidth for x in r2]
# # r4 = [x + barWidth for x in r3]
#
# plt.bar(r1, bar1, color='black', width=barWidth, edgecolor='white', label='train_gpu', alpha=0.7)
# # plt.bar(r1, bar3, color='black', width=barWidth, edgecolor='white', label='predict_gpu', alpha=0.7)
# # plt.bar(r2, bar4, color='grey', width=barWidth, edgecolor='white', label='predict_cpu', alpha=0.7)
# plt.bar(r2, bar2, color='grey', width=barWidth, edgecolor='white', label='train-cpu', alpha=0.7)
#
# # plt.xlabel('M', fontweight='bold')
# plt.ylabel("Time(s)", fontweight='bold')
# plt.ylabel("Time(min)", fontweight='bold')
#
# plt.xticks([r+0.15 for r in np.arange(0, 9, 1)], ["35x35", "45x45", "55x55", "65x65",
#                                                   "75x75", "85x85", "95x95", "105x105"])
# plt.yticks([0, 160, 1000, 2000, 3000, 4000])
# # plt.ylim(0.5, 1.0)
# plt.hlines(y=160, xmin=-1, xmax=8, color='red', linestyles='dashed')
# plt.legend(loc=2, prop={'size': 12})
#
# plt.show()
########################################


##############################################
# Line plot on Overall accuracy and Kappa
# Overall accuracy and Kappa on CNN with different M
# x = np.arange(0, 8, 1)
# oa = df["OA"]
# k = df["K"]
#
# plt.plot(x, oa, color='black', label="OA", marker='D', linestyle='-.')
# plt.plot(x, k, color='grey', label="Kappa", marker='v', linestyle=':')
# plt.xticks(x, ["35x35", "45x45", "55x55", "65x65", "75x75", "85x85", "95x95", "105x105"])
# plt.legend(loc='best', prop={'size': 12})
# plt.xlabel("M", fontweight='bold')
# plt.ylabel("OA & Kappa", fontweight='bold')
# plt.show()
############################################
############################
# Plot surface 3D
# The Overall accuracy with two different factors: M and S
# x = np.arange(35, 115, 10)
# y = np.arange(20, 90, 10)
# X = np.zeros((7, 8))
# Y = np.zeros((8, 7))
#
# for i in range(0, 7, 1):
#     X[i] = x
# # print(X)
# for j in range(0, 8, 1):
#     Y[j] = y
# # print(Y)
# a = np.array(df1['OA']).reshape(8, 7)
#
# norm = plt.Normalize(a.min(), a.max())
# colors = cm.rainbow(norm(a))
# rcount, ccount, _ = colors.shape
# ax = fig.add_subplot(111, projection='3d')
# c = ax.plot_surface(X, Y.T, a.T, facecolors=colors, shade=False)
# c.set_facecolor((0, 0, 0, 0))
# ax.scatter(45, 30, 0.9092, marker='^', s=100, color='red')
# ax.text(45-1, 30, 0.9092, s='M=45, S=30')
# ax.scatter(105, 60, 0.5169, marker='v', s=100, color='cyan')
# ax.text(105-1, 60, 0.5169, s='M=105, S=60')
#
# ax.set_xticks(x)
# ax.set_xlabel("M")
# ax.set_ylabel("S")
# ax.set_zlabel("OA")
#
# plt.show()
############################

################################
# 3d bar plot
# predict time in CPU and GPU -OCNN
# ax = fig.add_subplot(111, projection='3d')
# for z in [80, 70, 60, 50, 40, 30, 20]:
#     df2 = df1[df1['S'] == z].sort_values("M")
#     m = df2["M"]
#     t1 = df2["T-pre-cpu"]
#     t2 = df2['T-pre-gpu']
#     # cs = [c]*len(m)
#     ax.bar(m-0.75, t1, zs=z, zdir='y', alpha=0.7, color='black', width=1.5, label='CPU_predict')
#     ax.bar(m+0.75, t2, zs=z, zdir='y', alpha=1.0, width=1.5, color='white',edgecolor='black', hatch="//",
#            label='GPU_predict')
# ax.set_xticks(np.arange(35, 115, 10))
# ax.set_xlabel("M")
# ax.set_ylabel("S")
# ax.set_zlabel("Time(s)")
# ax.legend(loc=2)
# plt.show()
###################################################
##################################################
# # # # # Bar Plot
# Overall accuracy and Kappa comparison between MLP CNN OCNN and obejct oriented.

# mlp_oa = df2[1][0]
# mlp_kappa = df2[1][1]
# print('Mlp Overall Accuracy:{} and Kappa:{}'.format(mlp_oa, mlp_kappa))
#
# cnn_oa = df['OA'][1]
# cnn_kappa = df['K'][1]
# print('Best_CNN Overall Accuracy:{} and Kappa:{}'.format(cnn_oa, cnn_kappa))
#
# ocnn_oa = df1[(df1['M'] == 45) & (df1["S"] == 30)]['OA'].values
# ocnn_kappa = df1[(df1['M'] == 45) & (df1["S"] == 30)]['KAPPA'].values
# print('Best_OCNN Overall Accuracy:{} and Kappa:{}'.format(ocnn_oa, ocnn_kappa))
#
# svm_oa = df3['OA'][0]
# svm_kappa = df3["KAPPA"][0]
# print('SVM Overall Accuracy:{} and Kappa:{}'.format(svm_oa, svm_kappa))
#
# barWidth = 0.25
# r1 = np.arange(0, 4, 1)
# r2 = [r + barWidth for r in r1]
# plt.bar(r1, [mlp_oa, cnn_oa, ocnn_oa, svm_oa], width=barWidth, label="OA", color='black', alpha=.8)
# plt.bar(r2, [mlp_kappa, cnn_kappa, ocnn_kappa, svm_kappa], width=barWidth, label='KAPPA',
#         alpha=1.0, color='white', edgecolor='black', hatch="//")
# plt.xticks([r+0.125 for r in np.arange(0, 4, 1)], ["MLP", "CNN", "OCNN", "SVM-OBJECT"])
# plt.ylim([0.6, 1])
# plt.xlabel("Classifiers")
# plt.ylabel("OA & KAPPA")
# plt.legend(loc=1, prop={'size': 12})
# plt.show()
###########################################################
################################################

time_segmentation = df4['Segmentation']
print(time_segmentation)

time_feature_extraction = df4['Feature Extraction']
print(time_feature_extraction)

time_training = df4['Training']
print(time_training)

time_predicting = df4['Predicting']
print(time_predicting)

ind = np.arange(0, 4, 1)
barWidth = 0.5

# hatch {'/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'}
plt.bar(ind, time_segmentation, color='red', edgecolor='black', width=barWidth, label='Segmentation', hatch='+')
plt.bar(ind, time_feature_extraction, color='yellow', edgecolor='black', width=barWidth, bottom=time_segmentation,
        label='Feature Extraction', hatch='*')
plt.bar(ind, time_training, color='green', edgecolor='black', width=barWidth, bottom=time_segmentation +
        time_feature_extraction, label='Training', hatch='\\')
plt.bar(ind, time_predicting, color='blue', edgecolor='black', alpha = 0.8, width=barWidth, bottom=time_training +
        time_segmentation + time_feature_extraction, label='Predicting', hatch='/')

plt.xticks(ind, df4['Models'])
plt.ylabel("Time(s)")
plt.legend(loc=1)
plt.show()

# # 3D plot B & M & OA KAPPA COF
# DATA = ["IP", "P", "PU", "KSC"]
# index = ["OA", "KAPPA", "Cof1", "Cof2", "Cof3"]
# for i in range(0, 4):
#     df_pca = df[(df["DATA"] == DATA[i]) & (df["Model_Category"] == "CNN_2D_PCA")]
#     for j in range(0, 5):
#         ax = plt.subplot2grid((4, 5), (i, j), projection="3d")
#         # ax = Axes3D(fig)
#         for c, z in zip(["r", "g", "c", "y", "b"], [1, 3, 5, 7, 9]):
#             df1 = df_pca[df_pca["B"] == z].sort_values("M")
#             m = list(df1["M"])
#             oa = list(df1[index[j]])
#             cs = [c] * len(m)
#             # cs[0] = "c"
#             ax.bar(m, oa, zs=z, zdir='y', color=cs, alpha=0.8, width=1.5)
#         ax.set_xticks(np.arange(5, 42, 4))
#         ax.set_yticks(np.arange(1, 10, 2))
#         ax.tick_params(axis='both', labelsize=6)
#         ax.set_xlabel("M", fontsize=8)
#         ax.set_ylabel("B", fontsize=8)
#         ax.set_zlabel(index[j], fontsize=8)
#         print(i, j)
#         print(df_pca)
# # ax.set_zlim()

# plt.show()
# DATA = ["IP", "P", "PU", "KSC"]
# index = ["OA", "KAPPA", "Cof1", "Cof2", "Cof3"]
# # # 3D PLOT CNN_2D
# df_2d = df[df["Model_Category"] == "CNN_2D"]
# for j in range(0, 5):
#     ax = plt.subplot2grid((1, 5), (0, j), projection="3d")
#     for c, z in zip(["r", "g", "c", "b"], [0, 1, 2, 3]):
#         df1 = df_2d[df_2d["DATA"] == DATA[z]].sort_values("M")
#         m = list(df1["M"])
#         oa = list(df1[index[j]])
#         cs = [c] * len(m)
#         # cs[0] = "c"
#         ax.bar(m, oa, zs=z, zdir='y', color=cs, alpha=0.8, width=1.5)
#     ax.set_xticks(np.arange(5, 42, 4))
#     ax.set_yticks([0, 1, 2, 3])
#     ax.set_yticklabels(DATA)
#     ax.tick_params(axis='both', labelsize=6)
#     ax.set_xlabel("M", fontsize=8)
#     ax.set_ylabel("DATA", fontsize=8)
#     ax.set_zlabel(index[j], fontsize=8)
# ax.set_zlim()

# plt.show()

# fig = plt.figure(figsize=(10, 5))
# # ax = Axes3D(fig)
# #
# # x = np.arange(-4, 4, 0.05)
# # y = np.arange(-4, 4, 0.05)
# # x, y = np.meshgrid(x, y)
# # z = np.sqrt(x**2 + y**2)
# # z = np.cos(z)
# # ax.set_xlabel("x")
# # ax.set_ylabel("y")
# # ax.set_zlabel("z")
# # # ax.tick_params(axis='both', width=0.005, colors='green')
# # ax.plot_surface(x, y, z, rstride=2, cstride=2, cmap='rainbow')
# plt.subplot(2, 1, 1)
# # ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)
# x = np.linspace(-1, 1, 100)
# y = x**2
#
# plt.plot(x, y, color='red', alpha=1, linewidth=1.0, label="Curve")
# plt.xlim((-0.5, 1.0))
# plt.ylim((0, 0.5))
# plt.xticks([-0.5, 0, 0.5, 1.0])
# plt.yticks([0, 0.25, 0.5],
#            [r"$top$", r"$median$", r"$high$"])
# plt.xlabel("X")
# plt.ylabel("Y")
# ax = plt.gca()
# ax.spines['right'].set_color('none')
# ax.spines['top'].set_color('none')
# ax.xaxis.set_ticks_position('bottom')
# ax.yaxis.set_ticks_position('left')
#
# ax.spines['bottom'].set_position(('data', 0.))
# ax.spines['left'].set_position(('data', 0.5))
#
# x0 = 0.4
# y0 = x0**2
#
# plt.scatter(x0, y0, s=50, color='b')
# plt.plot([x0, x0], [y0, 0], 'k--', lw=2.5)
#
# plt.annotate(r"x^2=y", xy=(x0, y0), xycoords='data', xytext=(40, -30), textcoords='offset points',
#              fontsize=10, arrowprops=dict(arrowstyle='- >', connectionstyle='arc3, rad=.1'))
# plt.text(x0-0.5, y0, r"This is Annotation")
# plt.legend(loc='best')
#
# plt.subplot(2, 3, 4)
# plt.plot(x, y, c='green')
#
# plt.subplot(2, 3, 5)
# plt.plot(x, x**3, c='b')
#
# plt.subplot(2, 3, 6)
# plt.plot(x, x**2+2*x, c='y')
#


# plt.figure()
# ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3, rowspan=1)
# X = np.linspace(-1, 1, 100)
# Y = X**2
# ax1.plot(X, Y, c='red', alpha=0.8)
# ax1.set_title("AX1")
# ax1.set_xticks(())
# ax1.set_yticks(())
#
# ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
# X1 = np.random.normal(0, 1, 1000)
# ax2.hist(X1, bins=10, color='pink', alpha=0.6)
# ax2.set_title("AX2")
# ax2.set_xticks(())
# ax2.set_yticks(())
#
# ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
# ax3.set_title("AX3")
#
#
# ax4 = plt.subplot2grid((3, 3), (2, 0))
# ax4.set_title("AX4")
# X4 = np.random.randint(1, 100, 100)
# Y4 = np.random.randint(1, 100, 100)
# t = np.arctan2(X4, Y4)
# ax4.scatter(X4, Y4, c=t, cmap='rainbow', s=10, alpha=0.8)
#
# ax5 = plt.subplot2grid((3, 3), (2, 1))
# ax5.set_title("AX5")
#
# plt.show()
#
# # plt.hist()
# plt.bar()


# b, m = np.meshgrid(b, m)
# # b2, oa1 = np.meshgrid(b, oa)
# oa, kappa = np.meshgrid(oa, kappa)
#
# fig = plt.figure(num=0, figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.set_xticks(np.arange(5, 42, 4))
# ax.set_yticks(np.arange(1, 10, 2))
# # ax.plot_trisurf(m, b, oa, cmap=cm.jet, edgecolor='none')
# # ax.scatter(m, b, oa, cmap='rainbow', alpha=0.5, s=5)
# # ax.scatter(m, b, kappa, color='green', alpha=0.5)
# # ax.scatter3D(m, b, oa, color='blue', alpha=0.8)
# # ax.plot_wireframe(m, b, oa)
# ax.plot_surface(m, b, oa, color='red')
# # ax.plot(m, b, oa, zdir='z', color='pink', alpha=0.8)
# # plt.pcolormesh(m, b, oa)
# # plt.colorbar()
# # plt.contourf(m, b, oa, 30)
