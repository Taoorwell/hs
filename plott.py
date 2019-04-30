import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_path = "file.xlsx"
df = pd.read_excel(file_path)
fig = plt.figure(num=0, figsize=(8, 4))
# # B & M & IP_PCA DATA
# print(df_ip_pca)
# # IP & CNN_2D
# df_ip = df[df["Model_Category"] == "CNN_2D"].sort_values("M")
# print(df_ip_pca)


# # CNN_1D & MLP
df_1d = df[df['Model_Category'] == 'CNN_1D']
df_mlp = df[df['Model_Category'] == 'MLP']
# CNN_2D & DATA & M
# df_ip = df[(df["DATA"] == "IP") & (df["Model_Category"] == "CNN_2D")].sort_values("M")
# df_p = df[(df["DATA"] == "P") & (df["Model_Category"] == "CNN_2D")].sort_values("M")
# df_pu = df[(df["DATA"] == "PU") & (df["Model_Category"] == "CNN_2D")].sort_values("M")
# df_ksc = df[(df["DATA"] == "KSC") & (df["Model_Category"] == "CNN_2D")].sort_values("M")
# print(df_ip)


barWidth = 0.25
bar1 = list(df_1d["OA"])
bar11 = list(df_1d["KAPPA"])
bar2 = list(df_mlp["OA"])
bar22 = list(df_mlp["KAPPA"])
# print(bar1, bar2)
r1 = np.arange(0, 4, 1)
r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
# r4 = [x + barWidth for x in r3]

plt.bar(r1, bar11, color='red', width=barWidth, edgecolor='white', label='CNN_1D', alpha=0.5)
# plt.bar(r1, bar11, color='red', width=barWidth, edgecolor='white', alpha=0.5)
# plt.bar(r2, bar2, color='blue', width=barWidth, edgecolor='white', alpha=0.5)
plt.bar(r2, bar22, color='blue', width=barWidth, edgecolor='white', label='MLP', alpha=0.5)

plt.xlabel('DATA', fontweight='bold')
plt.ylabel("KAPPA", fontweight='bold')
plt.xticks([r+0.15 for r in np.arange(0, 4, 1)], ["IP", "KSC", "P", "PU"])
plt.ylim(0.5, 1.0)
plt.legend(loc=2, prop={'size': 8})

plt.show()

# # subplot for Data
# df3 = [df_ip, df_p, df_pu, df_ksc]
# tup = [(0, 0), (0, 1), (1, 0), (1, 1)]
# for df, i in zip(df3, tup):
#     ax = plt.subplot2grid((2, 2), i)
#     ax.bar(list(df["M"]), list(df["OA"]), label="OA", color="red", alpha=1.0)
#     # ax.bar(list(df["M"]), list(df["KAPPA"]), label="KAPPA", color="yellow", alpha=0.8)
#     ax.set_xticks(np.arange(5, 42, 4))
#     ax.set_ylim(0.9, 1.0)
#     ax.set_xlabel("M")
#     ax.set_ylabel("OA", fontsize=8)
#     # ax.legend()


# group bar plot
# barWidth = 0.25
# bar1 = list(df_ip["OA"])
# bar2 = list(df_p["OA"])
# bar3 = list(df_pu["OA"])
# # bar4 = list(df_ksc["OA"])
#
# r1 = np.arange(0, 20, 2)
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
# r4 = [x + barWidth for x in r3]
#
# plt.bar(r1, bar1, color='red', width=barWidth, edgecolor='white', label='IP')
# plt.bar(r2, bar2, color='blue', width=barWidth, edgecolor='white', label='P', alpha=0.5)
# plt.bar(r3, bar3, color='pink', width=barWidth, edgecolor='white', label='PU')
# # plt.bar(r4, bar4, color='blue', width=barWidth, edgecolor='white', label='KSC')
#
# plt.xlabel('M', fontweight='bold')
# plt.ylabel("OA", fontweight='bold')
# plt.xticks([r + barWidth for r in np.arange(0, 20, 2)], np.arange(5, 42, 4))
# plt.ylim(0.9, 1.0)
# plt.legend(loc=2, prop={'size': 6})
#
# plt.show()
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

plt.show()

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
