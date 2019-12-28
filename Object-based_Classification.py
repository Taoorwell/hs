# Load some packages and Classifiers
from python_gdal import *
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import mglearn

# File path, including train and full region file path
pwd = r"D:/JL/"
train_segments_path = pwd + r"images/WORKSPACE/GF2_4314_GS_3_train_1.shp"
region_segments_path = pwd + r"images/WORKSPACE/GF2_4314_GS_31.shp"

# Geo-pandas load and read shape-files to geo-pandas for training and prediction
train_segments = gpd.read_file(train_segments_path)
region_segments = gpd.read_file(region_segments_path)

# Select data values (features) to train
x_data = train_segments[train_segments.columns[1:13]]
y_data = train_segments["CLASS_ID"]

scaler = MinMaxScaler()
# Train and Test Split
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.0, random_state=3,
                                                    shuffle=True)
scaler.fit(x_train)
x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# Check Train and Test Samples Size
print("Train Samples List: {}\n".format(get_samples_info(y_train)))


# Load Classifiers and Grid Search and Cross Validation
# Define Parameter Grid
# knn = KNeighborsClassifier(n_neighbors=3)
# knn.fit(x_train, y_train)
# y_predict = knn.predict(x_test)
# print("Test Accuracy: {}".format(np.mean(y_test == y_predict)))

# param_grid = {'n_neighbors': [1, 3, 5, 7, 9]}
param_grid = {"C": [0.01, 0.1, 1, 10, 100, 1000], "gamma": [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(x_train, y_train)

# print("Test Set Score:{}".format(grid_search.score(x_test, y_test)))
# print(grid_search.predict(x_test))
print("Best Parameter:{}".format(grid_search.best_params_))
print("Best cross-validation score:{}".format(grid_search.best_score_))
print("Best estimator:{}".format(grid_search.best_estimator_))

results = pd.DataFrame(grid_search.cv_results_)
print(results)
scores = np.array(results['mean_test_score']).reshape(6, 6)
a = mglearn.tools.heatmap(scores, xlabel="gamma", xticklabels=param_grid["gamma"],
                          ylabel="C", yticklabels=param_grid["C"], cmap="viridis")
plt.colorbar(a)
plt.show()
# Predication on full region data
# Load
print(region_segments.columns)
region_data = region_segments[:][region_segments.columns[:-1]]
region_data = scaler.transform(region_data)
# print(len(region_data))
# #
# # # Prediction
region_predicts = grid_search.predict(region_data)
# # region_predicts = knn.predict(region_data)
print(region_predicts)
# #
# # # Write into origin shape-files
region_segments["predicts"] = region_predicts
print(region_segments)
region_segments.plot(column="predicts")
plt.show()
region_segments.to_file(filename=pwd + r"images/WORKSPACE/GF2_4314_GS_3_predicts1.shp")
