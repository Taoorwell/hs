import tensorflow as tf
from python_gdal import *
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
import mglearn

pwd = r"D:/JL/"
mat_images_path = pwd + r'images/mat/GF_2.mat'
mat_labels_path = pwd + r'images/mat/GF_2_LABEL_1.mat'
m = 35
c = 7
# lists = [400, 400, 400, 400, 400, 400, 400, 400]


def create_model(n, m):
    model1 = tf.keras.models.Sequential([tf.keras.layers.Dense(n, activation='relu', input_shape=(4,)),
                                         tf.keras.layers.Dropout(0.1),
                                         tf.keras.layers.Dense(m, activation='relu'),
                                         tf.keras.layers.Dropout(0.1),
                                         tf.keras.layers.Dense(c, activation='softmax')])
    model1.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                   loss='categorical_crossentropy', metrics=['accuracy'])
    return model1


# def create_model2(k, n):
#     model2 = tf.keras.models.Sequential([tf.keras.layers.Conv2D(k, (5, 5), padding='valid', input_shape=(m, m, 4)),
#                                          tf.keras.layers.BatchNormalization(),
#                                          tf.keras.layers.Activation(activation='relu'),
#                                          tf.keras.layers.MaxPool2D(2, padding='same'),
#                                          tf.keras.layers.Conv2D(k, (3, 3), padding='valid'),
#                                          tf.keras.layers.BatchNormalization(),
#                                          tf.keras.layers.Activation(activation='relu'),
#                                          tf.keras.layers.MaxPool2D(2, padding='same'),
#                                          tf.keras.layers.Conv2D(k, (3, 3), padding='valid'),
#                                          tf.keras.layers.BatchNormalization(),
#                                          tf.keras.layers.Activation(activation='relu'),
#                                          tf.keras.layers.MaxPool2D(2, padding='same'),
#                                          tf.keras.layers.Conv2D(k, (3, 3), padding='valid'),
#                                          tf.keras.layers.BatchNormalization(),
#                                          tf.keras.layers.Activation(activation='relu'),
#                                          tf.keras.layers.MaxPool2D(2, padding='same'),
#                                          tf.keras.layers.Flatten(),
#                                          tf.keras.layers.Dense(n, activation='relu'),
#                                          tf.keras.layers.Dropout(0.1),
#                                          tf.keras.layers.Dense(7, activation='softmax')])
#     model2.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
#     return model2
#
#
train_samples, train_labels = get_train_sample(data_path=mat_images_path,
                                               train_data_path=mat_labels_path,
                                               c=c,
                                               norma_methods='min-max,', m=1)

model2 = KerasClassifier(build_fn=create_model, batch_size=30, epochs=100, verbose=1)

# m = [48]
# k = [12, 18, 24, 30]
n = [16, 24, 32, 40]
m = [16, 24, 32, 40]

param_grid = dict(n=n, m=m)

grid_search = GridSearchCV(estimator=model2, param_grid=param_grid, n_jobs=-1, verbose=1, cv=5)
grid_search.fit(train_samples, train_labels)

print("Best Parameter:{}".format(grid_search.best_params_))
print("Best cross-validation score:{}".format(grid_search.best_score_))
print("Best estimator:{}".format(grid_search.best_estimator_))
results = pd.DataFrame(grid_search.cv_results_)
print(results)
scores = np.array(results['mean_test_score']).reshape(4, 4)
a = mglearn.tools.heatmap(scores, xlabel="n_1", xticklabels=param_grid["n"],
                          ylabel="n_2", yticklabels=param_grid["m"], cmap="viridis")
plt.colorbar(a)
plt.show()


