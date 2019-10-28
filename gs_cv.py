import tensorflow as tf
from python_gdal import *
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

pwd = r"D:/JL/"
mat_images_path = pwd + r'images/mat/GF_2.mat'
mat_labels_path = pwd + r'images/mat/GF_2_LABEL.mat'
lists = [400, 400, 400, 400, 400, 400, 400, 400]


def create_model(n, m, l):
    model1 = tf.keras.models.Sequential([tf.keras.layers.Dense(n, activation='relu', input_shape=(4,)),
                                         tf.keras.layers.Dropout(0.1),
                                         tf.keras.layers.Dense(m, activation='relu'),
                                         tf.keras.layers.Dropout(0.1),
                                         tf.keras.layers.Dense(8, activation='softmax')])
    model1.compile(optimizer=tf.keras.optimizers.Adam(lr=l),
                   loss='categorical_crossentropy', metrics=['accuracy'])
    return model1


train_samples, train_labels = get_train_sample(data_path=mat_images_path,
                                               train_data_path=mat_labels_path,
                                               c=8, seed=10, lists=lists,
                                               norma_methods='min-max,')

model1 = KerasClassifier(build_fn=create_model, batch_size=30, epochs=200, verbose=1)

n = [16, 24, 32]
m = [24, 32, 48]
l = [0.1, 0.01, 0.001]
param_grid = dict(n=n, m=m, l=l)

grid = GridSearchCV(estimator=model1, param_grid=param_grid, n_jobs=-1)
grid_result = grid.fit(train_samples, train_labels)
print('[INFO] Best: %f using %s' % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print('[INFO] %f (%f) with %r' % (scores.mean(), scores.std(), params))
# model1 = create_model(n=32, m=32, l=0.01)
# model1.summary()

