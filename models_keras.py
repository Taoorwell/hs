from keras.models import Sequential, save_model, load_model, Model
from keras.layers import Conv1D, Conv2D, Conv3D, Activation, MaxPool1D, MaxPool2D, MaxPool3D, Flatten, Dropout, Input
from keras.layers import Dense, GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D, concatenate
from keras.layers import SpatialDropout1D, SpatialDropout2D, SpatialDropout3D
from keras.regularizers import l2
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import backend as K
import numpy as np


def mlp(input_shape, c, lr, rate1, rate2, l):
    inputs = Input(input_shape)
    x = Dense(32, activation='relu')(inputs)
    # x = BatchNormalization()(x)
    # x = Activation(activation='relu')(x)
    x = Dropout(rate=rate1)(x)
    x = Dense(64, activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Activation(activation='relu')(x)
    x = Dropout(rate=rate2)(x)
    outputs = Dense(c, activation='softmax', activity_regularizer=l2(l))(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=lr, beta_1=0.95, decay=0.01), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def cnn_1d(input_shape, c, lr, rate1, rate2, rate3, l):

    inputs = Input(shape=input_shape)

    x = Conv1D(24, 3, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPool1D(2, padding='same')(x)

    x = Conv1D(36, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPool1D(3, padding='same')(x)
    x = Dropout(rate=rate1)(x)

    x = Conv1D(48, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPool1D(2, padding='same')(x)
    x = Dropout(rate=rate2)(x)

    x = Conv1D(96, 4, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPool1D(2, padding='same')(x)
    x = Dropout(rate=rate2)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(rate=rate3)(x)
    outputs = Dense(c, activation='softmax', activity_regularizer=l2(l))(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=lr, beta_1=0.95, decay=0.01), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def cnn_2d_pca(input_shape, c, lr, l, rate1, rate2):
    inputs = Input(shape=input_shape)

    x = Conv2D(12, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)

    x = Conv2D(24, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)

    x = Conv2D(48, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)
    x = Dropout(rate=rate1)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu', activity_regularizer=l2(l))(x)
    x = Dropout(rate=rate2)(x)
    outputs = Dense(c, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=lr, beta_1=0.95, decay=0.001), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


# def cnn_2d(input_shape, c, lr):
#     inputs = Input(shape=input_shape)
#
#     x = Conv2D(12, (3, 3), padding='same')(inputs)
#     x = BatchNormalization()(x)
#     x = Activation(activation='relu')(x)
#
#     x = Conv2D(24, (3, 3), padding='same')(x)
#     # x = SpatialDropout2D(0.5)(x)
#     x = BatchNormalization()(x)
#     x = Activation(activation='relu')(x)
#
#     x = Conv2D(48, (3, 3), padding='same')(x)
#     x = SpatialDropout2D(0.25)(x)
#     x = BatchNormalization()(x)
#     x = Activation(activation='relu')(x)
#     x = MaxPool2D(pool_size=2, padding='same')(x)
#
#     x = Flatten()(x)
#     x = Dense(128, activation='relu', activity_regularizer=l2(0.0001))(x)
#     x = Dropout(0.25)(x)
#     outputs = Dense(c, activation='softmax')(x)
#     model = Model(inputs=inputs, outputs=outputs)
#     model.compile(optimizer=Adam(lr=lr, beta_1=0.95, decay=0.001), loss="categorical_crossentropy",
#                   metrics=["accuracy"])
#     return model
def cnn_2d(input_shape, c, lr):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    # x = SpatialDropout2D(0.25)(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPool2D(pool_size=2, padding='same')(x)
    x = Dropout(rate=0.25)(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu', activity_regularizer=l2(0.0001))(x)
    x = Dropout(rate=0.25)(x)
    outputs = Dense(c, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=lr, beta_1=0.95, decay=0.001), loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def cnn_3d(input_shape, c, lr):
    inputs = Input(shape=input_shape)

    x = Conv3D(12, (3, 3, 25), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    x = Conv3D(24, (3, 3, 25), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)

    x = Conv3D(48, (3, 3, 25), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation(activation='relu')(x)
    x = MaxPool3D(pool_size=(2, 2, 2), padding='valid')(x)

    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(c, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def combined_1d_2d(inputs_1, inputs_2, c, lr):
    inputs_1 = Input(shape=inputs_1)
    inputs_2 = Input(shape=inputs_2)
    x_1 = Conv1D(filters=12, kernel_size=5, padding='same')(inputs_1)
    x_1 = BatchNormalization()(x_1)
    x_1 = Activation(activation='relu')(x_1)
    x_1 = MaxPool1D(pool_size=2, padding='same')(x_1)

    x_1 = Conv1D(filters=24, kernel_size=5, padding='same')(x_1)
    x_1 = BatchNormalization()(x_1)
    x_1 = Activation(activation='relu')(x_1)
    x_1 = MaxPool1D(pool_size=2, padding='same')(x_1)

    x_1 = Flatten()(x_1)

    x_2 = Conv2D(filters=24, kernel_size=(3, 3), padding='same')(inputs_2)
    x_2 = BatchNormalization()(x_2)
    x_2 = Activation(activation='relu')(x_2)

    x_2 = Conv2D(filters=48, kernel_size=(3, 3), padding='same')(x_2)
    x_2 = BatchNormalization()(x_2)
    x_2 = Activation(activation='relu')(x_2)

    x_2 = MaxPool2D(pool_size=2, padding='same')(x_2)

    x_2 = Flatten()(x_2)

    combine_layer = concatenate([x_1, x_2], axis=1)

    y = Dense(256, activation='relu')(combine_layer)

    output = Dense(c, activation='softmax')(y)

    model = Model(inputs=[inputs_1, inputs_2], outputs=output)
    model.compile(optimizer=Adam(lr=lr), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


def inception_1d_v1(input):
    input = Input(shape=input)

    tower1 = Conv1D(32, 3, padding='same')(input)
    tower1 = BatchNormalization()(tower1)
    tower1 = Activation(activation='relu')(tower1)

    tower2 = Conv1D(32, 5, padding='same')(input)
    tower2 = BatchNormalization()(tower2)
    tower2 = Activation(activation='relu')(tower2)

    tower3 = Conv1D(24, 1, padding='same')(input)
    tower3 = BatchNormalization()(tower3)
    tower3 = Activation(activation='relu')(tower3)

    tower4 = MaxPool1D(pool_size=3, strides=1, padding='same')(input)
    tower4 = Conv1D(24, 1, padding='same')(tower4)
    tower4 = BatchNormalization()(tower4)
    tower4 = Activation(activation='relu')(tower4)

    output = concatenate([tower1, tower2, tower3, tower4], axis=-1)
    return output


def inception_1d_v2(input):
    input = Input(shape=input)
    tower1 = Conv1D(12, 1, padding='same')(input)
    tower1 = BatchNormalization()(tower1)
    tower1 = Activation(activation='relu')(tower1)
    tower1 = Conv1D(24, 3, padding='same')(tower1)
    tower1 = BatchNormalization()(tower1)
    tower1 = Activation(activation='relu')(tower1)
    tower1 = Conv1D(36, 3, padding='same')(tower1)
    tower1 = BatchNormalization()(tower1)
    tower1 = Activation(activation='relu')(tower1)

    tower2 = Conv1D(12, 1, padding='same')(input)
    tower2 = BatchNormalization()(tower2)
    tower2 = Activation(activation='relu')(tower2)
    tower2 = Conv1D(24, 3, padding='same')(tower2)
    tower2 = BatchNormalization()(tower2)
    tower2 = Activation(activation='relu')(tower2)

    tower3 = Conv1D(12, 1, padding='same')(input)
    tower3 = BatchNormalization()(tower3)
    tower3 = Activation(activation='relu')(tower3)

    tower4 = MaxPool1D(pool_size=3, strides=1, padding='same')(input)
    tower4 = Conv1D(12, 1, padding='same')(tower4)
    tower4 = BatchNormalization()(tower4)
    tower4 = Activation(activation='relu')(tower4)

    output = concatenate([tower1, tower2, tower3, tower4], axis=-1)
    return output


def inception_2d_v1(input):
    input = Input(shape=input)
    # Inception_V2
    tower1 = Conv2D(64, (5, 5), padding='same')(input)
    tower1 = BatchNormalization()(tower1)
    tower1 = Activation(activation='relu')(tower1)

    tower2 = Conv2D(64, (3, 3), padding='same')(input)
    tower2 = BatchNormalization()(tower2)
    tower2 = Activation(activation='relu')(tower2)

    tower3 = Conv2D(32, (1, 1), padding='same')(input)
    tower3 = BatchNormalization()(tower3)
    tower3 = Activation(activation='relu')(tower3)

    tower4 = MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(input)
    tower4 = Conv2D(32, (1, 1), padding='same')(tower4)
    tower4 = BatchNormalization()(tower4)
    tower4 = Activation(activation='relu')(tower4)

    output = concatenate([tower1, tower2, tower3, tower4], axis=-1)
    return output


def inception_2d_v2(input):
    input = Input(shape=input)
    # Inception_V2
    tower1 = Conv2D(64, (1, 1), padding='same')(input)
    tower1 = BatchNormalization()(tower1)
    tower1 = Activation(activation='relu')(tower1)
    tower1 = Conv2D(96, (3, 3), padding='same')(tower1)
    tower1 = BatchNormalization()(tower1)
    tower1 = Activation(activation='relu')(tower1)
    tower1 = Conv2D(96, (3, 3), padding='same')(tower1)
    tower1 = BatchNormalization()(tower1)
    tower1 = Activation(activation='relu')(tower1)

    tower2 = Conv2D(64, (1, 1), padding='same')(input)
    tower2 = BatchNormalization()(tower2)
    tower2 = Activation(activation='relu')(tower2)
    tower2 = Conv2D(64, (3, 3), padding='same')(tower2)
    tower2 = BatchNormalization()(tower2)
    tower2 = Activation(activation='relu')(tower2)

    tower3 = Conv2D(64, (1, 1), padding='same')(input)
    tower3 = BatchNormalization()(tower3)
    tower3 = Activation(activation='relu')(tower3)

    tower4 = MaxPool2D(pool_size=(3, 3), strides=1, padding='same')(input)
    tower4 = Conv2D(32, (1, 1), padding='same')(tower4)
    tower4 = BatchNormalization()(tower4)
    tower4 = Activation(activation='relu')(tower4)

    output = concatenate([tower1, tower2, tower3, tower4], axis=-1)
    return output


def inception_3d_v1(input):
    input = Input(shape=input)

    tower1 = Conv3D(24, (5, 5, 32), padding='same')(input)
    tower1 = BatchNormalization()(tower1)
    tower1 = Activation(activation='relu')(tower1)

    tower2 = Conv3D(24, (3, 3, 32), padding='same')(input)
    tower2 = BatchNormalization()(tower2)
    tower2 = Activation(activation='relu')(tower2)

    tower3 = Conv3D(12, (1, 1, 32), padding='same')(input)
    tower3 = BatchNormalization()(tower3)
    tower3 = Activation(activation='relu')(tower3)

    tower4 = MaxPool3D(pool_size=(3, 3, 32), strides=1, padding='same')(input)
    tower4 = Conv3D(12, (1, 1, 32), padding='same')(tower4)
    tower4 = BatchNormalization()(tower4)
    tower4 = Activation(activation='relu')(tower4)

    output = concatenate([tower1, tower2, tower3, tower4], axis=-1)
    return output


def inception_3d_v2(input):
    input = Input(shape=input)
    tower1 = Conv3D(12, (1, 1, 32), padding='same')(input)
    tower1 = BatchNormalization()(tower1)
    tower1 = Activation(activation='relu')(tower1)
    tower1 = Conv3D(24, (3, 3, 32), padding='same')(tower1)
    tower1 = BatchNormalization()(tower1)
    tower1 = Activation(activation='relu')(tower1)
    tower1 = Conv3D(48, (3, 3, 32), padding='same')(tower1)
    tower1 = BatchNormalization()(tower1)
    tower1 = Activation(activation='relu')(tower1)

    tower2 = Conv3D(12, (1, 1, 32), padding='same')(input)
    tower2 = BatchNormalization()(tower2)
    tower2 = Activation(activation='relu')(tower2)
    tower2 = Conv3D(24, (3, 3, 32), padding='same')(tower2)
    tower2 = BatchNormalization()(tower2)
    tower2 = Activation(activation='relu')(tower2)

    tower3 = Conv3D(12, (1, 1, 32), padding='same')(input)
    tower3 = BatchNormalization()(tower3)
    tower3 = Activation(activation='relu')(tower3)

    tower4 = MaxPool3D(pool_size=(3, 3, 32), strides=1, padding='same')(input)
    tower4 = Conv3D(12, (1, 1, 32), padding='same')(tower4)
    tower4 = BatchNormalization()(tower4)
    tower4 = Activation(activation='relu')(tower4)

    output = concatenate([tower1, tower2, tower3, tower4], axis=-1)
    return output


def feature_extractor(model1, model2):
    model1 = load_model(model1)
    model2 = load_model(model2)
    extractor_from_model1 = K.function([model1.layers[0].input, K.learning_phase()],
                                       [model1.layers[-4].output])
    extractor_from_model2 = K.function([model2.layers[0].input, K.learning_phase()],
                                       [model2.layers[-4].output])
    return extractor_from_model1, extractor_from_model2


def feature_fusion_model(model1, model2, train_samples_1, train_samples_2, train_labels, c):
    extractor_from_model1, extractor_from_model2 = feature_extractor(model1, model2)
    feature_from_model1 = extractor_from_model1([train_samples_1])[0]
    feature_from_model2 = extractor_from_model2([train_samples_2])[0]
    feature_fusion = np.concatenate([feature_from_model1, feature_from_model2], axis=1)

    input = Input(shape=(feature_fusion.shape[-1],))
    x = Dense(256, activation='relu')(input)
    x = Dense(c, activation='softmax')(x)
    model = Model(inputs=input, outputs=x)
    model.compile(optimizer=Adam(0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(feature_fusion, train_labels, batch_size=50, epochs=100, verbose=1)
    return model

# model = cnn_2d(input_shape=(19, 19, 200), c=16, lr=0.001)
# model.summary()
# print(len(model.input.shape))

# # Set callback functions to early stop training and save the best model so far
# callbacks = [EarlyStopping(monitor='val_loss', patience=2),
# ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
