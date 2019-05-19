from python_gdal import *
from models_keras import *
from keras import backend as K

MAIN_FOLDER = r'E:/HSI/'
PAVIA_DATA_PATH = "Pavia/Pavia"
PAVIA_TRAIN_PATH = "Pavia/Pavia_gt"
PAVIA_U_DATA_PATH = "Pavia/PaviaU"
PAVIA_U_TRAIN_PATH = "Pavia/PaviaU_gt"

c = 9
lists = [200, 200, 200, 200, 200, 200, 200, 200, 200]
image_shape = [(1096, 715), (610, 340)]
DATA_PATH = [PAVIA_U_DATA_PATH, PAVIA_DATA_PATH]
TRAIN_PATH = [PAVIA_U_TRAIN_PATH, PAVIA_TRAIN_PATH]
features_fusion_path = r'E:/HSI/code/predicts_mat/features_fusion/'
model_path = r"e:/HSI/code/new_model_3/"
classification_report_path = r"e:/HSI/code/classification_report/"
svg_path = r'C:/Users/Lenovo/Desktop/author/svg/'
data = ['PU', 'P']
# i = 0
# # i = 1
# m = np.arange(5, 42, 4)
for i in range(1, 2):
    for m in np.arange(37, 42, 4):
        print(i, m)
        # # MODEL AND FEATURE PRE
        cnn_1d_path = r"e:/HSI/code/new_model_0/"
        cnn_2d_path = r"e:/HSI/code/new_model_2/"

        # print(cnn_2d_path + '{}-cnn_2d_PU.h5'.format(m))
        cnn_1d_model = load_model(cnn_1d_path + "cnn_1d_{}.h5".format(data[i]))
        cnn_2d_model = load_model(cnn_2d_path + "{}-cnn_2d_{}.h5".format(m, data[i]))
        print("MODEL LOADED SUCCESS!!!!!")

        # # FEATURE EXTRACTOR PRE
        cnn_1d_feature_extractor = K.function([cnn_1d_model.layers[0].input, K.learning_phase()],
                                              [cnn_1d_model.layers[-4].output])
        cnn_2d_feature_extractor = K.function([cnn_2d_model.layers[0].input, K.learning_phase()],
                                              [cnn_2d_model.layers[-4].output])

        # # DATA PREP
        # # 2D CNN DATA
        train_samples_2, train_labels = get_train_sample(data_path=MAIN_FOLDER+DATA_PATH[i],
                                                         train_data_path=MAIN_FOLDER+TRAIN_PATH[i],
                                                         c=c, m=m, lists=lists, d=4)
        # # 1D CNN DATA
        n = int((m-1)/2)
        train_samples_1 = train_samples_2[:, n, n, :]
        train_samples_1 = np.expand_dims(train_samples_1, axis=-1)

        # # ONE-HOT FOR LABELS
        train_labels = one_hot_encode(c=c, labels=train_labels)

        # # GENERATE FEATURES RESPECTIVELY
        features_1d = cnn_1d_feature_extractor([train_samples_1])[0]
        features_2d = cnn_2d_feature_extractor([train_samples_2])[0]

        # # FEATURES CONCATENATE
        fusion_layer = np.concatenate([features_1d, features_2d], axis=1)

        # # CONSTRUCT SIMPLE CLASSIFIER FOR TRAIN
        inputs = Input(shape=(fusion_layer.shape[-1],))
        y = Dense(128, activation='relu')(inputs)
        y = Dense(128, activation='relu')(y)
        output = Dense(c, activation='softmax')(y)
        model = Model(inputs=inputs, outputs=output)
        model.compile(optimizer=Adam(0.0001), loss="categorical_crossentropy", metrics=["accuracy"])
        # TRAIN MODEL
        network = model.fit(fusion_layer, train_labels, batch_size=50, epochs=50, verbose=0)

        # # SAVE MODEL OR NOT
        save_model(model, model_path + '{}_cnn_1d_2d_ff_{}.h5'.format(m, data[i]))
        print('MODEL SAVED')

        # # PREDICTIONS ON TEST SETS FOR GETTING OA AND KAPPA
        # GET TEST SETS
        bands_data, is_train, training_labels = get_prep_data(data_path=MAIN_FOLDER+DATA_PATH[i],
                                                              train_data_path=MAIN_FOLDER+TRAIN_PATH[i])
        _, x_test_index, _, y_test = custom_train_index(is_train, training_labels, c=c,
                                                        lists=lists)
        # # GET 1D DATA FROM BANDS DATA
        samples = []
        for j in x_test_index:
            sample = bands_data[j[0], j[1]]
            samples.append(sample)
        samples = np.stack(samples)
        samples = samples.reshape((samples.shape[0], samples.shape[1], -1))

        # features_1 = cnn_1d_feature_extractor([samples])[0]
        # del samples

        # # GET 2D DATA FROM BANDS DATA ! CAUTION RAM!!!
        samples_1 = []
        predictions_test = []
        x_test_nindex = x_test_index + n
        bands_data_ = np.pad(bands_data, ((n, n), (n, n), (0, 0)), 'constant', constant_values=0)
        for k, j in enumerate(x_test_nindex):
            k1 = j[0] - n
            k2 = j[0] + n + 1
            k3 = j[1] - n
            k4 = j[1] + n + 1
            block = bands_data_[k1:k2, k3:k4]
            samples_1.append(block)
            if len(samples_1) == 715 or k == x_test_nindex.shape[0] - 1:
                # print("Batches Features...")
                pre = np.stack(samples_1)
                features_2 = cnn_2d_feature_extractor([pre])[0]
                samples_1 = []
                if samples.shape[0] < 715:
                    features_1 = cnn_1d_feature_extractor([samples[:]])[0]
                else:
                    features_1 = cnn_1d_feature_extractor([samples[:715]])[0]
                    samples = samples[715:]
                features_test = np.concatenate([features_1, features_2], axis=1)
                predictions_test_800 = model.predict(features_test)
                predictions_test.append(predictions_test_800)

        # features_2 = np.concatenate(features)
        # del features

        # # CONCATENATE 1D AND 2D FEATURES
        # features_test = np.concatenate([features_1, features_2], axis=1)

        # # PREDICTION ON TEST
        # predictions_test = model.predict(features_test)
        predictions_test = np.concatenate(predictions_test, axis=0)

        OA, KAPPA = print_plot_cm(y_test, predictions_test)
        plt.savefig(svg_path + "{}cnn_1d_2d_ff_{}_{:.4f}_{:.4f}.svg".format(m, data[i], OA, KAPPA))
        print("TEST PREDICTIONS FINISHED!!!!")
        del predictions_test

        # # GET WHOLE IMAGES PREDICTIONS AND PROBABILITY
        # # GET IMAGES BANDS DATA AND ALREADY
        # 1D DATA OF WHOLE IMAGES
        samples_1d = bands_data.reshape((bands_data.shape[0]*bands_data.shape[1], bands_data.shape[2], -1))
        # features_1d_whole = cnn_1d_feature_extractor([samples_1d])[0]
        print(bands_data.shape)
        # # 2D DATA OF WHOLE IMAGES
        bands_data_1 = np.pad(bands_data, ((n, n), (n, n), (0, 0)), 'constant', constant_values=0)
        cols = bands_data_1.shape[1]-2*n
        rows = bands_data_1.shape[0]-2*n
        result1 = []
        predictions_whole = []
        for g in range(0, rows, 1):
            for h in range(0, cols, 1):
                data1 = bands_data_1[g: g + m, h: h + m, :]
                result1.append(data1)
                if len(result1) == 715:
                    # print("Batches Features...")
                    pre1 = np.stack(result1)
                    features_2d_715 = cnn_2d_feature_extractor([pre1])[0]
                    print(features_2d_715.shape)
                    result1 = []
                    features_1d_715 = cnn_1d_feature_extractor([samples_1d[:715]])[0]
                    print(features_1d_715.shape)
                    samples_1d = samples_1d[715:]
                    features_part_images = np.concatenate([features_1d_715, features_2d_715], axis=1)
                    predictions_part = model.predict(features_part_images)
                    predictions_whole.append(predictions_part)
        print('BATCHES PREDICTION FINISHED!!!')
        # features_2d_whole = np.concatenate(f2)
        # del f2
        # # CONCATENATE FEATURES 1D AND 2D FROM WHOLE IMAGES
        # features_whole_images = np.concatenate([features_1d_whole, features_2d_whole], axis=1)
        # print(features_whole_images.shape)
        predictions_whole = np.concatenate(predictions_whole, axis=0)

        # # PREDICTIONS ON WHOLE FEATURES
        # predictions_whole = model.predict(features_whole_images)
        del model

        # # SAVE PREDICTIONS .MAT FILES
        save_array_to_mat(predictions_whole, features_fusion_path + '{}-cnn_1d_2d_fusion_{}.h5'.format(m, data[i]))
        print('{}-cnn_1d_2d_fusion_{}'.format(m, data[i]) + ' SAVE FINISHED!!!!')
        del predictions_whole


