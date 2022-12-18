import keras
import numpy as np
from keras import layers
from keras.losses import MeanAbsoluteError
import directories as DIR

import data
from aes import Autoencoder


def run():
    # print("Loading Data...")
    x, y, file_names, n = data.load_simple(DIR.shapes_interpolated_custom, label_function=data.coordinate_label_negative,
                                           flatten_data=True)
    # print("Data Configured")
    #########
    
    # Autoencoder
    encoding_dim = 32
    ae = Autoencoder(encoding_dim=encoding_dim)
    ae.load()
    l = ae.encode(x)
    
    input = l
    
    # dummy data
    # input, y = generator.generate_mock_data(200, 1)
    
    # data info
    print("x_train shape:", input.shape)
    print(input.shape[0], "train samples")
    
    # NN
    h, w = np.shape(input)
    model = keras.Sequential()
    model.add(layers.Dense(w, input_shape=(w,), activation='tanh'))
    model.add(layers.Dense(w // 2, activation='tanh'))
    model.add(layers.Dense(2, activation='tanh'))
    opt = keras.optimizers.SGD(lr=0.01)
    model.compile(loss='mean_absolute_error', optimizer=opt,
                  metrics=[MeanAbsoluteError()])  # ,MeanAbsolutePercentageError()])#['accuracy'])#behindikindi
    model.fit(input, y, epochs=150, batch_size=1, verbose=2, validation_split=0.1)
    output_amount = 5
    print("\nInputs")
    # print(np.round(input,2)[:output_amount])
    print(file_names[:output_amount])
    print("\nLabels")
    print(np.round(y, 2)[:output_amount])
    print("\nPredictions")
    print(np.round(model.predict(input), 3)[:output_amount])
    
    # model.add(layers.Dense(100, activation='sigmoid'))
    # model.add(layers.Dense(w//2, activation='sigmoid'))
    # model.add(layers.Dense(1, activation='sigmoid'))#,activation='softmax'))
    # compile the keras model
    
    # model = keras.Sequential(
    #     [
    #         keras.Input(shape=(28,28,1)),
    #         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    #         layers.MaxPooling2D(pool_size=(2, 2)),
    #         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    #         layers.MaxPooling2D(pool_size=(2, 2)),
    #         layers.Flatten(),
    #         layers.Dropout(0.5),
    #         layers.Dense(num_classes, activation="softmax"),
    #     ]
    # )
    
    # Model / data parameters
    # num_classes = 10
    # input_shape = (28, 28, 1)
    # # Load the data and split it between train and test sets
    # (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    # # Scale images to the [0, 1] range
    # x_train = x_train.astype("float32") / 255
    # x_test = x_test.astype("float32") / 255
    # # Make sure images have shape (28, 28, 1)
    # x_train = np.expand_dims(x_train, -1)
    # x_test = np.expand_dims(x_test, -1)
    # # convert class vectors to binary class matrices
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)
    # model = keras.Sequential(
    #     [
    #         keras.Input(shape=(28,28,1)),
    #         layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    #         layers.MaxPooling2D(pool_size=(2, 2)),
    #         layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
    #         layers.MaxPooling2D(pool_size=(2, 2)),
    #         layers.Flatten(),
    #         layers.Dropout(0.5),
    #         layers.Dense(num_classes, activation="softmax"),
    #     ]
    # )
    # batch_size = 1
    # epochs = 15
    # model.compile(loss="mean_absolute_error", optimizer="adam", metrics=["accuracy"])
    # n = 200
    # # Split the data
    # x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, shuffle=True)
    # x_train = x_train[:n]
    # y_train = y_train[:n]
    # x_valid = x_valid[:n]
    # y_valid = y_valid[:n]
    # print("x_train shape:", x_train.shape)
    # print(x_train.shape[0], "train samples")
    # print(x_valid.shape[0], "test samples")
    # model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_valid,y_valid))#,validation_split=0.1)
    
    # evaluate the keras model
    # _, accuracy = model.evaluate(input, y)
    # _, accuracy = model.evaluate(x_train[:n], y_train[:n])
    # print('Accuracy: %.2f' % (accuracy * 100))
    
    # print(model.weights)
    
    # VIS
    # model.summary()
    # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # plt.plot(l,linestyle = 'dotted')
    # print(np.round(l,2))
    
    # fig, ax = plt.subplots()
    # im = ax.imshow(l)
    
    #
    # # Loop over data dimensions and create text annotations.
    # for i in range(w):
    #     for j in range(h):
    #         text = ax.text(j, i, np.round(l[i, j],1),
    #                        ha="center", va="center", color="w")
    
    # plt.show()