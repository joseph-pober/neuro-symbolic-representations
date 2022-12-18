import keras
import numpy as np
import math
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt
import directories

ae_name = "ae"
decoder_name = "d"
encoder_name = "e"


class Autoencoder:
    current = 0
    modulo = 200

    def save(self):
        self.autoencoder.save(directories.model_dir + self.name + "-" + ae_name)
        self.decoder.save(directories.model_dir + self.name + "-" + decoder_name)
        self.encoder.save(directories.model_dir + self.name + "-" + encoder_name)

    def load(self, name="autoencoder"):
        self.autoencoder = keras.models.load_model(directories.model_dir + name + "-" + ae_name)
        self.decoder = keras.models.load_model(directories.model_dir + name + "-" + decoder_name)
        self.encoder = keras.models.load_model(directories.model_dir + name + "-" + encoder_name)
        self.compile()

    def compile(self):
        # self.autoencoder.compile(optimizer='adadelta',
        #                          loss='binary_crossentropy')  # TODO use KERAS complier so it can be saved
        # self.autoencoder.compile(
        #     optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
        #     loss=keras.losses.BinaryCrossentropy()
        # )
        self.autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    def __init__(self, name="autoencoder", input_dim=784, encoding_dim=32):
        self.name = name
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.inputShape = (input_dim,)
        self.input_img = Input(shape=self.inputShape)

        self.encoded = Dense(128, activation='sigmoid')(self.input_img)
        # self.encoded = Dense(self.encoding_dim, activity_regularizer=regularizers.l1(10e-5), activation='relu')(self.encoded)
        self.encoded = Dense(self.encoding_dim, activation='sigmoid')(self.encoded)

        self.decoded = Dense(128, activation='sigmoid')(self.encoded)
        self.decoded = Dense(self.input_dim, activation='sigmoid')(self.encoded)

        self.autoencoder = Model(self.input_img, self.decoded)
        self.encoder = Model(self.input_img, self.encoded)
        self.encoded_input = Input(shape=(self.encoding_dim,))
        self.decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(self.encoded_input, self.decoder_layer(self.encoded_input))
        # self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.compile()

    def fit(self, x_train, validation_split=0.1, epochs=100):
        self.autoencoder.fit(x_train, x_train,
                             epochs=epochs,
                             batch_size=256,
                             shuffle=True,
                             # validation_data=(x_test, x_test)
                             validation_split=validation_split
                             )

    def encode(self, data):
        encodings = self.encoder.predict(data)
        return encodings

    def vis_output(self, data, n=10):  # How many digits we will display
        encoded_imgs = self.encode(data)
        decoded_imgs = self.decoder.predict(encoded_imgs)

        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(data[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            img_str = str(np.round(encoded_imgs[i], 2))
            print(img_str)
            # ax.set_title(img_str)
            ax.text(-30, (45+(20*i%2))*(-1*i%2), img_str, bbox=dict(facecolor='red', alpha=0.5))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
