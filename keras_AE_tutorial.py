# Code in part taken from https://blog.keras.io/building-aautoencoderutoencoders-in-keras.html (accessed 23.7.2018)

import numpy as np
import math
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import regularizers


# from recogniser import Object#SensorObject


class Autoencoder:
    current = 0
    modulo = 200

    def __init__(self, input_dim=784, encoding_dim=32):
        self.inputDim = input_dim
        self.encoding_dim = encoding_dim
        self.inputShape = (input_dim,)
        input_img = Input(shape=self.inputShape)
        # self.encoded = Dense(encoding_dim, activation='relu',
        #                      activity_regularizer=regularizers.l1(10e-5))(input_img)
        self.encoded = Dense(encoding_dim, activation='relu')(input_img)
        self.decoded = Dense(input_dim, activation='sigmoid')(self.encoded)
        self.autoencoder = Model(input_img, self.decoded)
        self.encoder = Model(input_img, self.encoded)
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = self.autoencoder.layers[-1]
        self.decoder = Model(encoded_input, decoder_layer(encoded_input))
        # self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    def visualize(self, states):
        vis = []
        for s in states:
            e = self.encode(s)
            d = self.decode(e)
            vis.append(d)
        return vis

    def encode(self, state):
        inputData = self.convertInput(state)
        encodedData = self.encoder.predict(inputData)
        return encodedData[0]

    def decode(self, state):
        inputData = self.convertTarget(state)
        decodedData = self.decoder.predict(inputData)
        return decodedData[0]

    def fit(self, state):
        self.current += 1
        v = 0
        if self.current % self.modulo == 0:
            v = 0
        data = self.convertInput(state)
        self.autoencoder.fit(data, data, batch_size=1, epochs=1, verbose=v, shuffle=False)

    def convertInput(self, state):
        stateMatrix = np.array(state).reshape(-1, self.inputDim)  # (np.fromstring(state, np.int8) - 48).reshape(-1,25)
        return stateMatrix

    def convertTarget(self, target):
        return target.reshape(-1, self.encoding_dim)


class CAE(Autoencoder):
    activation_threshold = 0.0  # 5

    def __init__(self, input_dim, encoding_dim=10):
        self.encoding_dim = encoding_dim
        self.inputDim = int(math.sqrt(input_dim))
        self.inputShape = (self.inputDim, self.inputDim, 1)
        input_img = Input(shape=self.inputShape)
        filters1 = 2 ** 3
        # filters2 = int(filters1 * 0.5)
        #		filters3 = int(filters2 * 0.5)
        pad = 'same'
        conv_shape = (2, 2)
        x = Conv2D(filters1, conv_shape, activation='relu', padding=pad, name="encoder")(input_img)
        self.encoded = x
        self.decoded = Conv2D(1, (3, 3), activation='sigmoid', padding=pad)(x)

        self.autoencoder = Model(input_img, self.decoded)
        self.autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

    def visualize(self, states):
        vis = []
        for s in states:
            v = self.predict(s)
            vis.append(v)
        return vis

    def predict(self, state):
        s = self.convertInput(state)
        v = self.autoencoder.predict(s)
        return v.reshape(self.inputDim ** 2)

    def convertInput(self, state):
        stateMatrix = np.array(state).reshape(1, self.inputDim, self.inputDim, 1)
        return stateMatrix

    def convertTarget(self, target):
        return target.reshape(-1, self.encoding_dim)

    def getSalientRegions(self, activations):
        a = activations[0]
        features = a.shape[3]
        x = a.shape[1]
        y = a.shape[2]
        sa = np.zeros((x, y, features))

        for a in activations:
            features = a.shape[3]
            if features < 2: continue
            x = a.shape[1]
            y = a.shape[2]
            for fmIndex in range(features):
                #				print(a)
                fmap = a[0, :, :, fmIndex]  # extracts one 2d feature map
                mv = np.amax(fmap)
                normalized = fmap
                if (mv != 0): normalized = fmap / mv
                #				l = []
                #				print(fmap)
                #				print(normalized)
                for x in range(a.shape[1]):
                    for y in range(a.shape[2]):
                        val = normalized[x, y]
                        #						print(val)
                        #						print(fmap[0][x][y])
                        if (val < self.activation_threshold): fmap[x, y] = 0
                sa[:, :, fmIndex] = fmap

        return activations, sa

    def getObjects(self, salientRegions, state):
        ol = []
        s = salientRegions
        f = s.shape[2]
        for i in range(s.shape[0]):
            for j in range(s.shape[1]):
                spec = np.zeros(f)
                for k in range(f):
                    #					print((i,j,k))
                    # TODO enable again
                    pass
                #					spec[k] = s[i,j,k]
                # TODO good idea?
                #				m = np.linalg.norm(spec)
                #				if m > 1/f:
                symv = state[0, i, j, 0]
                pos = np.array([j, i])
                so = Object(spec, symv, pos)
                ol.append(so)
        return ol



# def aef(x_train, t_train, x_test=None, y_test=None):
#     validation_split = 0.1
#     model = ae.Autoencoder()
#     encoder = model.encoder
#     decoder = model.decoder
#
#     x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#
#     if not x_test:
#         x_test = x_train[int((1-validation_split) * len(x_train)):]
#     # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
#     print(x_train.shape)
#     print(x_test.shape)
#
#     model.autoencoder.fit(x_train, x_train,
#                           epochs=10,
#                           batch_size=256,
#                           shuffle=True,
#                           # validation_data=(x_test, x_test)
#                           validation_split=validation_split
#                           )
#
#     # Encode and decode some digits
#     # Note that we take them from the *test* set
#     encoded_imgs = encoder.predict(x_test)
#     decoded_imgs = decoder.predict(encoded_imgs)
#
#     n = 10  # How many digits we will display
#     plt.figure(figsize=(20, 4))
#     for i in range(n):
#         # Display original
#         ax = plt.subplot(2, n, i + 1)
#         plt.imshow(x_test[i].reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#
#         # Display reconstruction
#         ax = plt.subplot(2, n, i + 1 + n)
#         plt.imshow(decoded_imgs[i].reshape(28, 28))
#         plt.gray()
#         ax.get_xaxis().set_visible(False)
#         ax.get_yaxis().set_visible(False)
#     plt.show()