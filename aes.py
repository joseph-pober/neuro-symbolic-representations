import keras
import numpy as np
import math
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras import regularizers
import matplotlib.pyplot as plt
from keras.optimizers import Adam

import directories


class Autoencoder:
    default_name = "autoencoder"
    current = 0
    modulo = 200
    model_name = "total_model"
    decoder_name = "decoder"
    encoder_name = "encoder"
    
    def save(self):
        self.total_model.save(directories.model_dir + self.name + "-" + self.model_name)
        self.decoder.save(directories.model_dir + self.name + "-" + self.decoder_name)
        self.encoder.save(directories.model_dir + self.name + "-" + self.encoder_name)

    def load(self, name=default_name):
        self.total_model = keras.models.load_model(directories.model_dir + name + "-" + self.model_name)
        self.decoder = keras.models.load_model(directories.model_dir + name + "-" + self.decoder_name)
        self.encoder = keras.models.load_model(directories.model_dir + name + "-" + self.encoder_name)
        self.compile()

    def compile(self):
        # COMPILE
        opt = Adam(lr=self.lr)
        self.total_model.compile(optimizer=opt, loss='binary_crossentropy')
    
    # def run(self):

    def __init__(self, name=default_name, input_dim=784, encoding_dim=32, lr=0.01):
        self.name = name
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.input_shape = (input_dim,)
        self.encoding_shape = (self.encoding_dim,)
        self.lr = lr
        
        self.encoder_activation = "sigmoid"
        self.code_activation = "sigmoid"
        self.decoder_activation = "sigmoid"
        self.decoder_output_activation = "sigmoid"
    
        # ENCODER
        self.encoder_input = Input(shape=self.input_shape, name="original_img")
        # h = Dense(256, activation='sigmoid')(self.encoder_input)
        h = Dense(128, activation=self.encoder_activation)(self.encoder_input)
        h = Dense(64, activation=self.encoder_activation)(h)
        self.encoder_output = Dense(encoding_dim, activation=self.code_activation,name="code",
                                    # activity_regularizer=regularizers.l1(0.0001)
                                    )(h)
        self.encoder = Model(self.encoder_input, self.encoder_output,name="Encoder")
    
        # DECODER
        self.decoder_input = Input(shape=self.encoding_shape, name="decoder_input")
        h = Dense(64, activation=self.decoder_activation)(self.decoder_input)
        h = Dense(128, activation=self.decoder_activation)(h)
        # h = Dense(256, activation='sigmoid')(h)
        self.decoder_output = Dense(self.input_dim, activation=self.decoder_output_activation)(h)
        self.decoder = Model(self.decoder_input, self.decoder_output, name="Decoder")
    
        # AUTOENCODER
        self.autoencoder_input = Input(shape=self.input_shape, name="autoencoder_input")  # self.encoder_input#
        encoded_img = self.encoder(self.autoencoder_input)
        decoded_img = self.decoder(encoded_img)
        self.total_model = Model(self.autoencoder_input, decoded_img, name="autoencoder")
        
        #COMPILE
        self.compile()

    def fit(self, x_train, validation_split=0.1, epochs=100,graph=False):
        history = self.total_model.fit(x_train, x_train,
                                       epochs=epochs,
                                       batch_size=16,
                                       shuffle=True,
                                       validation_split=validation_split,
                                       verbose=2
                                       )
        if not graph:
            return history
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper right')
        plt.show()
    
        return history
    
    def predict(self,data):
        predictions = self.total_model.predict(data)
        return predictions

    def encode(self, data):
        encodings = self.encoder.predict(data)
        return encodings
    
    def decode(self, data):
        decodings = self.decoder.predict(data)
        return decodings

    def vis_self(self):
        # VIS
        self.total_model.summary()
        keras.utils.plot_model(self.total_model, f"{directories.model_dir}{self.name}.png", show_shapes=True, show_layer_names=True,
                               expand_nested=True)
        keras.utils.model_to_dot(self.total_model)

    def vis_activations(self, data):
        fig, axs = plt.subplots(3, 3)
        # fig.colorbar(mappable=cm.ScalarMappable(norm=Normalize(), cmap='coolwarm'), ax=None)
        j = 0
        for shape in data:
            for i in range(len(shape)):
                id = i + (j * 3)
                d = shape[i]
                encoded = self.encode(d)
            
                # h,w = np.shape(encoded)
                # ax = plt.subplot(1,9,i+1+(j*3))
                im = axs[j][i].imshow(encoded, cmap='viridis', norm=None, vmin=0, vmax=1)
                # Loop over data dimensions and create text annotations.
                # for i in range(w):
                # 	for j in range(h):
                # 		text = ax.text(j, i, np.round(encoded[i, j],1),
                # 					   ha="center", va="center", color="w")
                title = ["Up", "Center", "Down"][i] + ["Circle", "Square", "Triangle"][j]
                axs[j][i].set_title(title)
            j = j + 1
        fig.tight_layout()
        plt.show()
        
    def vis_ouput_from_encoding(self, imgs, encodings, n=10):
        decoded_imgs = self.decode(encodings)
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Display original
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(imgs[i].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        
            # Display reconstruction
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(decoded_imgs[i].reshape(28, 28))
            plt.gray()
            img_str = str(np.round(encodings[i], 2))
            # print(img_str)
            # ax.set_title(img_str)
            ax.text(-30, (45 + (20 * i % 2)) * (-1 * i % 2), img_str, bbox=dict(facecolor='red', alpha=0.5))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

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
            # print(img_str)
            # ax.set_title(img_str)
            ax.text(-30, (45 + (20 * i % 2)) * (-1 * i % 2), img_str, bbox=dict(facecolor='red', alpha=0.5))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

    def vis_output_old(self, data, n=10):  # How many digits we will display
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
