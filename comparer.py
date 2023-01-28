import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Lambda, Dense, concatenate
from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import TensorBoard

import matplotlib.pyplot as plt

import data
from vis import heatmap

# log_dir=os.path.join("C:/Diverses/Uni/PhD/subsym/data", 'tensorboard')
log_dir = os.path.join('.\logs', 'tensorboard')
tensorboard = TensorBoard(
    log_dir=log_dir, histogram_freq=0, write_graph=True,
    write_grads=False, write_images=False, embeddings_freq=0,
    embeddings_layer_names=None, embeddings_metadata=None,
    embeddings_data=None, update_freq='epoch'
)
keras_callbacks = [
    tensorboard
]

use_bias = False


# class Linear(keras.layers.Layer):
#     def __init__(self, id, units=32, input_dim=32):
#         super(Linear, self).__init__()
#         w_init = tf.random_normal_initializer()
#         self.w = tf.Variable(
#             initial_value=w_init(shape=(input_dim, units), dtype="float32"),
#             trainable=True,
#         )
#         b_init = tf.zeros_initializer()
#         self.b = tf.Variable(
#             initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
#         )
#
#     def call(self, inputs):
#         return tf.matmul(inputs, self.w) + self.b


class DefaultComparer:
    def __init__(self, single_input_size):
        self.input_size = single_input_size
        self.total_input_size = single_input_size * 2

        self.inputTensor = Input(shape=(self.total_input_size,), name="input")
        self.h1 = Dense(self.input_size * 2, activation='sigmoid', use_bias=use_bias, name="h1")(self.inputTensor)
        self.output = Dense(1, activation='sigmoid', use_bias=use_bias)(self.h1)
        self.model = Model(inputs=self.inputTensor, outputs=self.output)
        self.model.compile(optimizer='adam', loss='binary_crossentropy')

    def fit(self, x, y, epochs=100, validation_split=0.1):
        self.fit_history = self.model.fit(x, y, batch_size=10, epochs=epochs,
                                          shuffle=True,
                                          # validation_data=(x_test, x_test)
                                          validation_split=validation_split,
                                          callbacks=keras_callbacks
                                          )
        return self.fit_history

    def get_weights(self):
        W = []
        for layer in self.model.layers:
            w = layer.get_weights()
            W.append(np.round(w, 2))
        return W

    def vis_history(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # Plot history: MAE
        ax1.plot(self.fit_history.history['loss'], label='MAE (training data)')
        ax1.plot(self.fit_history.history['val_loss'], label='MAE (validation data)')
        ax1.set_title('MAE for Chennai Reservoir Levels')
        ax1.set_ylabel('MAE value')
        ax1.set_xlabel('No. epoch')
        ax1.legend(loc="upper left")

        # summarize history for accuracy
        # ax2.plot(self.fit_history.history['accuracy'])
        # ax2.plot(self.fit_history.history['val_accuracy'])
        # ax2.title('model accuracy')
        # ax2.ylabel('accuracy')
        # ax2.xlabel('epoch')
        # ax2.legend(['train', 'test'], loc='upper left')
        #
        # # summarize history for loss
        # ax3.plot(self.fit_history.history['loss'])
        # ax3.plot(self.fit_history.history['val_loss'])
        # ax3.title('model loss')
        # ax3.ylabel('loss')
        # ax3.xlabel('epoch')
        # ax3.legend(['train', 'test'], loc='upper left')

        fig.show()

    def vis(self, imgs1, imgs2, y, ae, names, n=10):
        enc1 = ae.encode(imgs1)
        enc2 = ae.encode(imgs2)
        pe, py = data.assemble_pairs(enc1, enc2, y, y)
        predictions = self.model.predict(pe)
        # decoded_imgs = self.decoder.predict(encoded_imgs)
        p = np.random.permutation(len(pe))

        plt.figure(figsize=(20, 4))
        plt.tight_layout()
        # i = 0
        for j in range(n):
            index = p[j]
            # Display original
            # img_index = index*2
            ax = plt.subplot(2, n, j + 1)
            title = str(predictions[index]) + "/" + str(py[index])
            ax.set_title(title)
            plt.imshow(imgs1[index].reshape(28, 28))
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # i += 1
            ax = plt.subplot(2, n, j + 1 + n)
            # ax.set_title(names[i-1]+"|"+names[i])
            ax.set_title(names[index])
            # ax.set_xlabel(names[index+1])
            plt.imshow(imgs2[index].reshape(28, 28))
            plt.gray()
            # ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # i += 1
        plt.show()

    def plot_model(self):
        self.model.summary()
        tf.keras.utils.plot_model(self.model, to_file='models/model_plot.png', show_shapes=True, show_layer_names=True)


class Comparer(DefaultComparer):

    def __init__(self, single_input_size):
        self.input_size = int(single_input_size)
        self.total_input_size = single_input_size * 2

        self.inputTensor = Input(shape=(self.total_input_size,), name="input")
        self.groups, self.rs = self.make_groups()
        # self.rs = self.make_r()
        self.fit_history = None

        if len(self.rs) < 2:
            # self.rs = self.rs[0]
            self.hidden = self.rs[0]
        else:
            self.hidden = concatenate(self.rs)
        # self.hidden = concatenate(self.groups)
        self.outputTensor = Dense(1, activation='sigmoid', name="output", use_bias=use_bias)(self.hidden)
        self.model = Model(self.inputTensor, self.outputTensor)
        self.compile()

    def make_groups(self):
        groups = []
        rs = [1] * self.input_size
        for i in range(self.input_size):
            g = Lambda(lambda x: x[:, (i * 2):((i + 1) * 2)], name="g" + str(i + 1), output_shape=((2,)))(
                self.inputTensor)
            # g = Lambda(lambda x: x[:, (i * 2):(i * 2)+1], output_shape=((1,2)), name="g" + str(i + 1))(self.inputTensor)
            # g = Lambda(lambda x: x[:, [i, self.input_size+i]], output_shape=((2,)), name="g" + str(i + 1))(self.inputTensor)
            # g = Lambda(lambda x: tf.gather(x, [i, self.input_size+i], axis=1), name="g" + str(i + 1))(
            #     self.inputTensor)
            r11 = Dense(1, activation="sigmoid", name="r1" + str(i + 1), use_bias=use_bias)(g)  # (self.inputTensor)#
            rs[i] = r11
            # groups.append(g)
        return groups, rs

    def make_r(self):
        i = 0
        rs = [1] * self.total_input_size
        for g in self.groups:
            r11 = Dense(1, activation="sigmoid", use_bias=False, name="r1" + str(i + 1))(g)
            r21 = Dense(1, activation="sigmoid", use_bias=False, name="r2" + str(i + 1))(g)
            rs[i] = r11
            rs[i + self.input_size] = r21
            i += 1
        return rs

    def compile(self):
        self.model.compile(optimizer='adam', loss='binary_crossentropy') #MAYBE use KERAS complier so it can be saved

    def get_w(self, m):
        s, i, j = 6, 0, 0
        W = np.zeros((s, s))
        for layer in m.layers:
            name = layer.name
            w = layer.get_weights()
            # print(name)
            if not w: continue
            w = w[0][:]
            num_rows, num_cols = w.shape
            # print(w.shape)
            for c in range(num_cols):
                v = w[:, c]
                # print(v)
                l = len(v)
                # print(np.shape(w))
                # print(np.shape(w[:,0]))
                # print(np.shape(W[i:l,j]))
                W[i:i + l, j] = v
                i += l
                if i >= s:
                    i = 0
                    j += 1
                # print(W)
        return W.copy()

    def vis_model(self, x, y):
        W1 = self.get_w(self.model)
        print(W1)
        self.model.fit(x, y,
                       # batch_size=10,
                       epochs=100)
        # print(W1)
        W2 = self.get_w(self.model)
        # print(W1)
        # print(W2)
        heatmap([W2])
        # heatmap(W1)
        # plt.show()
        # heatmap(W2)
        # plt.show()

        # print(history.history)
        # print(layer.get_config(), layer.get_weights())


class DeepComparer(Comparer):
    def __init__(self, single_input_size):
        self.input_size = int(single_input_size)
        self.total_input_size = single_input_size * 2

        self.inputTensor = Input(shape=(self.total_input_size,), name="input")
        self.g1 = Dense(self.input_size, activation='sigmoid', use_bias=use_bias)(self.inputTensor)
        self.g2 = Dense(self.input_size, activation='sigmoid', use_bias=use_bias)(self.inputTensor)
        self.r1 = Dense(self.input_size, activation='sigmoid', use_bias=use_bias)(self.g1)
        self.r2 = Dense(self.input_size, activation='sigmoid', use_bias=use_bias)(self.g2)
        # self.o1 = Dense(1, activation='sigmoid', use_bias=use_bias)(self.r1)
        # self.o2 = Dense(1, activation='sigmoid', use_bias=use_bias)(self.r2)
        self.o = concatenate([self.r1, self.r2])
        self.outputTensor = Dense(1, activation='sigmoid', name="output", use_bias=use_bias)(self.o)
        self.model = Model(self.inputTensor, self.outputTensor)
        # self.model.compile(optimizer='adam', loss='binary_crossentropy')
        # self.rs = self.make_r()
        self.fit_history = None

        self.compile()

### OLD
# group1 = Lambda(lambda x: x[:, :2], output_shape=((2,)), name="g1")(inputTensor)
#       # group2 = Lambda(lambda x: x[:, 2:4], output_shape=((2,)), name="g2")(inputTensor)
#       # group3 = Lambda(lambda x: x[:, 4:6], output_shape=((2,)), name="g3")(inputTensor)
#       #
#       # r11 = Dense(1, use_bias=False, name="r11")(group1)
#       # r21 = Dense(1, use_bias=False, name="r21")(group1)
#       #
#       # r12 = Dense(1, use_bias=False, name="r12")(group2)
#       # r22 = Dense(1, use_bias=False, name="r22")(group2)
#       #
#       # r13 = Dense(1, use_bias=False, name="r13")(group3)
#       # r23 = Dense(1, use_bias=False, name="r23")(group3)
#
#       # outputTensor = concatenate([r11, r12, r13, r21, r22, r23])
#       # outputTensor = Dense(2)(outputTensor)
#
#       # h1_out = Dense(1, activation='sigmoid')(inp2)  # only connected to the second neuron
#       # h2_out = Dense(1, activation='sigmoid')(inp)  # connected to both neurons
#       # h_out = concatenate([h1_out, h2_out])
#
#       # out = Dense(2, activation='sigmoid')(h_out)
#
#       self.outputTensor = concatenate(self.rs)
#       self.outputTensor = Dense(1, activation='sigmoid')(self.outputTensor)
#       self.model = Model(self.inputTensor, self.outputTensor)
#       self.compile()
#       # self.model.compile(optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
#       #                    loss=keras.losses.BinaryCrossentropy()
#       #                    )
#       # loss=keras.losses.CategoricalCrossentropy())
#       # metrics=["sparse_categorical_accuracy"])
#       # print(model.summary())
