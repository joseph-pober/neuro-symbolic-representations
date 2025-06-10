

import keras
import numpy as np
from keras import Input, Model
from keras.layers import Dense, concatenate, Conv2D, MaxPooling2D, Flatten
import tensorflow as tf

import directories

use_bias = True


class SplitNN:
    def __init__(self, body_size, assemblies_amount, input_shape, num_classes):
        self.assemblies_amount = assemblies_amount
        self.inputTensor = Input(shape=input_shape, name="input")
        self.body_size = body_size
        self.a_size = int(self.body_size / assemblies_amount)
        self.name = "split-nn_"+str(assemblies_amount)+"_"+str(body_size)

        # Conv
        self.conv1 = Conv2D(3, kernel_size=(3, 3), activation="relu")(self.inputTensor)
        self.pool1 = MaxPooling2D(pool_size=(2, 2))(self.conv1)
        self.conv2 = Conv2D(6, kernel_size=(3, 3), activation="relu")(self.pool1)
        self.pool2 = MaxPooling2D(pool_size=(2, 2))(self.conv2)
        self.flat = Flatten()(self.pool2)
        # self.convolution_model = Model(self.inputTensor, self.flat,name="Convolution")

        # Preprocessing
        self.pp = Dense(self.body_size, activation='relu', use_bias=use_bias)(self.flat)
        # self.preprocessing_model = Model(self.flat, self.pp,name="Preprocessing")

        # Assemblies
        self.assemblies = []
        self.last_layer=None
        for i in range(assemblies_amount):
            assembly = Dense(self.a_size, activation='relu', use_bias=use_bias)(self.pp)
            self.last_layer = assembly
            self.assemblies.append(assembly)
        # self.g2 = Dense(self.input_size, activation='sigmoid', use_bias=use_bias)(self.inputTensor)
        # self.r1 = Dense(self.input_size, activation='sigmoid', use_bias=use_bias)(self.g1)
        # self.r2 = Dense(self.input_size, activation='sigmoid', use_bias=use_bias)(self.g2)

        # Ending
        if assemblies_amount > 1:
            self.o = concatenate(self.assemblies)
            self.last_layer = self.o
        self.outputTensor = Dense(num_classes, activation="softmax")(self.last_layer)

        self.total_model = Model(self.inputTensor, self.outputTensor)

        # Setup
        self.fit_history = None
        self.compile()

    def save_history(self):
        np.save(self.name+'_history.npy', self.fit_history.history)

    def compile(self):
        self.total_model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    def train(self, x_train, y_train, batch_size, epochs):
        self.fit_history = self.total_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
        return self.fit_history

    def vis_self(self):
        # VIS
        self.total_model.summary()
        tf.keras.utils.plot_model(self.total_model, f"{directories.model_dir}{self.name}.png", show_shapes=True, show_layer_names=True,expand_nested=True)
        tf.keras.utils.model_to_dot(self.total_model)
        
    def plot(self):
        keras.utils.plot_model(self.total_model, to_file=(self.name + '_model.png'), show_shapes=True,
                               show_layer_names=True)
