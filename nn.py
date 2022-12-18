import keras
import numpy as np
from keras import layers
from keras.losses import MeanAbsoluteError


class neural_network:
    def __init__(self,input_size):
        w = input_size
        self.model = keras.Sequential()
        self.model.add(layers.Dense(w, input_shape=(w,), activation='tanh'))
        self.model.add(layers.Dense(w // 2, activation='tanh'))
        self.model.add(layers.Dense(2, activation='tanh'))
        opt = keras.optimizers.SGD(lr=0.01)
        self.model.compile(loss='mean_absolute_error', optimizer=opt,
                      metrics=[MeanAbsoluteError()])  # ,MeanAbsolutePercentageError()])#['accuracy'])#behindikindi
    
    def train(self,x, y):
        self.model.fit(x, y, epochs=150, batch_size=16, verbose=2, validation_split=0.1, shuffle=True)
        output_amount = 5
        # print("\nInputs")
        # print(np.round(input,2)[:output_amount])
        # print(file_names[:output_amount])
        print("\nLabels")
        print(np.round(y, 2)[:output_amount])
        print("\nPredictions")
        print(np.round(self.model.predict(x), 3)[:output_amount])

    def vis_self(self):
        # VIS
        self.model.summary()
        keras.utils.plot_model(self.model, "position_nn.png", show_shapes=True, show_layer_names=True,
                               expand_nested=True)
        # keras.utils.plot_model(self.model, "propertyAE_compact.png", show_shapes=True, show_layer_names=True,
        #                        expand_nested=False)
        keras.utils.model_to_dot(self.model)