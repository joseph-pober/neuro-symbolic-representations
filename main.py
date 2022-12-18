import keras
import numpy as np
from keras.metrics import MeanAbsolutePercentageError, MeanAbsoluteError
from sklearn.model_selection import train_test_split

from tensorflow import keras
from tensorflow.keras import layers

import exp_propertyAE
import generator
import nn
from comparer import Comparer, DefaultComparer, DeepComparer
from aes import Autoencoder
import directories as DIR

import data

# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x, y, file_names, n = data.load_simple(DIR.shapes_interpolated_custom_mixed,
                                       label_function=data.coordinate_label_negative,
                                       flatten_data=True)

c_x_up, _, _, _ = data.load_simple(DIR.circle_interpolated_up,
                                       label_function=data.positive_label,
                                       flatten_data=True)
c_x_center, _, _, _ = data.load_simple(DIR.circle_interpolated_center,
                                       label_function=data.positive_label,
                                       flatten_data=True)
c_x_down, _, _, _ = data.load_simple(DIR.circle_interpolated_down,
                                       label_function=data.positive_label,
                                       flatten_data=True)

s_x_up, _, _, _ = data.load_simple(DIR.square_interpolated_up,label_function=data.positive_label,flatten_data=True)
s_x_center, _, _, _ = data.load_simple(DIR.square_interpolated_center, label_function=data.positive_label,flatten_data=True)
s_x_down, _, _, _ = data.load_simple(DIR.square_interpolated_down,label_function=data.positive_label,flatten_data=True)

t_x_up, _, _, _ = data.load_simple(DIR.triangle_interpolated_up,label_function=data.positive_label,flatten_data=True)
t_x_center, _, _, _ = data.load_simple(DIR.triangle_interpolated_center, label_function=data.positive_label,flatten_data=True)
t_x_down, _, _, _ = data.load_simple(DIR.triangle_interpolated_down,label_function=data.positive_label,flatten_data=True)



# ae = Autoencoder()
vis_up = [c_x_up,s_x_up,t_x_up]
vis_center = [c_x_center,s_x_center,t_x_center]
vis_down = [c_x_down,s_x_down,t_x_down]
vis_data = [vis_up,vis_center,vis_down]

vis_c = [c_x_up,c_x_center,c_x_down]
vis_s = [s_x_up,s_x_center,s_x_down]
vis_t = [t_x_up,t_x_center,t_x_down]
vis_data = [vis_c,vis_s,vis_t]
# vis_data=[x_up,x_center,x_down]
# exp_propertyAE.run(data=x,epochs=150,encoding_dim=32,vis_self=False,vis_output=True, vis_data=vis_data, loss_graph=True)

encoding_dims = 6

ae = exp_propertyAE.run(data=x, epochs=100, encoding_dim=encoding_dims,iterations=1, vis_self=True,vis_activations=False, vis_output=False, loss_graph=False, vis_data=vis_data)
# encoded = ae.encode(x)
# position_nn = nn.neural_network(input_size=encoding_dims)
# position_nn.train(x=encoded,y=y)