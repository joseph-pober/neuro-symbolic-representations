from enum import Enum

from keras import regularizers
from matplotlib import pyplot as plt
import keras
import numpy as np
# from keras.metrics import MeanAbsolutePercentageError, MeanAbsoluteError
from sklearn.model_selection import train_test_split
#
# from tensorflow import keras
# from tensorflow.keras import layers

import aes
import exp_propertyAE
import generator
import nn
# from comparer import Comparer, DefaultComparer, DeepComparer
from aes import Autoencoder
import directories as DIR

import data

# Just disables the warning, doesn't take advantage of AVX/FMA to run faster
import os

from split_nn import SplitNN

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class RunMode(Enum):
    TRAIN = 1
    LOAD = 2

x, y, file_names, n = data.load_simple(DIR.shapes_interpolated_custom_mixed_old,#DIR.square_and_circle_interpolated_custom_mixed,#
                                       label_function=data.coordinate_and_shape_label,
                                       flatten_data=True, shuffle=True)


c_x_up, _, _, _ = data.load_simple(DIR.circle_interpolated_up,
                                       label_function=data.positive_label,
                                       flatten_data=True, shuffle=True)
c_x_center, _, _, _ = data.load_simple(DIR.circle_interpolated_center,
                                       label_function=data.positive_label,
                                       flatten_data=True, shuffle=True)
c_x_down, _, _, _ = data.load_simple(DIR.circle_interpolated_down,
                                       label_function=data.positive_label,
                                       flatten_data=True, shuffle=True)

s_x_up, _, _, _ = data.load_simple(DIR.square_interpolated_up,label_function=data.positive_label,flatten_data=True, shuffle=True)
s_x_center, _, _, _ = data.load_simple(DIR.square_interpolated_center, label_function=data.positive_label,flatten_data=True, shuffle=True)
s_x_down, _, _, _ = data.load_simple(DIR.square_interpolated_down,label_function=data.positive_label,flatten_data=True, shuffle=True)

t_x_up, _, _, _ = data.load_simple(DIR.triangle_interpolated_up,label_function=data.positive_label,flatten_data=True, shuffle=True)
t_x_center, _, _, _ = data.load_simple(DIR.triangle_interpolated_center, label_function=data.positive_label,flatten_data=True, shuffle=True)
t_x_down, _, _, _ = data.load_simple(DIR.triangle_interpolated_down,label_function=data.positive_label,flatten_data=True, shuffle=True)

vis_up =    [c_x_up,s_x_up,t_x_up]
vis_center =[c_x_center,s_x_center,t_x_center]
vis_down =  [c_x_down,s_x_down,t_x_down]
vis_data_pos = [vis_up,vis_center,vis_down]

vis_c = [c_x_up,c_x_center,c_x_down]
vis_s = [s_x_up,s_x_center,s_x_down]
vis_t = [t_x_up,t_x_center,t_x_down]
vis_data_shape = [vis_c,vis_s,vis_t]


vis_data =[
    [c_x_up,c_x_center,c_x_down],
    [s_x_up,s_x_center,s_x_down],
    [t_x_up,t_x_center,t_x_down]
]

# DATA FOR EXPERIMENT (C)
exp_c_data = np.concatenate([s_x_up,t_x_up,c_x_center,s_x_center,t_x_center,c_x_down,s_x_down,t_x_down])

# DATA FOR EXPERIMENT 2SHAPES (square and triangle)
exp_2shapes_data, _, _, _  = data.load_simple(DIR.square_and_triangle_interpolated, label_function=data.coordinate_and_shape_label,
                                       flatten_data=True, shuffle=True)

# PREPROCESSING via regular AUTOENCODER
ae_name = "AE.3shapes-reg"
ae_epochs = 150
ae_encoding_dim = 32
ae_input_dim=784
ae_lr = 0.01
ae_reg_term=10e-5
ae_reg_mode=regularizers.l1
ae_full_name = f"{ae_name}_{ae_input_dim}-{ae_encoding_dim}-{ae_epochs}"
ae = aes.AutoencoderRegularised(name=ae_full_name,  input_dim=ae_input_dim, encoding_dim=ae_encoding_dim, lr=ae_lr,
                                reg_mode=ae_reg_mode,reg_term=ae_reg_term)

# RUN
ae_data = x#exp_2shapes_data #
# ae_run_mode : RunMode = RunMode.TRAIN
ae_run_mode : RunMode = RunMode.LOAD
# train
if ae_run_mode == RunMode.TRAIN:
    ae.fit(ae_data,epochs=ae_epochs,graph=True)
    ae.save()
    # VIS
    ae.vis_output(ae_data)
    ae.vis_self()
# load
elif ae_run_mode == RunMode.LOAD:
    ae.load()
    

# Create feature vectors
encode_data = ae_data #exp_c_data # usually x (the full data)
z = ae.encode(encode_data)

# PROPERTY NETWORK
input_dim=32
property_dim=3
epochs=150
pae_lr = 0.01
pae_reg_term = 10e-5
# version="p" #"b"#"c"
# version = "differentP"
version="3shapesB"
pae_name = f"PAE.{version}_{input_dim}-{property_dim}-{epochs}"
pae : exp_propertyAE.PAEa = exp_propertyAE.PAEa(name=pae_name,input_dim=input_dim, property_dim=property_dim, lr=0.01)
# pae : exp_propertyAE.PAEb = exp_propertyAE.PAEregularised(name=pae_name,input_dim=input_dim, property_dim=property_dim,
#                                                           lr=pae_lr, reg_term = pae_reg_term)

# RUN
# mode : RunMode = RunMode.TRAIN
mode : RunMode = RunMode.LOAD
# train
if mode == RunMode.TRAIN:
    pae.fit(x_train=z,validation_split=0.1,epochs=epochs,graph=True)
    pae.save()
# load
elif mode == RunMode.LOAD:
    pae.load()
    
# ENCODINGS
# manual encoding experiment 2
npa0 = np.array([0])
npa1 = np.array([0.25])
npa2 = np.array([0.5])
npa3 = np.array([0.75])
npa4 = np.array([1])
npa = [npa0,npa1,npa2,npa3,npa4]

# EXPERIMENT
class ExperimentType(Enum):
    NOTHING = -1
    TEST = 0
    PROPERTIES = 1
    PROPERTIES2 = 2
    DIFFERENT_PROPETIES=3
    VIS_META = 4
    VIS_OUTPUTS = 5
    MANUAL_ENCODINGS = 6
# experiment_mode : ExperimentType = ExperimentType.NOTHING
# experiment_mode : ExperimentType = ExperimentType.TEST
experiment_mode : ExperimentType = ExperimentType.PROPERTIES2
# experiment_mode : ExperimentType = ExperimentType.DIFFERENT_PROPETIES
# experiment_mode : ExperimentType = ExperimentType.VIS_META
# experiment_mode : ExperimentType = ExperimentType.VIS_OUTPUTS
# experiment_mode : ExperimentType = ExperimentType.MANUAL_ENCODINGS

match experiment_mode:
    case ExperimentType.MANUAL_ENCODINGS:
        exp_encodings = [
            [
                np.array([0.43]), np.array([0.71]), np.array([0.43])
            ],
            [np.array([0.43]),np.array([0.43]),np.array([0.57])],
        ]
        
        pae.vis_encodings(encodings=exp_encodings, autoencoder=ae)
        
    case ExperimentType.TEST:
        # encodings = pae.encode(z)
        # pae.img_from_encoding(autoencoder=ae,encodings=encodings,n=10)
        pae.img(images=ae_data, ae_z=z, ae=ae, n=10)
    case ExperimentType.PROPERTIES:
        for k in npa:
            exp_encodings = []
            for i in npa:
                for j in npa:
                    exp_encodings.append([k,j,i])
        
            pae.vis_encodings(encodings=exp_encodings, autoencoder=ae)
    case ExperimentType.PROPERTIES2:
        n = 8
        m = n -1
        for k in range(n):
            exp_encodings = []
            # exp_encoding1 = np.zeros((n,n))
            # exp_encoding2 = np.zeros((n,n))
            # exp_encoding3 = np.zeros((n,n))
            # exp_encodings = [exp_encoding1,exp_encoding2,exp_encoding3]
            for i in range(n):
                for j in range(n):
                    # exp_encoding1[i,j]= np.array([(1/m)*k])
                    # exp_encoding2[i,j]= np.array([(1/m)*i])
                    # exp_encoding3[i,j]= np.array([(1/m)*j])
                    # exp_encodings[i,j,:] = (1/m)*k, (1/m)*i, (1/m)*j
                    exp_encodings.append([np.array([(1/m)*k]),np.array([(1/m)*i]),np.array([(1/m)*j])])
        
            pae.vis_encodings(encodings=exp_encodings, autoencoder=ae)
    case ExperimentType.DIFFERENT_PROPETIES:
        n = 8
        m = n -1
        exp_encodings = []
        for l in range(n):
            for i in range(n):
                for j in range(n):
                    exp_encodings.append([
                        np.array([(1/m)*l]),
                        np.array([(1/m)*i]),
                        np.array([(1/m)*j])
                    ])
            pae.vis_encodings(encodings=exp_encodings, autoencoder=ae)
            exp_encodings = []
    case ExperimentType.VIS_META:
        n = 3
        m = n - 1
        
        topfig =plt.figure()
        topfig.tight_layout()
        figs = topfig.subfigures(n,n)
        
        for i in range(n):
            for j in range(n):
                small_axes = figs[i,j].subplots(n, n)
                
                for k in range(n):
                    for l in range(n):
                        ax = small_axes[k,l]
                        encodings = [
                            np.atleast_2d(
                                [(1 / m) * i, (1 / m) * j]
                            ),
                            np.atleast_2d([(1 / m) * k]),
                            np.atleast_2d([(1 / m) * l])
                        ]
                        decodings = pae.decode(encodings, batchsize=1)
                        img = ae.decode(decodings)
                        img = img.reshape(28, 28)
                        ax.imshow(img)
                        element1 = encodings[0][0]
                        element2 = encodings[1][0]
                        element3 = encodings[2][0]
                        str1 = f"[{np.round(element1[0], 2)}, {np.round(element1[1], 2)}],"
                        str2 = f"{np.round(element2, 2)},"
                        str3 = f"{np.round(element3, 2)}"
                        ax.set_title(str1 + str2 + str3, fontsize=8)
                        ax.get_xaxis().set_visible(False)
                        ax.get_yaxis().set_visible(False)
        plt.gray()
        plt.show()
        
    case ExperimentType.VIS_OUTPUTS:
        topfig =plt.figure()
        # topfig.tight_layout()
        figs=topfig.subfigures(2,1)
        
        n = 10
        
        random_indicies = np.random.randint(0,len(z),n)
        rx=x[random_indicies]
        rz=z[random_indicies]
        pae_encodings=pae.encode(rz)
        pae_reconstruction=pae.decode(pae_encodings)
        img_reconstruction = ae.decode(pae_reconstruction)
        
        axes_imgs = figs[0].subplots(1,n)
        axes_reconstructions = figs[1].subplots(1,n)
        
        decimals = 2
        # rounded=np.round(pae_encodings, decimals)
        
        for i in range(n):
            img = rx[i].reshape(28, 28)
            axes_imgs[i].imshow(img)
            
            img2=img_reconstruction[i].reshape(28,28)
            axes_reconstructions[i].imshow(img2)
            
            rounded = pae_encodings
            element1a = str(round(rounded[0][i][0],decimals))
            element1b = str(round(rounded[0][i][1],decimals))
            element2 = str(round(rounded[1][i][0], decimals))
            element3 = str(round(rounded[2][i][0], decimals))
            # element1 = np.round(rounded[0][i], decimals)
            # element2 = np.round(rounded[1][i], decimals)
            # element3 = np.round(rounded[2][i], decimals)
            str1 = "["+element1a+", "+element1b+"], "
            str2 = element2+", "
            str3 = element3
            ts=str1 + str2 + str3
            axes_reconstructions[i].set_title(str1 + str2 + str3, fontsize=8)
            print(ts)
        
        plt.show()
        
                        
                        
# manual encoding experiment 1
# manual_encodings_exp = [[npa4,npa0,npa0],[npa4,npa1,npa1],[npa4,npa2,npa2],[npa4,npa3,npa3],[npa4,npa4,npa4]]
# manual_encodings0 = [[npa0,npa0,npa0],[npa1,npa0,npa0],[npa2,npa0,npa0],[npa3,npa0,npa0],[npa4,npa0,npa0]]# shape [s,c,t,t,t] # y
# manual_encodings1 = [[npa0,npa0,npa0],[npa0,npa1,npa0],[npa0,npa2,npa0],[npa0,npa3,npa0],[npa0,npa4,npa0]]# x # shape ? + y?
# manual_encodings2 = [[npa0,npa0,npa0],[npa0,npa0,npa1],[npa0,npa0,npa2],[npa0,npa0,npa3],[npa0,npa0,npa4]]# y # x + shape?
# for manual_encodings in [manual_encodings0,manual_encodings1,manual_encodings2]:
#     pae.img_from_encoding(encodings=manual_encodings, autoencoder=ae)

# visualisation
# pae.vis_self()



# SPLIT NN for experiments
# snn = SplitNN(body_size=3, assemblies_amount=3, input_shape=z.shape, num_classes=3)

# SMALL AUTOENCODER for testing / checking
# aes = aes.AutoencoderSmall()
# aes.vis_self()
# aes.fit(z,epochs=15,graph=False)
# aes.vis_activations(vis_data, ae)

# UNSUPERVISED
# pae = exp_propertyAE.PAEa()
# pae.vis_self()
# pae.fit(z, epochs=100, graph=True)
# pae.vis_activations(vis_data, ae)

# SUPERVISED
# z = ae.encode(x)
# pae_s = exp_propertyAE.PAEa_supervised()
# pae_s.vis_self()
# pae_s.fit(z, y, epochs=100, graph=True)
# pae_s.vis_activations(vis_data, ae)

# encodings = pae.predict(z)
# pae.vis_ouput_from_encoding(imgs=x,encodings=encodings,n=10)


# ae = exp_propertyAE.run(data=x, epochs=100, encoding_dim=encoding_dims,iterations=1, vis_self=True,vis_activations=False, vis_output=False, loss_graph=False, vis_data=vis_data)
# encoded = ae.encode(x)
# position_nn = nn.neural_network(input_size=encoding_dims)
# position_nn.train(x=encoded,y=y)