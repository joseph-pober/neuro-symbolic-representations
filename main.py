from enum import Enum

from h5py.h5pl import size
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

# DATA LOADING
x, y, file_names, data_amount = data.load_simple(DIR.shapes_interpolated_custom_mixed_old,#DIR.square_and_circle_interpolated_custom_mixed,#
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

# data set construction
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
#################################



# RUNNING PARAMETERS
# experiment_name = "property_visualisation4"  # testing PAEb
# experiment_name = "property_visualisation3" # currently testing ground
# experiment_name = "property_visualisation2" # this one has a different learning rate for PAE, currently either 0.005 or 0.001
# experiment_name = "property_visualisation"
# experiment_name = "2shapes"
experiment_name = "integrated"

class RunMode(Enum):
    TRAIN = 1
    LOAD = 2
    NOTHING = 3
# overall_run_mode:RunMode=RunMode.TRAIN
# overall_run_mode:RunMode=RunMode.LOAD
overall_run_mode:RunMode=RunMode.NOTHING
if overall_run_mode == RunMode.NOTHING:
    # ae_run_mode : RunMode = RunMode.TRAIN
    # ae_run_mode : RunMode = RunMode.LOAD
    ae_run_mode : RunMode=RunMode.NOTHING

    pae_run_mode : RunMode = RunMode.TRAIN
    # pae_run_mode : RunMode = RunMode.LOAD

    data_run_mode = RunMode.TRAIN
    # data_run_mode = RunMode.LOAD
else:
    ae_run_mode: RunMode = overall_run_mode
    pae_run_mode: RunMode = overall_run_mode
    data_run_mode = overall_run_mode

ae_graph = False
pae_graph = False

ae_vis = False
###################


# MODEL PARAMAETERS
# autoencoder
ae_epochs = 150
ae_encoding_dim = 32
ae_input_dim=784
ae_lr = 0.01
ae_reg_term=10e-5
ae_validation_split = 0.1

# property autoencoder
pae_epochs=100
pae_property_dim=3
pae_input_dim=32
pae_lr = 0.005#0.005 #0.01
pae_validation_split = 0.1

# data
parameter_exploration_steps = 2
parameter_exploration_step_size = parameter_exploration_steps - 1
###################


# MODELS
# autoencoder
ae_full_name = f"AE.{experiment_name}_{ae_input_dim}-{ae_encoding_dim}-{ae_epochs}"
ae = aes.Autoencoder(name=ae_full_name,  input_dim=ae_input_dim, encoding_dim=ae_encoding_dim, lr=ae_lr)

# property autoencoder
pae_name = f"PAE.{experiment_name}_{pae_input_dim}-{pae_property_dim}-{pae_epochs}"
# pae : exp_propertyAE.PAEa = exp_propertyAE.PAEa(
#     name=pae_name, input_dim=pae_input_dim, property_dim=pae_property_dim, lr=pae_lr)
# pae : exp_propertyAE.PAEb = exp_propertyAE.PAEb(
#     name=pae_name, input_dim=pae_input_dim, property_dim=pae_property_dim, lr=pae_lr)
pae : exp_propertyAE.PAEintegrated = exp_propertyAE.PAEintegrated(
    name=pae_name, features_dim=32,img_dim=784, lr=pae_lr)
########


###########
# PREPROCESSING via regular AUTOENCODER
# ae_name = "AE.3shapes-reg"
# ae_name = "AE.shapes3a"

# REGULARISATION
# ae_reg_mode=regularizers.l1
# ae = aes.AutoencoderRegularised(name=ae_full_name,  input_dim=ae_input_dim, encoding_dim=ae_encoding_dim, lr=ae_lr,
#                                 reg_mode=ae_reg_mode,reg_term=ae_reg_term)

# RUN
ae_data = x#exp_2shapes_data #
# train
if ae_run_mode == RunMode.TRAIN:
    ae.fit(ae_data,epochs=ae_epochs,graph=ae_graph,validation_split=ae_validation_split)
    ae.save()
    # VIS
    if ae_vis:
        ae.vis_output(ae_data)
        ae.vis_self()
# load
elif ae_run_mode == RunMode.LOAD:
    ae.load()
elif ae_run_mode == RunMode.NOTHING:
    pass
    
# Create feature vectors
if ae_run_mode != RunMode.NOTHING:
    encode_data = ae_data #exp_c_data # usually x (the full data)
    z = ae.encode(encode_data)
else:
    z = ae_data
# PROPERTY NETWORK
# pae_reg_term = 10e-5
# version="3shapesB"
# version="shapes3a"
# pae : exp_propertyAE.PAEb = exp_propertyAE.PAEregularised(name=pae_name,input_dim=input_dim, property_dim=property_dim,
#                                                           lr=pae_lr, reg_term = pae_reg_term)

# RUN
# train
if pae_run_mode == RunMode.TRAIN:
    pae.fit(x_train=z, validation_split=pae_validation_split, epochs=pae_epochs, graph=pae_graph)
    pae.save()
# load
elif pae_run_mode == RunMode.LOAD:
    pae.load()
    
# ENCODINGS and DECODINGS
# manual encoding experiment 2
npa0 = np.array([0])
npa1 = np.array([0.25])
npa2 = np.array([0.5])
npa3 = np.array([0.75])
npa4 = np.array([1])
npa = [npa0,npa1,npa2,npa3,npa4]

if data_run_mode == RunMode.TRAIN:
    all_encodings = []
    encoding = []
    for k in range(parameter_exploration_steps):
        # encodings = []
        for i in range(parameter_exploration_steps):
            for j in range(parameter_exploration_steps):
                index = (k * parameter_exploration_steps ** 2) + (i * parameter_exploration_steps) + j
                encoding = [(1 / parameter_exploration_step_size) * j, (1 / parameter_exploration_step_size) * i, (1 / parameter_exploration_step_size) * k]
                # encodings.append(encoding)
                np.savetxt(f"{DIR.experiment_data_dir}{experiment_name}/encodings/{index}.txt",encoding)
                all_encodings.append(encoding)
        
    decodings=[]
    imgs = []
    for k in range(parameter_exploration_steps):
        for i in range(parameter_exploration_steps):
            # encodings = all_encodings[i]
            for j in  range(parameter_exploration_steps):#range(len(encodings)):
                index= (k * parameter_exploration_steps ** 2) + (i * parameter_exploration_steps) + j
                e= all_encodings[index]
                encoding = [
                    np.array([e[0]]),
                    np.array([e[1]]),
                    np.array([e[2]])
                ]
                decoding = pae.decode(encoding)
                decodings.append(decoding)
                img = ae.decode(decoding)
                imgs.append(img)
                np.savetxt(f"{DIR.experiment_data_dir}{experiment_name}/decodings/{index}.txt",decoding)
                np.savetxt(f"{DIR.experiment_data_dir}{experiment_name}/imgs/{index}.txt",img)

# if data_mode == RunMode.LOAD:
total_encodings = []
total_imgs = []
for k in range(parameter_exploration_steps):
    # imgs=[]
    # encodings=[]
    for i in range(parameter_exploration_steps): # n x n images loaded
        for j in range(parameter_exploration_steps):
            index= (k * (parameter_exploration_steps ** 2)) + (i * parameter_exploration_steps) + j
            total_imgs.append(np.loadtxt(f"{DIR.experiment_data_dir}{experiment_name}/imgs/{index}.txt"))
            total_encodings.append(np.loadtxt(f"{DIR.experiment_data_dir}{experiment_name}/encodings/{index}.txt"))
    # total_encodings.append(encodings)
    # total_imgs.append(imgs)
pae.vis_nice_images_load(imgs=total_imgs,encodings=total_encodings)


class ExperimentType(Enum):
    NOTHING = -1
    TEST = 0
    PROPERTIES = 1
    PROPERTIES2 = 2
    DIFFERENT_PROPETIES=3
    VIS_META = 4
    VIS_OUTPUTS = 5
    MANUAL_ENCODINGS = 6
    NICE_IMAGES = 7
# experiment_mode : ExperimentType = ExperimentType.NOTHING
# experiment_mode : ExperimentType = ExperimentType.TEST
# experiment_mode : ExperimentType = ExperimentType.PROPERTIES
# experiment_mode : ExperimentType = ExperimentType.PROPERTIES2
# experiment_mode : ExperimentType = ExperimentType.DIFFERENT_PROPETIES
# experiment_mode : ExperimentType = ExperimentType.VIS_META
# experiment_mode : ExperimentType = ExperimentType.VIS_OUTPUTS
# experiment_mode : ExperimentType = ExperimentType.MANUAL_ENCODINGS
experiment_mode : ExperimentType = ExperimentType.NOTHING

match experiment_mode:
    case ExperimentType.NICE_IMAGES:
        parameter_exploration_steps = 8
        parameter_exploration_step_size = parameter_exploration_steps - 1
        exp_encodings = []
        for l in range(parameter_exploration_steps):
            for i in range(parameter_exploration_steps):
                for j in range(parameter_exploration_steps):
                    exp_encodings.append([
                        np.array([(1 / parameter_exploration_step_size) * l]),
                        np.array([(1 / parameter_exploration_step_size) * i]),
                        np.array([(1 / parameter_exploration_step_size) * j])
                    ])
            pae.vis_nice_images(encodings=exp_encodings, autoencoder=ae)
            exp_encodings = []
    
    case ExperimentType.MANUAL_ENCODINGS:
        exp_encodings = [
            [
                np.array([0.43]), np.array([0.71]), np.array([0.43])
            ],
            [np.array([0.43]),np.array([0.43]),np.array([0.57])],
        ]
        # exp_encodings = np.array(
        #     [
        #         np.array([1,1,1]),
        #         np.array([0,0,0])
        #     ]
        # ).T
        # exp_encodings = np.array(
        #             [
        #                 np.array([0.43]), np.array([0.71]), np.array([0.43])
        #             ]
        # )
        # exp_encodings = np.zeros((3,1))
        # exp_encodings[0]=1
        # exp_encodings[1]=1
        # exp_encodings[2]=1
        pae.vis_nice_images(encodings=exp_encodings, autoencoder=ae)
        
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
        
            pae.vis_nice_images(encodings=exp_encodings, autoencoder=ae)
            
    case ExperimentType.PROPERTIES2:
        parameter_exploration_steps = 8
        parameter_exploration_step_size = parameter_exploration_steps - 1
        for k in range(parameter_exploration_steps):
            # exp_encodings = []
            # exp_encoding1 = np.zeros((n,n))
            # exp_encoding2 = np.zeros((n,n))
            # exp_encoding3 = np.zeros((n,n))
            exp_encodings = np.zeros((parameter_exploration_steps, parameter_exploration_steps, 3))#[exp_encoding1,exp_encoding2,exp_encoding3]
            for i in range(parameter_exploration_steps):
                for j in range(parameter_exploration_steps):
                    # exp_encoding1[i,j]= np.array([(1/m)*k])
                    # exp_encoding2[i,j]= np.array([(1/m)*i])
                    # exp_encoding3[i,j]= np.array([(1/m)*j])
                    exp_encodings[i,j,:] = (1 / parameter_exploration_step_size) * k, (1 / parameter_exploration_step_size) * i, (1 / parameter_exploration_step_size) * j
                    # exp_encodings.append([np.array([(1/m)*k]),np.array([(1/m)*i]),np.array([(1/m)*j])])
        
            pae.vis_encodings(encodings=exp_encodings, autoencoder=ae)
    case ExperimentType.DIFFERENT_PROPETIES:
        parameter_exploration_steps = 8
        parameter_exploration_step_size = parameter_exploration_steps - 1
        exp_encodings = []
        for l in range(parameter_exploration_steps):
            for i in range(parameter_exploration_steps):
                for j in range(parameter_exploration_steps):
                    exp_encodings.append([
                        np.array([(1 / parameter_exploration_step_size) * l]),
                        np.array([(1 / parameter_exploration_step_size) * i]),
                        np.array([(1 / parameter_exploration_step_size) * j])
                    ])
            pae.vis_encodings(encodings=exp_encodings, autoencoder=ae)
            exp_encodings = []
    case ExperimentType.VIS_META:
        parameter_exploration_steps = 3
        parameter_exploration_step_size = parameter_exploration_steps - 1
        
        topfig =plt.figure()
        topfig.tight_layout()
        figs = topfig.subfigures(parameter_exploration_steps, parameter_exploration_steps)
        
        for i in range(parameter_exploration_steps):
            for j in range(parameter_exploration_steps):
                small_axes = figs[i,j].subplots(parameter_exploration_steps, parameter_exploration_steps)
                
                for k in range(parameter_exploration_steps):
                    for l in range(parameter_exploration_steps):
                        ax = small_axes[k,l]
                        encodings = [
                            np.atleast_2d(
                                [(1 / parameter_exploration_step_size) * i, (1 / parameter_exploration_step_size) * j]
                            ),
                            np.atleast_2d([(1 / parameter_exploration_step_size) * k]),
                            np.atleast_2d([(1 / parameter_exploration_step_size) * l])
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
        
        parameter_exploration_steps = 10
        
        random_indicies = np.random.randint(0, len(z), parameter_exploration_steps)
        rx=x[random_indicies]
        rz=z[random_indicies]
        pae_encodings=pae.encode(rz)
        pae_reconstruction=pae.decode(pae_encodings)
        img_reconstruction = ae.decode(pae_reconstruction)
        
        axes_imgs = figs[0].subplots(1, parameter_exploration_steps)
        axes_reconstructions = figs[1].subplots(1, parameter_exploration_steps)
        
        decimals = 2
        # rounded=np.round(pae_encodings, decimals)
        
        for i in range(parameter_exploration_steps):
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