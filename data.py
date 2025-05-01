import os
from typing import List

from keras.datasets import mnist
import numpy as np
# load and show an image with Pillow
from PIL import Image
from numpy import ndarray

import generator
# from generator import shapes
from shapes import total_img_size, ShapeEnum
import directories as DIR

import re

num_channels = 1


def assemble_pairs(data1, data2, labels1, labels2):
    assert np.all(labels1 == labels2)
    # (amount, width) = np.shape(data1)
    # half_amount = int(amount / 2)
    pairs = np.concatenate((data1, data2), axis=1)
    pair_labels = labels1
    return pairs, pair_labels


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


#
# def unison_shuffled_copies(args):
#     a = args[0]
#     b = args[0]
#     assert len(a) == len(b)
#     p = np.random.permutation(len(a))
#     r = []
#     for a in args:
#         r.append(a[p])
#     return r


def randomise_data(data):
    np.random.shuffle(data)
    return data


def split_data_label(data):
    x = data[:, :-1]
    y = data[:, -1]
    return x, y


def load_mnist():
    (input_train, target_train), (input_test, target_test) = mnist.load_data()
    img_width, img_height = input_train.shape[1], input_train.shape[2]
    # Reshape data
    input_train = input_train.reshape(input_train.shape[0], img_height, img_width, num_channels)
    input_test = input_test.reshape(input_test.shape[0], img_height, img_width, num_channels)
    input_shape = (img_height, img_width, num_channels)
    # Parse numbers as floats
    input_train = input_train.astype('float32')
    input_test = input_test.astype('float32')
    # Normalize data
    input_train = input_train / 255
    input_test = input_test / 255
    return input_train, target_train


def sample_data(data, labels, target_labels, per_label):
    sampled_labels = []
    # target_labels = [0, 1, 2, 3, 4]
    label_amount = len(target_labels)
    amounts = np.zeros(label_amount)
    done = 0
    # (input_train, target_train), (input_test, target_test) = get_mnist(num_channels)
    # labels = target_train
    # data = input_train
    sampled_data = np.zeros((label_amount * per_label, data.shape[1], data.shape[2], data.shape[3]))
    # plt.figure(figsize=(10, 10))
    i = 0
    for k in range(data.shape[0]):
        l = labels[k]
        if l not in target_labels: continue
        label_pos = target_labels.index(l)
        if amounts[label_pos - 1] >= per_label: continue
        amounts[label_pos - 1] += 1
        sampled_data[i] = data[k]
        sampled_labels.append(l)
        i += 1
        if amounts[label_pos - 1] > per_label:
            done += 1
            continue
        if done >= label_amount: break
    return sampled_data, sampled_labels


def get_dir_info(d, n=None):
    names = os.listdir(d)
    total = len(names)
    if n:
        names = names[:n]
        total = n
    return names, total


def positive_label(file_names, data_count):
    return np.ones((data_count,1))

def coordinate_label_norm(file_names, data_count, norm_min =0, norm_max=1):
    # targets = np.zeros((data_count,2))
    targets = file_names[:,:2]
    # i=0
    # for name in file_names:
    #     splitByUnderscore = name.split('_')
    #     x_number = splitByUnderscore[1][1:]
    #     y_number = splitByUnderscore[2].split('.')[0][1:]
    #     targets[i][0]=x_number
    #     targets[i][1]=y_number
    #     i+=1
    targets_normalized = (norm_max - norm_min) * ((targets - np.min(targets)) / (np.max(targets) - np.min(targets))) + norm_min
    return targets_normalized

def coordinate_label_negative(file_names, data_count):
    return coordinate_label_norm(file_names, data_count, norm_min=-1, norm_max=1)

def negative_label(args):
    return np.zeros((args, 1))

def coordinate_and_shape_one_hot_label(file_names, data_count):
    coordinates = coordinate_label_norm(file_names,data_count)
    shapes_amount = len(ShapeEnum)
    shapes = np.zeros((data_count,shapes_amount)) # one hot vector for each shape
    i=0
    for name in file_names:
        s = name[2]
        id = int(s-1)
        shapes[i,id]=1
        i+=1
    labels = np.concatenate((coordinates, shapes), axis=1)
    return labels

def coordinate_and_shape_label(file_names, data_count):
    coordinates = coordinate_label_norm(file_names,data_count)
    shapes_amount = len(ShapeEnum)
    shapes = np.zeros((data_count,1)) # one hot vector for each shape
    i=0
    for name in file_names:
        s = name[2]
        id = int(s-1)
        shapes[i]=(id)/(shapes_amount-1)
        i+=1
    labels = np.concatenate((coordinates, shapes), axis=1)
    return labels
    

def categorical_label(args):
    s = args.split('_')
    label = s[0] + s[1]
    return label


def flatten(data):
    flat = data.reshape((len(data), np.prod(data.shape[1:])))
    return flat


def img_to_array(image):
    img_array = np.asarray(image, dtype='float32')
    array = np.reshape(img_array, (total_img_size, total_img_size, num_channels))
    return array


def img_to_flat_array(image):
    return flatten(img_to_array(image))


def load_imgs_from_directory(directory, n, img_to_array_function):
    names, data_count = get_dir_info(directory, n)
    train = np.zeros((data_count, total_img_size, total_img_size, num_channels))
    index = 0
    
    # SORT NAMES ACTUALLY IN ORDER
    numbers = re.compile(r'(\d+)')
    def numericalSort(value):
        parts = numbers.split(value)
        parts[1::2] = map(int, parts[1::2])
        return parts
    names=sorted(names, key=numericalSort)
    
    for name in names:
        image = Image.open(directory + name)
        resized_array = img_to_array_function(image)
        train[index] = resized_array
        index += 1
        if n and index > n:
            break
    return train, data_count, np.reshape(np.array(names), (data_count,))


def load_example_pairs(d=DIR.shapes_left_dir, label_function=positive_label):
    names, total = get_dir_info(d)
    train = np.zeros((total, total_img_size, total_img_size, num_channels))
    labels = label_function(total)
    index = 0
    # for s in shapes:
    for n in names:
        image = Image.open(d + n)
        array = np.asarray(image, dtype='float32')
        resized_array = np.reshape(array, (total_img_size, total_img_size, num_channels))
        train[index] = resized_array
        index += 1
    train = train.reshape((len(train), np.prod(train.shape[1:])))
    complete = np.concatenate((train, labels), axis=1)
    return complete  # train, labels


def process_file_names(file_names) -> ndarray:
    """
    
    :param file_names:
    :return: np array with entries of the form [x,y,shape_enum_number]
    """
    processed_names =[]
    for name in file_names:
        processed_name = []
        splitByUnderscore = name.split('_')
        if len(splitByUnderscore)<=2:
            processed_name.append(int(splitByUnderscore[-1][:-4]))
        else:
            x_number = float(splitByUnderscore[1][1:])
            y_number = float(splitByUnderscore[2][1:])
            shape = splitByUnderscore[-1].split('.')[0]
            shape = ShapeEnum[shape].value
            processed_name.append(x_number)
            processed_name.append(y_number)
            processed_name.append(shape)
            processed_names.append(processed_name)
    return np.array(processed_names)

def load_simple(directory, label_function=positive_label, img_to_array_function=img_to_array, n=None,
                flatten_data=True, shuffle=False):
    data, data_count1, file_names1 = load_imgs_from_directory(directory, n, img_to_array_function)
    
    p_names = process_file_names(file_names1)
    labels = label_function(p_names,data_count1)
    labels_amount = len(labels[0])
    if flatten_data:
        data = flatten(data)
    data_point_length = len(data[0])
    if shuffle:
        full_data = np.concatenate((data, labels), axis=1)
        np.random.shuffle(full_data)
        # data,labels = np.array_split(full_data,data_point_length-labels_amount,axis=1)
        data = full_data[:,:-labels_amount]
        labels = full_data[:,-labels_amount:]
    return data, labels, p_names, data_count1

# def load_simple_with_targets(directory, label_function=positive_label, img_to_array_function=img_to_array, n=None):
#     examples1, file_names1, data_count1 = load_simple(directory,label_function,img_to_array_function,n)
#     targets = np.zeros(data_count1)
#     i=0
#     for name in file_names1:
#         splitByUnderscore = name.split('_')
#         number = splitByUnderscore[1][1:]
#         targets[i]=number
#         i+=1
#     targets = 2*(targets-np.min(targets)/np.max(targets)-np.min(targets))-1
#     return examples1, file_names1, targets, data_count1

def load_examples(directory, label_function=positive_label, img_to_array_function=img_to_array, n=None):
    examples1, data_count1, file_names1 = load_imgs_from_directory(directory + "1/", n, img_to_array_function)
    examples2, data_count2, file_names2 = load_imgs_from_directory(directory + "2/", n, img_to_array_function)
    labels = label_function(data_count1)
    examples1 = flatten(examples1)
    examples1 = np.concatenate((examples1, labels), axis=1)
    examples2 = flatten(examples2)
    examples2 = np.concatenate((examples2, labels), axis=1)
    return examples1, examples2, file_names1, file_names2


def load_shapes(d, label_function=categorical_label):
    names = os.listdir(d)
    total = len(names)
    # total = len([name for name in os.listdir(dir) if os.path.isfile(name)])
    train = np.zeros((total, total_img_size, total_img_size, num_channels))
    labels = []
    index = 0
    # for s in shapes:
    for n in names:
        s = n.split('_')
        label = s[0] + s[1]
        image = Image.open(d + n)
        array = np.asarray(image, dtype='float32')
        resized_array = np.reshape(array, (total_img_size, total_img_size, num_channels))
        train[index] = resized_array
        labels.append(label)
        index += 1
    return train, labels

# Viz data
# for i in range(label_amount*per_label):
#     plt.subplot(label_amount,per_label,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(data[i], cmap=plt.cm.binary)
#     plt.xlabel(labels[i])
# plt.show()


### VIS z
# mk = 10
# mp = 6
# plts, axs = plt.subplots(mp, mk)
# for k in range(mk):
#     a, (m, s, z) = sl.get_sensory_data(k)
#     Z = AE.sample_z((m, s))
#     p = [decoder.predict(m),
#          # decoder.predict(s),
#          decoder.predict(z),
#          decoder.predict(Z.numpy())]
#     axs[0, k].imshow(np.reshape(data[k:k + 1], (28, 28)))
#     for i in range(len(p)):
#         axs[i + 1, k].imshow(np.reshape(p[i], (28, 28)))
#     # axs[mp-1, k].imshow(np.reshape(a, (20, 1)))
# plt.show()


### Printing
# a = get_activations(encoder, data[:1], layer_name='activation_pattern')
# print(a)
# display_activations(a, cmap="gray", save=False)
