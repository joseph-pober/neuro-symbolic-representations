import random

import shapes
from shapes import Shape, generate_img, generate_img_pair
import numpy as np

# colors = ["red", "blue"]
default_colors = ["white"]
default_shapes = ['c']  # ['r', 'c']
default_directions = ['l', 'r', 'u', 'd']
pos = ['l', 'c', 'r']


def generate_data(n, start=0, random_size=False, random_centre=False, colors=default_colors, shapes=default_shapes):
    total = n * len(shapes)
    for i in range(start, n):
        for shape in shapes:
            for color in colors:
                for p in pos:
                    generate_img(name=shape + "_" + p + "_" + str(i), pos=p, shape=shape, random_size=random_size,
                                 random_centre=random_centre, color=color)


def generate_shape(directory, n, min_displacement=1, directions=default_directions, random_size=False, random_centre=False,
                         colors=default_colors, shapes=default_shapes, start=0):
    for i in range(start, n):
        position = random.choice(directions)
        shape = random.choice(shapes)
        generate_img(directory=directory, name=str(i), position=position, shape=shape, random_size=random_size, random_centre=random_centre)
        
def generate_shape_interpolation(directory, shape : Shape = Shape.CIRCLE, min_displacement:int=1, position='u', custom_y = None):
    starting_y = 0
    starting_x = shapes.shape_size
    if position == 'u':
        starting_y = shapes.shape_size
    elif position == 'd':
        starting_y = shapes.total_img_size - shapes.shape_size -1
    elif position == 'c':
        starting_y = shapes.c_y
    elif position == 'custom':
        starting_y = custom_y
    n = shapes.total_img_size - (shapes.shape_size*2)
    x = starting_x
    y = starting_y
    for i in range(n):
        shapes.create_img(directory=directory, name=f"i{i}_x{x}_y{y}_{shape.name}", position_center=(x,y), shape_size=shapes.shape_size,shape=shape)
        x += 1

def generate_shape_pairs(directory, n, min_displacement=1, directions=default_directions, random_size=False, random_centre=False,
                         colors=default_colors, shapes=default_shapes, start=0):
    for i in range(start, n):
        # for shape in shapes:
            # for direction in directions:
        direction = random.choice(directions)
        shape = random.choice(shapes)
        generate_img_pair(directory=directory, name=str(i) + "." + direction, direction=direction,
                          min_displacement=min_displacement, shape=shape)


def generate_mock_data(n,s):
    training = np.random.rand(n, s*2)
    labels = np.zeros((n,)) + 0.1
    gap = 0.0
    for i in range(n):
        # if training[i][0] - training[i][1] > gap:
        #     labels[i] = 0.5#0.999999999999999999999999999
        labels[i]= training[i][0] - training[i][1]
    return training, labels
