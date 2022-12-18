import random
from enum import Enum
import numpy as np
from PIL import Image, ImageDraw

total_img_size = 28
edge = 1


# shapes = ['r', 'c']

# class Shape(Enum):

Shape = Enum("Shape", "CIRCLE TRIANGLE RECTANGLE")
# Shape = Enum("Shape", "RECTANGLE TRIANGLE")

c_x = total_img_size // 2
c_y = c_x

shape_size = total_img_size // 4 # is already 'half' i think
max_cx = total_img_size - shape_size - edge
max_cy = max_cx


min_r_w = 1
max_r_w = 5
min_r_h = 1
max_r_h = 5

min_c_w = 2
max_c_w = max_r_w
min_c_h = 2
max_c_h = max_r_h

max_r = total_img_size / 2 - 1

min_x = 1
min_y = 1
max_x = total_img_size - 1
max_y = total_img_size - 1


def draw_triangle(draw, x,y,r,color):
    draw.regular_polygon(bounding_circle=(x, y, r), n_sides=3, fill=color, rotation=0, outline=None)
    # draw.ellipse((1,1,17,14), fill="white", outline=None, width=1)
    # draw.rectangle((20, 20, 15, 30), fill='red',
    #                outline='red')  # can vary this bit to draw different shapes in different positions


def draw_rectangle(draw, x1, y1, x2, y2, color):
    draw.rectangle((x1, y1, x2, y2), fill=color)


def draw_circle(draw, x1, y1, x2, y2, color):
    # width = random.randint(min_r_w, max_r_w)
    # height = random.randint(min_r_h, max_r_h)
    draw.ellipse((x1, y1, x2, y2), fill=color)


def get_r_w_h(shape : Shape):
    if shape == Shape.RECTANGLE:
        width = random.randint(min_r_w, max_r_w)
        height = random.randint(min_r_h, max_r_h)
    elif shape == Shape.TRIANGLE:
        width = random.randint(min_r_w, max_r_w)
        height = random.randint(min_r_h, max_r_h)
    elif shape == Shape.CIRCLE:
        width = random.randint(min_c_w, max_c_w)
        height = random.randint(min_c_h, max_c_h)
    return width, height


def draw_shape(draw, shape: Shape, x1, y1, x2, y2, color, w, h, cx, cy):
    if shape == Shape.RECTANGLE:
        draw_rectangle(draw, x1, y1, x2, y2, color)
    elif shape == Shape.CIRCLE:
        draw_circle(draw, x1, y1, x2, y2, color)
    elif shape == Shape.TRIANGLE:
        draw_triangle(draw, cx, cy, h, color)


def get_bounding_box(width, height, cx, cy):
    x1 = cx - width
    y1 = cy - height
    x2 = cx + width
    y2 = cy + height
    return (x1, y1), (x2, y2)  # , (cx, cy)

def get_bounding(width, height, pos, random_centre, min_edge_distance_l=0, min_edge_distance_r=0):
    x, y = get_centre(width, height, pos, random_centre, min_edge_distance_l, min_edge_distance_r)
    return get_bounding_box(width, height, x, y)


def get_centre(shape_width, shape_height, position, random_centre, min_edge_distance_l=0, min_edge_distance_r=0, min_edge_distance_u=0,
               min_edge_distance_d=0):
    spacing_l = edge + min_edge_distance_l
    spacing_r = edge + min_edge_distance_r
    spacing_u = edge + min_edge_distance_u
    spacing_d = edge + min_edge_distance_d
    spacing = edge
    if random_centre:
        x = random.randint(shape_width + spacing_l, total_img_size - (shape_width + spacing_r))
        y = random.randint(shape_height + spacing_u, total_img_size - (shape_height + spacing_d))
    else:
        if position == 'l':
            x = spacing_l + shape_width
            y = random.randint(spacing_u + shape_height, total_img_size - (spacing_d+shape_height))
        if position == 'c':
            x = c_x
            y = c_y
        if position == 'r':
            x = total_img_size - shape_width - spacing_r
            y = random.randint(spacing_u + shape_height, total_img_size - (spacing_d+shape_height))
    return x, y


def get_w_h(shape:Shape, random_size : bool):
    if random_size:
        return get_r_w_h(shape)
    else:
        return shape_size, shape_size

def create_img(directory, name, shape:Shape, position_center, shape_size = shape_size, img_size = total_img_size, color="white"):
    image = Image.new('1', (img_size, img_size))  # binary
    # image = Image.new('RGB', (size, size)) # rgb
    draw = ImageDraw.Draw(image)
    cx, cy = position_center
    w, h = get_w_h(shape, random_size=False)
    (x1, y1), (x2, y2) = get_bounding_box(w,h,cx,cy)
    draw_shape(draw, shape, x1, y1, x2, y2, color, w,h,cx,cy)
    save_img(image, name, directory)

def generate_img(directory, name, shape : Shape, position, random_size=True, random_centre=True, color="white"):
    w, h = get_w_h(shape, random_size)
    (x1, y1), (x2, y2) = get_bounding(w, h, position, random_centre)
    
    image = Image.new('1', (total_img_size, total_img_size))  # binary
    # image = Image.new('RGB', (size, size)) # rgb
    draw = ImageDraw.Draw(image)
    draw_shape(draw, shape, x1, y1, x2, y2, color, w, h, cx, cy)
    save_img(image=image, name=name, directory=directory,position=position)
    


def generate_img_pair(directory, name, shape: Shape, direction, min_displacement, random_size=False, random_centre=True, color="white"):
    w, h = get_w_h(shape, random_size)
    distance = 0
    # if cx1 == cx2:
    #     raise ValueError
    if direction is 'l':
        cx1, cy1 = get_centre(w, h, min_edge_distance_l=min_displacement, position=None, random_centre=random_centre)
        (x11, y11), (x12, y12) = get_bounding_box(w, h, cx1, cy1)
        cy2 = cy1
        cx2 = random.randint(w + edge, cx1 - 1)
        (x21, y21), (x22, y22) = get_bounding_box(w, h, cx2, cy2)
        distance=cx1-cx2
    if direction is 'r':
        cx1, cy1 = get_centre(w, h, min_edge_distance_r=min_displacement, position=None, random_centre=random_centre)
        (x11, y11), (x12, y12) = get_bounding_box(w, h, cx1, cy1)
        cy2 = cy1
        cx2 = random.randint(cx1 + 1, max_cx)
        (x21, y21), (x22, y22) = get_bounding_box(w, h, cx2, cy2)
        distance=cx2-cx1
    if direction is 'u':
        cx1, cy1 = get_centre(w, h, min_edge_distance_u=min_displacement, position=None, random_centre=random_centre)
        (x11, y11), (x12, y12) = get_bounding_box(w, h, cx1, cy1)
        cx2 = cx1
        cy2 = random.randint(w + edge, cy1 - 1)
        (x21, y21), (x22, y22) = get_bounding_box(w, h, cx2, cy2)
        distance=cy1-cy2
    if direction is 'd':
        cx1, cy1 = get_centre(w, h, min_edge_distance_d=min_displacement, position=None, random_centre=random_centre)
        (x11, y11), (x12, y12) = get_bounding_box(w, h, cx1, cy1)
        cx2 = cx1
        cy2 = random.randint(cy1 + 1, max_cy)
        (x21, y21), (x22, y22) = get_bounding_box(w, h, cx2, cy2)
        distance=cy2-cy1

    image1 = Image.new('1', (total_img_size, total_img_size))  # binary
    draw1 = ImageDraw.Draw(image1)
    draw_shape(draw1, shape, x11, y11, x12, y12, color, w, h, cx, cy)
    save_img(image1, name, directory+"1/")

    image2 = Image.new('1', (total_img_size, total_img_size))  # binary
    draw2 = ImageDraw.Draw(image2)
    draw_shape(draw2, shape, x21, y21, x22, y22, color, w, h, cx, cy)
    save_img(image2, str(name + "." + str(distance)), directory+"2/")


def save_img(image, name, directory, position = None, format="png"):
    if position: name=f"{name}_{position}"
    image.save(f"{directory}{name}.{format}")
    print("Saved " + name + " in " + directory)


def create_random_shape():
    # shape = Shape[random.randint(0,len(Shape)-1)]
    shape = Shape(random.randint(1, 2))
    return create_shape(shape)


def create_rectangle(array, s_x, s_y):
    width = random.randint(min_r_w, max_r_w)
    height = random.randint(min_r_h, max_r_h)
    for x in range(s_x, s_x + width):
        for y in range(s_y, s_y + height):
            if x >= total_img_size or y >= total_img_size:
                continue
            array[x, y] = 1
    return array


def create_wide(array, s_x, s_y):
    width = random.randint(min_r_w, max_r_w)
    height = random.randint(1, 2)
    for x in range(s_x, s_x + width):
        for y in range(s_y, s_y + height):
            if x >= total_img_size or y >= total_img_size:
                continue
            array[x, y] = 1
    return array


def create_high(array, s_x, s_y):
    width = random.randint(1, 2)
    height = random.randint(min_r_h, max_r_h)
    for x in range(s_x, s_x + width):
        for y in range(s_y, s_y + height):
            if x >= total_img_size or y >= total_img_size:
                continue
            array[x, y] = 1
    return array


def create_circle(array, s_x, s_y):
    r = random.randint(2, max_r)
    for x in range(s_x - r, s_x + r):
        for y in range(s_y - r, s_y + r):
            if x >= total_img_size or y >= total_img_size: continue
            d = np.sqrt((x - s_x) ** 2 + (y - s_y) ** 2)
            if d > r: continue
            array[x, y] = 1
    return array


def create_triangle(array, s_x, s_y):
    height = random.randint(min_r_h, max_r_h)
    for y in range(s_y, s_y + height):
        for x in range(0, y - s_y):
            if x >= total_img_size or y >= total_img_size:
                continue
            array[x, y] = 1
    return array


def create_shape(shape):
    array = np.zeros((total_img_size, total_img_size))
    x = random.randint(min_x, max_x)
    y = random.randint(min_y, max_y)
    if shape == Shape.RECTANGLE:
        array = create_rectangle(array, x, y)
    elif shape == Shape.TRIANGLE:
        array = create_triangle(array, x, y)
    elif shape == Shape.CIRCLE:
        array = create_circle(array, x, y)
    elif shape == Shape.WIDE:
        array = create_wide(array, x, y)
    elif shape == Shape.HIGH:
        array = create_high(array, x, y)
    return array


def create(s):
    return create_shape(Shape(s))
