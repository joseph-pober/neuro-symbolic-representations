import shapes
from generator import generate_data, generate_shape_pairs, generate_shape, generate_shape_interpolation
import directories as d


n = 10
start = 0

# for i in range(shapes.shape_size, shapes.total_img_size-shapes.shape_size):
# 	generate_shape_interpolation(directory=d.shapes_interpolated_custom_mixed,position='custom',custom_y=i,shape=shapes.Shape.TRIANGLE) # range from 7 to 20, 7 = -1, 14 = 0, 20 = 1


shape = shapes.ShapeEnum.RECTANGLE
generate_shape_interpolation(directory=d.square_interpolated_up,position='u', shape=shape)
generate_shape_interpolation(directory=d.square_interpolated_down,position='d', shape=shape)
generate_shape_interpolation(directory=d.square_interpolated_center,position='c', shape=shape)

# generate_shape(directory=d.shapes_left_dir,n=n,directions=['l'],shapes=['c'])
# generate_shape(directory=d.shapes_right_dir,n=n,directions=['r'],shapes=['c'])

# generate_data(n, start, random_size=False, random_centre=True, shapes=['c'])

# generate_shape_pairs(directory=d.negative_dir, n=n*3, directions=['r', 'u', 'd'], min_displacement=1)
# generate_shape_pairs(directory=d.shapes_left_dir, n=n, directions=['l'], min_displacement=1)
