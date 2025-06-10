import shapes
from generator import generate_shape_interpolation
import directories as d


# n = 10
# start = 0

# make circles and squares, without triangles
# sc=[shapes.ShapeEnum.CIRCLE, shapes.ShapeEnum.RECTANGLE]

# make squares and triangles
st=[shapes.ShapeEnum.TRIANGLE, shapes.ShapeEnum.RECTANGLE]
for s in st:
	for i in range(shapes.shape_size, shapes.total_img_size-shapes.shape_size):
		generate_shape_interpolation(directory=d.square_and_triangle_interpolated,position='custom',custom_y=i,shape=s) # range from 7 to 20, 7 = -1, 14 = 0, 20 = 1


# for i in range(shapes.shape_size, shapes.total_img_size-shapes.shape_size):
# 	generate_shape_interpolation(directory=d.shapes_interpolated_custom_mixed,position='custom',custom_y=i,shape=shapes.Shape.TRIANGLE) # range from 7 to 20, 7 = -1, 14 = 0, 20 = 1

# MAKE NEW EXTRA TRIANGLES
# shape = shapes.ShapeEnum.TRIANGLE
# for i in range(5):
# 	generate_shape_interpolation(directory=d.triangle_interpolated_test, position='custom',custom_y=20+i, shape=shape)

# generate_shape_interpolation(directory=d.square_interpolated_up,position='u', shape=shape)
# generate_shape_interpolation(directory=d.square_interpolated_down,position='d', shape=shape)
# generate_shape_interpolation(directory=d.square_interpolated_center,position='c', shape=shape)

# generate_shape(directory=d.shapes_left_dir,n=n,directions=['l'],shapes=['c'])
# generate_shape(directory=d.shapes_right_dir,n=n,directions=['r'],shapes=['c'])

# generate_data(n, start, random_size=False, random_centre=True, shapes=['c'])

# generate_shape_pairs(directory=d.negative_dir, n=n*3, directions=['r', 'u', 'd'], min_displacement=1)
# generate_shape_pairs(directory=d.shapes_left_dir, n=n, directions=['l'], min_displacement=1)
