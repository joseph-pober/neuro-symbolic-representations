import math

import keras
import numpy as np
from keras import Input, Model, regularizers
from keras.layers import Dense, Concatenate, concatenate
from keras.optimizers import SGD, Adam
from matplotlib import pyplot as plt, cm
from matplotlib.pyplot import title

import directories as DIR
from matplotlib.colors import Normalize
import mpl_toolkits
from aes import Autoencoder



class PAEa(Autoencoder):
	default_name = "propertyAE_a"
	
	def compile(self):
		opt = Adam(lr=self.lr)
		self.total_model.compile(optimizer=opt, loss='mean_squared_error')
	
	def __init__old(self, name=default_name, input_dim=32, property_dim=3, lr=0.01):
		self.name = name
		self.input_dim = input_dim
		# self.encoding_dim = property_dim * 2
		self.input_shape = (input_dim,)
		# self.encoding_shape = (self.encoding_dim,)
		self.lr = lr
		# self.property_output_dim = self.input_dim//2
		
		# ENCODER
		p1_dims = 1  # property_dim
		p2_dims = 1  # property_dim
		p3_dims = 1  # property_dim
		self.encoder_input = Input(shape=self.input_shape, name="original_code")
		# properties
		p_activation = 'sigmoid'
		code_activation = 'sigmoid'
		# p1 = Dense(code_dim, activation=p_activation, name="p1_0")(self.encoder_input)
		p1 = Dense(self.input_dim // 2, activation=p_activation, name="p1_1")(self.encoder_input)
		p1_code = Dense(p1_dims, activation=code_activation, name="p1_code",
		                # activity_regularizer=regularizers.l1(0.001)
		                )(p1)
		p1 = Dense(self.input_dim // 2, activation=p_activation, name="p1_2")(p1_code)
		p1_output = Dense(self.input_dim, activation=p_activation, name="p1_3")(p1)
		# p1_output = Dense(code_dim, activation=p_activation, name="p1_4")(p1)
		
		# p2 = Dense(code_dim, activation=p_activation, name="p2_0")(self.encoder_input)
		p2 = Dense(self.input_dim // 2, activation=p_activation, name="p2_1")(self.encoder_input)
		p2_code = Dense(p2_dims, activation=p_activation, name="p2_code")(p2)
		p2 = Dense(self.input_dim // 2, activation=p_activation, name="p2_2")(p2_code)
		p2_output = Dense(self.input_dim, activation=p_activation, name="p2_3")(p2)
		# p2_output = Dense(code_dim, activation=p_activation, name="p2_4")(p2)
		
		p3 = Dense(self.input_dim // 2, activation=p_activation, name="p3_1")(self.encoder_input)
		p3_code = Dense(p3_dims, activation=p_activation, name="p3_code")(p3)
		p3 = Dense(self.input_dim // 2, activation=p_activation, name="p3_2")(p3_code)
		p3_output = Dense(self.input_dim, activation=p_activation, name="p3_3")(p3)
		
		ps = [p1_output, p2_output, p3_output]
		codes = [p1_code, p2_code, p3_code]
		self.encoder_output = concatenate(ps)
		encoder_m = Model(self.encoder_input, outputs=self.encoder_output, name="encoder_m")
		self.encoder_model = Model(self.encoder_input, outputs=codes, name="Encoder")
		
		# DECODER
		self.concat_input = Input(shape=(input_dim * 3,), name="decoder_input")
		# h = Dense(self.input_dim, activation=p_activation, name="d1_1")(self.decoder_input)
		# h = Dense(64, activation='sigmoid')(self.decoder_input)
		# h = Dense(128, activation='sigmoid')(h)
		self.concat_output = Dense(self.input_dim, name="output", activation=p_activation)(self.concat_input)
		self.concat_model = Model(self.concat_input, self.concat_output, name="Decoder")
		
		# AUTOENCODER
		self.autoencoder_input = Input(shape=self.input_shape, name="autoencoder_input")  # self.encoder_input#
		encoded_img = encoder_m(self.autoencoder_input)
		decoded_img = self.concat_model(encoded_img)
		self.total_model = Model(self.autoencoder_input, decoded_img, name="autoencoder")
		
		# COMPILE
		self.compile()
	
	def __init__(self, name=default_name, input_dim=32, property_dim=3, lr=0.01):
		self.name = name
		self.input_dim = input_dim
		# self.encoding_dim = property_dim * 2
		self.input_shape = (input_dim,)
		# self.encoding_shape = (self.encoding_dim,)
		self.lr = lr
		# self.property_output_dim = self.input_dim//2
		
		# ENCODER
		p1_dims = 1#property_dim
		p2_dims = 1#property_dim
		p3_dims = 1#property_dim
		self.encoder_input = Input(shape=self.input_shape, name="original_code")
		# properties
		p_activation = 'sigmoid'
		code_activation = 'sigmoid'
		# p1 = Dense(code_dim, activation=p_activation, name="p1_0")(self.encoder_input)
		p1 = Dense(self.input_dim//2, activation=p_activation, name="p1_1")(self.encoder_input)
		p1_code = Dense(p1_dims, activation=code_activation, name="p1_code")(p1)
		p1 = Dense(self.input_dim//2, activation=p_activation, name="p1_2")(p1_code)
		p1_output = Dense(self.input_dim, activation=p_activation, name="p1_3")(p1)
		
		p2 = Dense(self.input_dim//2, activation=p_activation, name="p2_1")(self.encoder_input)
		p2_code = Dense(p2_dims, activation=p_activation, name="p2_code")(p2)
		p2 = Dense(self.input_dim//2, activation=p_activation, name="p2_2")(p2_code)
		p2_output = Dense(self.input_dim, activation=p_activation, name="p2_3")(p2)

		p3 = Dense(self.input_dim//2, activation=p_activation, name="p3_1")(self.encoder_input)
		p3_code = Dense(p3_dims, activation=p_activation, name="p3_code")(p3)
		p3 = Dense(self.input_dim//2, activation=p_activation, name="p3_2")(p3_code)
		p3_output = Dense(self.input_dim, activation=p_activation, name="p3_3")(p3)
		
		ps = [p1_output,p2_output,p3_output]
		codes=[p1_code, p2_code, p3_code]
		self.encoder_output = concatenate(ps)
		# encoder_m = Model(self.encoder_input,outputs=self.encoder_output, name="encoder_m")
		self.encoder_model = Model(self.encoder_input, outputs=codes, name="Encoder")
		
		# CONCAT
		# self.concat_input = Input(shape=(input_dim * 3,), name="concat_input")
		# self.concat_output = Dense(self.input_dim, name="concat_output", activation=p_activation)(self.concat_input)
		# self.concat_model = Model(self.concat_input, self.concat_output, name="Concat")
		self.concat_output = Dense(self.input_dim, name="concat_output", activation=p_activation)(self.encoder_output)

		# PROPERTY DECODER
		self.decoder_input = codes#p1_code#Input(shape=(input_dim*3,), name="decoder_input")
		self.decoder_output = self.concat_output
		self.decoder_model=Model(self.decoder_input, self.decoder_output, name="Property_decoder")
		
		# AUTOENCODER
		self.autoencoder_input = self.encoder_input#Input(shape=self.input_shape, name="autoencoder_input")  # self.encoder_input#
		# encoded_img = encoder_m(self.autoencoder_input)
		# decoded_img = self.concat_model(encoded_img)
		self.total_model = Model(self.autoencoder_input, self.concat_output, name="autoencoder")

		# AUTOENCODER OLD
		# self.autoencoder_input = Input(shape=self.input_shape, name="autoencoder_input")  # self.encoder_input#
		# encoded_img = encoder_m(self.autoencoder_input)
		# decoded_img = self.concat_model(encoded_img)
		# self.total_model = Model(self.autoencoder_input, decoded_img, name="autoencoder")

		
		# COMPILE
		self.compile()
	
	def img(self, images, ae_z, ae, n=10):
		pae_z = self.encode(ae_z)
		pae_reconstructed_z = self.predict(ae_z)
		ae_reconstruced_images = ae.decode(ae_z)
		pae_reconstructed_images = ae.decode(pae_reconstructed_z)
		# decoded_imgs = self.decode(encodings)
		plt.figure(figsize=(20, 4))
		nrows = 3
		for i in range(n):
			# Display original
			ax = plt.subplot(nrows, n, i + 1)
			plt.imshow(images[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			# img_str = str(np.round(pae_reconstructed_z[i], 2))
			ax.set_title("OG",wrap=True)
			
			# Display ae reconstruction
			ax = plt.subplot(nrows, n, i + 1 + n)
			plt.imshow(ae_reconstruced_images[i].reshape(28, 28))
			plt.gray()
			# img_str = f"{str(np.round(pae_z[0][i], 2))},{str(np.round(pae_z[1][i], 2))},{str(np.round(pae_z[2][i], 2))}"
			# print(img_str)
			ax.set_title("AE")
			# ax.text(-30, (45 + (20 * i % 2)) * (-1 * i % 2), img_str, bbox=dict(facecolor='red', alpha=0.5))
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			
			# Display property reconstruction
			ax = plt.subplot(nrows, n, i + 1 + (n*2))
			plt.imshow(pae_reconstructed_images[i].reshape(28, 28))
			plt.gray()
			img_str = f"{str(np.round(pae_z[0][i], 2))},{str(np.round(pae_z[1][i], 2))},{str(np.round(pae_z[2][i], 2))}"
			# print(img_str)
			ax.set_title(img_str)
			# ax.text(-30, (45 + (20 * i % 2)) * (-1 * i % 2), img_str, bbox=dict(facecolor='red', alpha=0.5))
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		plt.show()
	
	def vis_nice_images_load(self, imgs, encodings):
		n = len(encodings)  # shape[0]
		n = int(round(n**(1/3),ndigits=0))
		title_n = 2
		x_n=0
		y_n=1
		for kindex in range(n):
			fig, axs = plt.subplots(nrows=n, ncols=n)
			fig.set_figwidth(8)
			fig.set_figheight(8)
			plt.subplots_adjust(left=0.2, wspace=0.07, hspace=0.07)
			title_index = kindex*(n**2)
			fig.suptitle(f"N{title_n+1}={np.round(encodings[title_index][title_n], 2)}", fontsize=9)
			plt.gray()
			
			for index in range(n):
				for jindex in range(n):
					# i = (index*(n**2))+(jindex * n) + kindex # Inverted index, sometimes results in better readable image configurations
					# i = (index * (n ** 2)) + (jindex * n) + kindex  # Other Inverted index
					i = (kindex*(n**2))+(index * n) + jindex # Regular index
					ax_index = (index * n) + jindex
					ax = axs.flat[ax_index]
					ax.imshow(imgs[i].reshape(28, 28))
					
					ax.get_xaxis().set_visible(False)
					ax.get_yaxis().set_visible(False)
					if index == 0:
						ax.set_title(
							f"N{x_n+1}={np.round(encodings[i][x_n], 2)}",
							fontsize=9)
					if jindex == 0:
						ax.set_ylabel(f"N{y_n+1}={np.round(encodings[i][y_n], 2)}",
						              fontsize=9, rotation=45, labelpad=20)
						ax.get_yaxis().set_visible(True)
						ax.tick_params(
							axis='y',  # changes apply to the x-axis
							which='both',  # both major and minor ticks are affected
							bottom=False,  # ticks along the bottom edge are off
							top=False,  # ticks along the top edge are off
							left=False,
							right=False,
							labelleft=False,
							labelbottom=False)  # labels along the bottom edge are off
			# fig.align_labels()
			fig.tight_layout()
			plt.show()
	
	def vis_nice_images(self,encodings, autoencoder):
		n = len(encodings)  # shape[0]
		
		imgs = []
		for i in range(n):
			decodings = self.decode(encodings[i])
			img = autoencoder.decode(decodings)
			imgs.append(img)
		
		n = int(math.sqrt(n))
		fig, axs = plt.subplots(nrows=n, ncols=n)
		fig.set_figwidth(8)
		fig.set_figheight(8)
		plt.subplots_adjust(left=0.2,wspace=0.07,hspace=0.07)
		fig.suptitle(f"N_1={np.round(encodings[0][0][0], 2)}",fontsize=10)
		plt.gray()
		
		for index in range(n):
			for jindex in range(n):
				i = (index * n) + jindex
				ax = axs.flat[i]
				ax.imshow(imgs[i].reshape(28, 28))

				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)
				if index==0:
					ax.set_title(
						f"N_3={np.round(encodings[i][2][0], 2)}",#, {np.round(encodings[i][1][0], 2)}, {np.round(encodings[i][2][0], 2)})",
						fontsize=10)
				if jindex==0:
					# mpl_toolkits. axes_grid1.axes_divider.make_axes_area_auto_adjustable(ax)
					ax.set_ylabel(f"N_2={np.round(encodings[i][1][0], 2)}",#, {np.round(encodings[i][1][0], 2)}, {np.round(encodings[i][2][0], 2)})",
						fontsize=10,
						          # rotation=45,
						          labelpad=40)
					ax.get_yaxis().set_visible(True)
					ax.tick_params(
						axis='y',  # changes apply to the x-axis
						which='both',  # both major and minor ticks are affected
						bottom=False,  # ticks along the bottom edge are off
						top=False,  # ticks along the top edge are off
						left=False,
						right=False,
						labelleft=False,
						labelbottom=False)  # labels along the bottom edge are off
					# ax.ylabel('X-Axis Label')
				# ax.set_title(
				# f"({np.round(encodings[i][0][0], 2)}, {np.round(encodings[i][1][0], 2)}, {np.round(encodings[i][2][0], 2)})",
				# 		fontsize=8)
		# fig.tight_layout()
		# fig.align_labels()
		plt.show()
		
		# double_mode = False
		# if n > 10:
		# 	n = int(math.sqrt(n))
		# 	double_mode = True
		#
		# if double_mode:
		# 	index = 0
		# 	# fig, axs = plt.subplots(n, n)
		# 	for i in range(n):
		# 		for j in range(n):
		#
		# 			img = imgs[index].reshape(28, 28)
		#
		# 			ax = plt.subplot(n, n, index + 1)
		# 			plt.imshow(img)
		# 			plt.gray()
		# 			ax.set_title(
		# 				f"{np.round(encodings[index][0], 2)},{np.round(encodings[index][1], 2)},{np.round(encodings[index][2], 2)}",
		# 				fontsize=8)
		# 			ax.get_xaxis().set_visible(False)
		# 			ax.get_yaxis().set_visible(False)
		#
		# if not double_mode:
		
	def vis_encodings(self, encodings, autoencoder):
		n = len(encodings)  # shape[0]
		
		imgs = []
		for i in range(n):
			decodings = self.decode(encodings[i])
			img = autoencoder.decode(decodings)
			imgs.append(img)
		
		# decodings = self.decode(encodings)
		# imgs = autoencoder.decode(decodings)
		
		n = int(math.sqrt(n))

		double_mode = False
		if n > 10:
			n = int(math.sqrt(n))
			double_mode= True
			
		if double_mode:
			index = 0
			# fig, axs = plt.subplots(n, n)
			for i in range(n):
				for j in range(n):
					index = (i*n)+j
					img = imgs[index].reshape(28, 28)
					
					ax = plt.subplot(n, n, index+1)
					plt.imshow(img)
					plt.gray()
					ax.set_title(f"{np.round(encodings[index][0],2)},{np.round(encodings[index][1],2)},{np.round(encodings[index][2],2)}", fontsize=8)
					ax.get_xaxis().set_visible(False)
					ax.get_yaxis().set_visible(False)
					
		if not double_mode:
			for i in range(n):
				# Display images
				# ax = plt.subplot(nrows=1, ncols=n, index=i + 1)
				ax = plt.subplot(1, n, i + 1)
				plt.imshow(imgs[i].reshape(28, 28))
				plt.gray()
				ax.get_xaxis().set_visible(False)
				ax.get_yaxis().set_visible(False)
		plt.show()
		
	def vis_activations(self, data,ae):
		fig, axs = plt.subplots(3, 3)
		# fig.colorbar(mappable=cm.ScalarMappable(norm=Normalize(), cmap='coolwarm'), ax=None)
		j = 0
		for d in data:
			for i in range(len(d)):
				x = d[i]
				id = i + (j * 3)
				z=ae.encode(x)
				encoded = self.encode(z)
				# im = axs[j][i].plot(encoded[:, :2], 'r', alpha=0.6)  # , cmap='viridis', norm=None, vmin=0, vmax=1)
				# im = axs[j][i].plot(encoded[:, 2:], 'b', alpha=0.6)
				# im = axs[j][i].plot(encoded[0][:,0], 'r', alpha=0.6)
				# im = axs[j][i].plot(encoded[0][:,1], 'm', alpha=0.6)
				im = axs[j][i].plot(encoded[0], 'r', alpha=0.6)
				im = axs[j][i].plot(encoded[1], 'm', alpha=0.6)
				im = axs[j][i].plot(encoded[2], 'b', alpha=0.6)
				title = ["Up", "Center", "Down"][i] + ["Circle", "Square", "Triangle"][j]
				axs[j][i].set_title(title)
			j = j + 1
		fig.tight_layout()
		plt.show()
	


class PAEa_supervised(PAEa):
	default_name = "propertyAE_a_supervised"

	def compile(self):
		opt = Adam(lr=self.lr)
		self.total_model.compile(optimizer=opt, loss='mean_squared_error')
	
	def __init__(self, name=default_name, input_dim=32, property_dim=3, lr=0.01):
		self.name = name
		self.input_dim = input_dim
		self.encoding_dim = property_dim * 2
		self.input_shape = (input_dim,)
		self.encoding_shape = (self.encoding_dim,)
		self.lr = lr
		self.property_output_dim = self.input_dim//2
		
		# ENCODER
		p1_dims = 2#property_dim
		p2_dims = 1#property_dim
		self.encoder_input = Input(shape=self.input_shape, name="original_code")
		# properties
		p_activation = 'relu'
		p1 = Dense(self.input_dim, activation=p_activation, name="p1_0")(self.encoder_input)
		p1 = Dense(self.input_dim//2, activation=p_activation, name="p1_1")(p1)
		p1_output = Dense(p1_dims, activation='sigmoid', name="p1_code",
		                # activity_regularizer=regularizers.l1(0.001)
		                )(p1)
		# p1_output = Dense(self.property_output_dim, activation=p_activation, name="p1_3")(p1_code)
		# p1_output = Dense(code_dim, activation=p_activation, name="p1_4")(p1)
		
		p2 = Dense(self.input_dim, activation=p_activation, name="p2_0")(self.encoder_input)
		p2 = Dense(self.input_dim//2, activation=p_activation, name="p2_1")(p2)
		p2_output = Dense(p2_dims, activation='sigmoid', name="p2_code")(p2)
		# p2_output = Dense(self.property_output_dim, activation=p_activation, name="p2_3")(p2_code)
		# p2_output = Dense(code_dim, activation=p_activation, name="p2_4")(p2)
		
		ps = [p1_output, p2_output]
		self.encoder_output = concatenate(ps)
		self.total_model = Model(self.encoder_input,outputs=self.encoder_output, name="Model")
		
		# COMPILE
		self.compile()
		
	def fit(self, x_train, y, validation_split=0.1, epochs=100,graph=False):
		history = self.total_model.fit(x_train, y,
		                               epochs=epochs,
		                               batch_size=64,
		                               shuffle=True,
		                               validation_split=validation_split,
		                               verbose=2
		                               )
		if not graph:
			return history
		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper right')
		plt.show()
		
		return history
	
	def encode(self, data):
		encodings = self.total_model.predict(data)
		return encodings


class PAEa_supervised_from_img(PAEa_supervised):
	default_name = "propertyAE_a_supervised_from_img"
	
	def compile(self):
		opt = Adam(lr=self.lr)
		self.total_model.compile(optimizer=opt, loss='mean_absolute_error')
	
	def __init__(self, name=default_name, input_dim=784, property_dim=3, lr=0.01):
		self.name = name
		self.input_dim = input_dim
		self.encoding_dim = property_dim * 2
		self.input_shape = (input_dim,)
		self.encoding_shape = (self.encoding_dim,)
		self.lr = lr
		self.property_output_dim = self.input_dim // 2
		
		# ENCODER
		p1_dims = 2  # property_dim
		p2_dims = 1  # property_dim
		self.encoder_input = Input(shape=self.input_shape, name="original_code")
		# properties
		p_activation = 'relu'
		p1 = Dense(self.input_dim, activation=p_activation, name="p1_0")(self.encoder_input)
		p1 = Dense(self.input_dim // 2, activation=p_activation, name="p1_1")(p1)
		p1_output = Dense(p1_dims, activation='sigmoid', name="p1_code",
		                  # activity_regularizer=regularizers.l1(0.001)
		                  )(p1)
		# p1_output = Dense(self.property_output_dim, activation=p_activation, name="p1_3")(p1_code)
		# p1_output = Dense(code_dim, activation=p_activation, name="p1_4")(p1)
		
		p2 = Dense(self.input_dim, activation=p_activation, name="p2_0")(self.encoder_input)
		p2 = Dense(self.input_dim // 2, activation=p_activation, name="p2_1")(p2)
		p2_output = Dense(p2_dims, activation='sigmoid', name="p2_code")(p2)
		# p2_output = Dense(self.property_output_dim, activation=p_activation, name="p2_3")(p2_code)
		# p2_output = Dense(code_dim, activation=p_activation, name="p2_4")(p2)
		
		ps = [p1_output, p2_output]
		self.encoder_output = concatenate(ps)
		self.total_model = Model(self.encoder_input, outputs=self.encoder_output, name="Model")
		
		# COMPILE
		self.compile()


class PropertyAE:
	def __init__(self, name="property_autoencoder", input_dim=784, encoding_dim=32, lr=0.01):
		self.name = name
		self.input_dim = input_dim
		self.encoding_dim = encoding_dim
		self.input_shape = (input_dim,)
		self.encoding_shape = (self.encoding_dim,)
		
		# ENCODER
		p1_dims=32
		p2_dims=0
		self.encoder_input = Input(shape=self.input_shape, name="original_img")
		h = Dense(128, activation='sigmoid')(self.encoder_input)
		# properties
		p_activation = 'sigmoid'
		# p1 = Dense(64, activation=p_activation, name="p1")(h)
		# p2 = Dense(64, activation=p_activation, name="p2")(h)
		p1_output = Dense(p1_dims, activation=p_activation, name="p1_output")(h)
		if p2_dims > 0 :
			p2_output = Dense(p2_dims, activation=p_activation, name="p2_output")(h)
			ps = [p1_output, p2_output]
			self.encoder_output = concatenate(ps)
		# model
		else:
			self.encoder_output = p1_output
		self.encoder = Model(self.encoder_input, self.encoder_output,name="Encoder")
		
		# INBETWEEN
		# inb_activation = 'sigmoid'
		# inb_input = Input((1*(encoding_dim//3),),name="inb_input")
		# h = Dense(encoding_dim//2,activation=inb_activation)(inb_input)
		# inb_output = Dense(encoding_dim,activation=inb_activation,name="inb_output")(inb_input)
		# self.inb = Model(inb_input,inb_output,name="Inbetween")
		# inb_input = Input((128,),name="inb_input")
		# h = Dense(32, activation=inb_activation)(inb_input)
		# inb_output = Dense(128,activation=inb_activation,name="inb_output")(h)
		# self.inb = Model(inb_input,inb_output,name="Inbetween")
		
		# DECODER
		# self.decoder_input = Input(shape=(2*len(ps),), name="decoder_input")
		decoder_input_dims = p1_dims+p2_dims
		self.decoder_input = Input(shape=(decoder_input_dims,), name="decoder_input")
		h = Dense(64, activation='sigmoid')(self.decoder_input)
		h = Dense(128, activation='sigmoid')(h)
		self.decoder_output = Dense(self.input_dim, activation='sigmoid')(h)
		self.decoder = Model(self.decoder_input, self.decoder_output, name="Decoder")
		
		# AUTOENCODER
		self.autoencoder_input = Input(shape=self.input_shape, name="autoencoder_input")#self.encoder_input#
		encoded_img = self.encoder(self.autoencoder_input)
		decoded_img = self.decoder(encoded_img)
		self.autoencoder = Model(self.autoencoder_input, decoded_img, name="autoencoder")
		
		# COMPILE
		# opt = SGD(lr=lr)
		opt = Adam(lr=lr)
		self.autoencoder.compile(optimizer=opt, loss='binary_crossentropy')
	
		
	def fit(self, x_train, validation_split=0.1, epochs=100,graph=False):
		history = self.autoencoder.fit(x_train, x_train,
							 epochs=epochs,
							 batch_size=16,
							 shuffle=True,
							 validation_split=validation_split,
							 verbose=2
							 )
		if not graph:
			return history
		# summarize history for loss
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['train', 'test'], loc='upper right')
		plt.show()
		
		return history
	
	def encode(self, data):
		encodings = self.encoder.predict(data)
		return encodings
	
	def vis_self(self):
		# VIS
		self.autoencoder.summary()
		keras.utils.plot_model(self.autoencoder, "propertyAE_unrolled.png", show_shapes=True, show_layer_names=True, expand_nested=True)
		# keras.utils.plot_model(self.autoencoder, "propertyAE_compact.png", show_shapes=True, show_layer_names=True, expand_nested=False)
		keras.utils.model_to_dot(self.autoencoder)
	
	def vis_activations(self, data):
		fig, axs = plt.subplots(3,3)
		# fig.colorbar(mappable=cm.ScalarMappable(norm=Normalize(), cmap='coolwarm'), ax=None)
		j = 0
		for shape in data:
			for i in range(len(shape)):
				id = i+(j*3)
				d = shape[i]
				encoded = self.encode(d)
				
				# h,w = np.shape(encoded)
				# ax = plt.subplot(1,9,i+1+(j*3))
				im = axs[j][i].imshow(encoded,cmap='viridis',norm=None,vmin=0, vmax=1)
				# Loop over data dimensions and create text annotations.
				# for i in range(w):
				# 	for j in range(h):
				# 		text = ax.text(j, i, np.round(encoded[i, j],1),
				# 					   ha="center", va="center", color="w")
				title = ["Up", "Center", "Down"][i] + ["Circle", "Square", "Triangle"][j]
				axs[j][i].set_title(title)
			j=j+1
		fig.tight_layout()
		plt.show()
	
	def vis_output(self, data, n=10):  # How many digits we will display
		encoded_imgs = self.encode(data)
		decoded_imgs = self.decoder.predict(encoded_imgs)
		
		plt.figure(figsize=(20, 4))
		for i in range(n):
			# Display original
			ax = plt.subplot(2, n, i + 1)
			plt.imshow(data[i].reshape(28, 28))
			plt.gray()
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
			
			# Display reconstruction
			ax = plt.subplot(2, n, i + 1 + n)
			plt.imshow(decoded_imgs[i].reshape(28, 28))
			plt.gray()
			img_str = str(np.round(encoded_imgs[i], 2))
			# print(img_str)
			# ax.set_title(img_str)
			ax.text(-30, (45 + (20 * i % 2)) * (-1 * i % 2), img_str, bbox=dict(facecolor='red', alpha=0.5))
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		plt.show()
	
def run(data, epochs, encoding_dim,iterations=1, vis_self=False, vis_activations=True, vis_output=True, vis_data=None, loss_graph=False):
	histories = []
	for i in range(iterations):
		p_ae = PropertyAE(encoding_dim=encoding_dim, lr=0.01)
		if vis_self: p_ae.vis_self()
		history = p_ae.fit(data, epochs=epochs,graph=loss_graph)
		histories.append(history)
	np.save(file=DIR.histories_dir+"pae_code22-10", arr=histories)
	if vis_output:
		v_d = data
		# if vis_data: v_d = vis_data[0]
		p_ae.vis_output(v_d)
	if vis_activations:
		p_ae.vis_activations(vis_data)
	return p_ae


class PAEb(PAEa):
	
	def __init__(self, name, input_dim=32, property_dim=3, lr=0.01):
		self.name = name
		self.input_dim = input_dim
		self.input_shape = (input_dim,)
		self.lr = lr
		
		# ENCODER
		p1_dims = 1  # property_dim
		p2_dims = 1  # property_dim
		p3_dims = 1  # property_dim
		self.encoder_input = Input(shape=self.input_shape, name="original_code")
		# properties
		p_activation = 'sigmoid'
		code_activation = 'sigmoid'
		p1 = Dense(self.input_dim // 2, activation=p_activation, name="p1_1")(self.encoder_input)
		p1_code = Dense(p1_dims, activation=code_activation, name="p1_code",
		                # activity_regularizer=regularizers.l1(0.001)
		                )(p1)
		p1 = Dense(self.input_dim // 2, activation=p_activation, name="p1_2")(p1_code)
		p1_output = Dense(self.input_dim, activation=p_activation, name="p1_3")(p1)
		# p1_output = Dense(code_dim, activation=p_activation, name="p1_4")(p1)
		
		# p2 = Dense(code_dim, activation=p_activation, name="p2_0")(self.encoder_input)
		p2 = Dense(self.input_dim // 2, activation=p_activation, name="p2_1")(self.encoder_input)
		p2_code = Dense(p2_dims, activation=p_activation, name="p2_code")(p2)
		p2 = Dense(self.input_dim // 2, activation=p_activation, name="p2_2")(p2_code)
		p2_output = Dense(self.input_dim, activation=p_activation, name="p2_3")(p2)
		# p2_output = Dense(code_dim, activation=p_activation, name="p2_4")(p2)
		
		p3 = Dense(self.input_dim // 2, activation=p_activation, name="p3_1")(self.encoder_input)
		p3_code = Dense(p3_dims, activation=p_activation, name="p3_code")(p3)
		p3 = Dense(self.input_dim // 2, activation=p_activation, name="p3_2")(p3_code)
		p3_output = Dense(self.input_dim, activation=p_activation, name="p3_3")(p3)
		
		ps = [p1_output, p2_output, p3_output]
		codes = [p1_code, p2_code, p3_code]
		self.encoder_output = concatenate(ps)
		# encoder_m = Model(self.encoder_input,outputs=self.encoder_output, name="encoder_m")
		self.encoder_model = Model(self.encoder_input, outputs=codes, name="Encoder")
		
		# CONCAT
		# self.concat_input = Input(shape=(input_dim * 3,), name="concat_input")
		# self.concat_output = Dense(self.input_dim, name="concat_output", activation=p_activation)(self.concat_input)
		# self.concat_model = Model(self.concat_input, self.concat_output, name="Concat")
		self.concat_output = Dense(self.input_dim, name="concat_output", activation=p_activation)(self.encoder_output)
		
		# PROPERTY DECODER
		self.decoder_input = codes  # p1_code#Input(shape=(input_dim*3,), name="decoder_input")
		self.decoder_output = self.concat_output
		self.decoder_model = Model(self.decoder_input, self.decoder_output, name="Property_decoder")
		
		# AUTOENCODER
		self.autoencoder_input = self.encoder_input  # Input(shape=self.input_shape, name="autoencoder_input")  # self.encoder_input#
		# encoded_img = encoder_m(self.autoencoder_input)
		# decoded_img = self.concat_model(encoded_img)
		self.total_model = Model(self.autoencoder_input, self.concat_output, name="autoencoder")
		
		# AUTOENCODER OLD
		# self.autoencoder_input = Input(shape=self.input_shape, name="autoencoder_input")  # self.encoder_input#
		# encoded_img = encoder_m(self.autoencoder_input)
		# decoded_img = self.concat_model(encoded_img)
		# self.total_model = Model(self.autoencoder_input, decoded_img, name="autoencoder")
		
		# COMPILE
		self.compile()

	
	def plot_encodings(self, encodings, autoencoder):
		n = len(encodings)
		imgs = []
		for i in range(n):
			decodings = self.decode(encodings[i])
			img = autoencoder.decode(decodings)
			imgs.append(img)
		
		m = int(math.sqrt(n))
		for i in range(n):
			index = i
			img = imgs[index].reshape(28, 28)
			
			ax = plt.subplot(m, m, index + 1)
			plt.imshow(img)
			plt.gray()
			element1 = encodings[index][0]#[0]
			element2 = encodings[index][1]#[0]
			element3 = encodings[index][2]#[0]
			# str1 = f"[{np.round(element1[0], 2)}, {np.round(element1[1], 2)}],"
			str1 = f"({np.round(element1, 2)},"
			str2 = f"{np.round(element2, 2)},"
			str3 = f"{np.round(element3, 2)})"
			ax.set_title(str1 + str2 + str3, fontsize=8)
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
		# return ax
	
	def vis_encodings(self, encodings, autoencoder):
		self.plot_encodings(encodings,autoencoder)
		plt.tight_layout()
		plt.show()
	
	def vis_encodings_sets(self, encodings, autoencoder):
		n = len(encodings)
		m = n#int(n/2)
		fig = plt.figure()
		subfigs = fig.subfigures(2, 2, wspace=0.07)
		index = 0
		# axsLeft = subfigs[0].subplots(1, 2, sharey=True)
		for i in range(m):
			for j in range(m):
				# axs = subfigs[i][j].subplots(2, 2)
				axs = self.plot_encodings(encodings[index], autoencoder)
				subfigs[i][j].ax
				index += 1
		# for enc in encodings:
		# 	axs = subfigs[index].subplots(2, 2)
		# 	axs[index] = self.plot_encodings(enc,autoencoder)
		# 	index +=1
		plt.show()


class PAEregularised(PAEb):
	
	def __init__(self, name, input_dim=32, property_dim=3, lr=0.01, reg_term=0.01):
		self.name = name
		self.input_dim = input_dim
		self.input_shape = (input_dim,)
		self.lr = lr
		self.reg = reg_term
		
		# ENCODER
		p1_dims = 1  # property_dim
		p2_dims = 1  # property_dim
		p3_dims = 1  # property_dim
		self.encoder_input = Input(shape=self.input_shape, name="original_code")
		# properties
		p_activation = 'sigmoid'
		code_activation = 'sigmoid'
		p1 = Dense(self.input_dim // 2, activation=p_activation, name="p1_1")(self.encoder_input)
		p1_code = Dense(p1_dims, activation=code_activation, name="p1_code",
		                activity_regularizer=regularizers.l1(self.reg)
		                )(p1)
		p1 = Dense(self.input_dim // 2, activation=p_activation, name="p1_2")(p1_code)
		p1_output = Dense(self.input_dim, activation=p_activation, name="p1_3")(p1)
		# p1_output = Dense(code_dim, activation=p_activation, name="p1_4")(p1)
		
		# p2 = Dense(code_dim, activation=p_activation, name="p2_0")(self.encoder_input)
		p2 = Dense(self.input_dim // 2, activation=p_activation, name="p2_1")(self.encoder_input)
		p2_code = Dense(p2_dims, activation=p_activation, name="p2_code",
		                activity_regularizer=regularizers.l1(self.reg))(p2)
		p2 = Dense(self.input_dim // 2, activation=p_activation, name="p2_2")(p2_code)
		p2_output = Dense(self.input_dim, activation=p_activation, name="p2_3")(p2)
		# p2_output = Dense(code_dim, activation=p_activation, name="p2_4")(p2)
		
		p3 = Dense(self.input_dim // 2, activation=p_activation, name="p3_1")(self.encoder_input)
		p3_code = Dense(p3_dims, activation=p_activation, name="p3_code",
		                activity_regularizer=regularizers.l1(self.reg))(p3)
		p3 = Dense(self.input_dim // 2, activation=p_activation, name="p3_2")(p3_code)
		p3_output = Dense(self.input_dim, activation=p_activation, name="p3_3")(p3)
		
		ps = [p1_output, p2_output, p3_output]
		codes = [p1_code, p2_code, p3_code]
		self.encoder_output = concatenate(ps)
		# encoder_m = Model(self.encoder_input,outputs=self.encoder_output, name="encoder_m")
		self.encoder_model = Model(self.encoder_input, outputs=codes, name="Encoder")
		
		# CONCAT
		# self.concat_input = Input(shape=(input_dim * 3,), name="concat_input")
		# self.concat_output = Dense(self.input_dim, name="concat_output", activation=p_activation)(self.concat_input)
		# self.concat_model = Model(self.concat_input, self.concat_output, name="Concat")
		self.concat_output = Dense(self.input_dim, name="concat_output", activation=p_activation)(self.encoder_output)
		
		# PROPERTY DECODER
		self.decoder_input = codes  # p1_code#Input(shape=(input_dim*3,), name="decoder_input")
		self.decoder_output = self.concat_output
		self.decoder_model = Model(self.decoder_input, self.decoder_output, name="Property_decoder")
		
		# AUTOENCODER
		self.autoencoder_input = self.encoder_input  # Input(shape=self.input_shape, name="autoencoder_input")  # self.encoder_input#
		# encoded_img = encoder_m(self.autoencoder_input)
		# decoded_img = self.concat_model(encoded_img)
		self.total_model = Model(self.autoencoder_input, self.concat_output, name="autoencoder")
		
		# AUTOENCODER OLD
		# self.autoencoder_input = Input(shape=self.input_shape, name="autoencoder_input")  # self.encoder_input#
		# encoded_img = encoder_m(self.autoencoder_input)
		# decoded_img = self.concat_model(encoded_img)
		# self.total_model = Model(self.autoencoder_input, decoded_img, name="autoencoder")
		
		# COMPILE
		self.compile()