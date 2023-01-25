import keras
import numpy as np
from keras import Input, Model, regularizers
from keras.layers import Dense, Concatenate, concatenate
from keras.optimizers import SGD, Adam
from matplotlib import pyplot as plt, cm
import directories as DIR
from matplotlib.colors import Normalize

from aes import Autoencoder


class PAEa(Autoencoder):
	default_name = "propertyAE_a"
	
	def compile(self):
		opt = Adam(lr=self.lr)
		self.total_model.compile(optimizer=opt, loss='mean_squared_error')
	
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
		p1_code = Dense(p1_dims, activation=code_activation, name="p1_code",
		                # activity_regularizer=regularizers.l1(0.001)
		                )(p1)
		p1 = Dense(self.input_dim//2, activation=p_activation, name="p1_2")(p1_code)
		p1_output = Dense(self.input_dim, activation=p_activation, name="p1_3")(p1)
		# p1_output = Dense(code_dim, activation=p_activation, name="p1_4")(p1)
		
		# p2 = Dense(code_dim, activation=p_activation, name="p2_0")(self.encoder_input)
		p2 = Dense(self.input_dim//2, activation=p_activation, name="p2_1")(self.encoder_input)
		p2_code = Dense(p2_dims, activation=p_activation, name="p2_code")(p2)
		p2 = Dense(self.input_dim//2, activation=p_activation, name="p2_2")(p2_code)
		p2_output = Dense(self.input_dim, activation=p_activation, name="p2_3")(p2)
		# p2_output = Dense(code_dim, activation=p_activation, name="p2_4")(p2)
		

		p3 = Dense(self.input_dim//2, activation=p_activation, name="p3_1")(self.encoder_input)
		p3_code = Dense(p3_dims, activation=p_activation, name="p3_code")(p3)
		p3 = Dense(self.input_dim//2, activation=p_activation, name="p3_2")(p3_code)
		p3_output = Dense(self.input_dim, activation=p_activation, name="p3_3")(p3)
		
		ps = [p1_output,p2_output,p3_output]
		self.encoder_output = concatenate(ps)
		encoder_m = Model(self.encoder_input,outputs=self.encoder_output, name="Encoder")
		self.encoder = Model(self.encoder_input,outputs=[p1_code,p2_code,p3_code], name="Encoder")
		
		# DECODER
		self.decoder_input = Input(shape=(input_dim*3,), name="decoder_input")
		# h = Dense(self.input_dim, activation=p_activation, name="d1_1")(self.decoder_input)
		# h = Dense(64, activation='sigmoid')(self.decoder_input)
		# h = Dense(128, activation='sigmoid')(h)
		self.decoder_output = Dense(self.input_dim,name="output", activation=p_activation)(self.decoder_input)
		self.decoder = Model(self.decoder_input, self.decoder_output, name="Decoder")
		
		# AUTOENCODER
		self.autoencoder_input = Input(shape=self.input_shape, name="autoencoder_input")  # self.encoder_input#
		encoded_img = encoder_m(self.autoencoder_input)
		decoded_img = self.decoder(encoded_img)
		self.total_model = Model(self.autoencoder_input, decoded_img, name="autoencoder")
		
		# COMPILE
		self.compile()
		
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