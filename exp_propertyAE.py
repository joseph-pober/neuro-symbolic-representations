import keras
import numpy as np
from keras import Input, Model
from keras.layers import Dense, Concatenate, concatenate
from keras.optimizers import SGD, Adam
from matplotlib import pyplot as plt, cm
import directories as DIR
from matplotlib.colors import Normalize


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