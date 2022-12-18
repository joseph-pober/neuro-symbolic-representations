def run():
	# DATA
	validation_split = 0.1
	positive_examples_amount = 10
	negative_examples_amount = positive_examples_amount
	# (x_train, _), (x_test, _) = data.load_mnist()
	# examples = data.load_examples("C:/Users/error/PycharmProjects/ml/data/simple/")
	#########
	# Left
	print("Loading Data...")
	p1, p_names1 = data.load_simple(DIR.shapes_left_dir, label_function=data.positive_label,
	                                n=positive_examples_amount)
	# Right
	n1, n_names1 = data.load_simple(DIR.shapes_right_dir, label_function=data.negative_label,
	                                n=negative_examples_amount)
	print("Data Loaded")
	examples1 = np.concatenate((p1, n1), axis=0)
	# examples2 = np.concatenate((p2, n2), axis=0)
	names1 = np.concatenate((p_names1, n_names1), axis=0)  # p_names+n_names#
	# names2 = np.concatenate((p_names2, n_names2), axis=0)  # p_names+n_names#
	print("Data Configured")
	#########
	# examples = data.randomise_data(examples)
	x1, y1 = data.split_data_label(examples1)
	# x2, y2 = data.split_data_label(examples2)
	
	# Autoencoder
	encoding_dim = 32
	# x = np.concatenate((x1, x2))
	# np.random.shuffle(x)
	ae = Autoencoder(encoding_dim=encoding_dim)
	# training
	# ae.fit(x, epochs=25)
	# ae.save()
	ae.load()
	
	# vis
	n = 10
	# ae.vis_output(x1[5:15], n=n)
	
	l = ae.encode(x1[:n])
	r = ae.encode(x1[-n:])
	
	avl = np.average(l, 0)
	avr = np.average(r, 0)
	print(np.round(avl, 2))
	print(np.round(avr, 2))
	print(np.round(avl - avr, 2))
# for i in range(n):



def run2():
    # DATA
    validation_split = 0.1
    positive_examples_amount = 10
    negative_examples_amount = positive_examples_amount * 3
    # (x_train, _), (x_test, _) = data.load_mnist()
    # examples = data.load_examples("C:/Users/error/PycharmProjects/ml/data/simple/")
    #########
    # Positive
    print("Loading Data...")
    p1, p2, p_names1, p_names2 = data.load_examples(DIR.shapes_left_dir, label_function=data.positive_label,
                                                    n=positive_examples_amount)
    # Negative
    n1, n2, n_names1, n_names2 = data.load_examples(DIR.negative_dir, label_function=data.negative_label,
                                                    n=negative_examples_amount)
    print("Data Loaded")
    examples1 = np.concatenate((p1, n1), axis=0)
    examples2 = np.concatenate((p2, n2), axis=0)
    names1 = np.concatenate((p_names1, n_names1), axis=0)  # p_names+n_names#
    names2 = np.concatenate((p_names2, n_names2), axis=0)  # p_names+n_names#
    print("Data Configured")
    #########
    # examples = data.randomise_data(examples)
    x1, y1 = data.split_data_label(examples1)
    x2, y2 = data.split_data_label(examples2)

    # ae.aef(x_train, x_train)
    # print(x_train.shape)
    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # x_test = x_train[int((1-validation_split) * len(x_train)):]
    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    # print(x_train.shape)
    # print(x_test.shape)

    # MODELS
    encoding_dim = 32
    #
    # Autoencoder
    # x = np.concatenate((x1, x2))
    # np.random.shuffle(x)
    ae = Autoencoder(encoding_dim=encoding_dim)
    # training
    # ae.fit(x, epochs=25)
    # ae.save()
    ae.load()
    # vis
    n = 10
    # ae.vis_output(x1[-n:], n=n)
    #
    #

    # Comparer
    # encoding
    enc_x1 = ae.encode(x1)
    enc_x2 = ae.encode(x2)
    enc_x, enc_y = data.assemble_pairs(enc_x1, enc_x2, y1, y2)
    enc_x, enc_y = data.unison_shuffled_copies(enc_x, enc_y)


    s=2
    # c = DefaultComparer(s)
    # c = Comparer(s)
    c = DeepComparer(s)
    c.plot_model()
    # tx=np.ones((10,10))
    # ty=np.ones((10,))
    enc_x, enc_y=generator.generate_mock_data(10000,s)
    c.fit(enc_x, enc_y, epochs=20)
    W = c.get_weights()
    for w in W:
        print(w)


    c.vis_history()
    c.vis(imgs1=x1, imgs2=x2, ae=ae, y=y1, n=n, names=names2)
