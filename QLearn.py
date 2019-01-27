from tensorflow.contrib.layers import convolution2d, fully_connected


def q_network(X_state, scope):
	prev_layer = X_state
	conv_layers = []
	with tf.variable_scope(scope) as scope:
		for n_maps, kernel_size, stride, padding, activation in zip(conv_n_maps, conv_kernel_sizes, conv_strides, conv_paddings, conv_activation):
			prev_layer = convolution2d(
				prev_layer, num_outputs=n_maps, kernel_size=kernel_size,
				stride=stride, padding=padding, activation_fn=activation,
				weights_initializer=initializer)
			conv_layers.append(prev_layer)
		last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
		hidden = fully_connected(
			last_conv_layer_flat, n_hidden, activation_fn=hidden_activation,
			weights_initializer=initializer)
		outputs = fully_connected(
			hidden, n_outputs, activation_fn=None,
			weights_initializer=initializer)
	trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
							scope=scope.name)
	trainable_vars_by_name = {var.name[len(scope.name):]: var
								for var in trainable_vars} 
	return outputs, trainable_vars_by_name

