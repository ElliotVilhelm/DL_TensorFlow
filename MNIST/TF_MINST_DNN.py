import tensorflow as tf

'''
feed forward
input>w1>h1(activation)>w2>h2(activation)>w3>outputlayer

compare output to intended output with cost function (ex: cross entropy)

apply optimizer to minimize cost (ex: Adam, SGD), this is our back prop

feed forward + backprop = epoch

'''
from tensorflow.examples.tutorials.mnist import input_data
minst = input_data.read_data_sets("/tmp/data/", one_hot=True)
#print(minst.index(0))

n_nodes_hl1 = 150
n_nodes_hl2 = 300
n_nodes_hl3 = 600
n_nodes_hl4 = 300
n_nodes_hl5 = 150


n_classes = 10 # 0- 9
batch_size = 100 # go 100 features at a time so we dont have to load it all to memory

# height by width = 28x28 = 764
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	'''
	Define Model
	'''
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	hidden_4_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl4]))}

	hidden_5_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl4, n_nodes_hl5])),
					  'biases':tf.Variable(tf.random_normal([n_nodes_hl5]))}

	# notice biases size 3 x number of classes
	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl5, n_classes])),
					'biases':tf.Variable(tf.random_normal([n_classes]))}


	# forward prop
	# (input data * weights) + biases
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	# activation
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
	l4 = tf.nn.relu(l4)

	l5 = tf.add(tf.matmul(l4, hidden_5_layer['weights']), hidden_5_layer['biases'])
	l5 = tf.nn.relu(l5)

	output = tf.add(tf.matmul(l5, output_layer['weights']), output_layer['biases'])

	# output is one-hot array
	return output

def train_neural_network(x):
	# prediction is one hot array
	prediction = neural_network_model(x)

	# avg error
	cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

	# LR: .0001 default
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# feed forward + back prop = 1 epoch
	n_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for epoch in range(n_epochs):
			epoch_loss = 0
			for _ in range(int(minst.train.num_examples/batch_size)):
				epoch_x, epoch_y = minst.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c
			print('Epoch ', epoch, 'completed out of', n_epochs, ', loss:', epoch_loss)


		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy: ', accuracy.eval({x:minst.test.images, y:minst.test.labels}))



train_neural_network(x)















