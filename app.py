import click
import urllib
import zipfile
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf

def le_net_model(features, labels, mode):
	input_layer = tf.reshape(features["x"], [-1, 32, 32, 3])

	# Convolutional Layer #1
	conv1 = tf.layers.conv2d(
			inputs=input_layer,
			filters=6,
			kernel_size=[5, 5],
			padding="valid",
			activation=tf.nn.relu)

	# Pooling Layer #1
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	# Convolutional Layer #2 and Pooling Layer #2
	conv2 = tf.layers.conv2d(
			inputs=pool1,
			filters=16,
			kernel_size=[5, 5],
			padding="valid",
			activation=tf.nn.relu)
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

	# Dense Layer
	pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])
	dense1 = tf.layers.dense(inputs=pool2_flat, units=120, activation=tf.nn.relu)
	dense2 = tf.layers.dense(inputs=dense1, units=84, activation=tf.nn.relu)

	# Logits Layer
	logits = tf.layers.dense(inputs=dense2, units=43)

	predictions = {
		# Generate predictions (for PREDICT and EVAL mode)
		"classes": tf.argmax(input=logits, axis=1),
		# Add `softmax_tensor` to the graph. It is used for PREDICT and by the
		# `logging_hook`.
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	if mode == tf.estimator.ModeKeys.PREDICT:
		return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

	# Calculate Loss (for both TRAIN and EVAL modes)
	loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode == tf.estimator.ModeKeys.TRAIN:
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
		train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
		return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

	# Add evaluation metrics (for EVAL mode)
	eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
	
	return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

@click.group()
def challenge():
	pass

@challenge.command()
def download():
	dataset = urllib.URLopener()
	dataset.retrieve("http://benchmark.ini.rub.de/Dataset_GTSDB/FullIJCNN2013.zip", "dataset.zip")
	zip_ref = zipfile.ZipFile("dataset.zip", 'r')
	zip_ref.extractall('images')
	zip_ref.close()

@challenge.command()
def log_reg_sk():
	path = 'images/FullIJCNN2013'
	img_files = [(os.path.join(root, name))
			for root, dirs, files in os.walk(path)
			for name in files
			if name.endswith((".ppm"))]
	# Resizing images size
	im_size = 32
	# reads every image, resizes them to 28x28 and flattens them, finally stacks them into a numpy array 
	features = np.vstack([np.reshape(cv2.resize(cv2.imread(i), (im_size, im_size)),(1,im_size*im_size*3)) for i in img_files])
	# Reads the classes from the foldar names and converts them to one hot notation
	labels = np.array([int(os.path.basename(os.path.dirname(i))) for i in img_files])

	# Split data into train and test
	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

	logReg = LogisticRegression()
	logReg.fit(X_train,y_train)

	print 'Testing score:', logReg.score(X_test,y_test)

@challenge.command()
def softmax_tf():
	path = 'images/FullIJCNN2013'
	img_files = [(os.path.join(root, name))
			for root, dirs, files in os.walk(path)
			for name in files
			if name.endswith((".ppm"))]
	# Resizing images size
	im_size = 28
	# reads every image, resizes them to 28x28 and flattens them, finally stacks them into a numpy array 
	features = np.vstack([np.reshape(cv2.resize(cv2.imread(i), (im_size, im_size)),(1,im_size*im_size*3)) for i in img_files])
	# Reads the classes from the foldar names and converts them to one hot notation
	labels = np.array([int(os.path.basename(os.path.dirname(i))) for i in img_files])
	labels =  labels.reshape(len(labels), 1)
	onehot_encoder = OneHotEncoder(sparse=False)
	labels_onehot_encoded = onehot_encoder.fit_transform(labels)

	# Split data into train and test
	X_train, X_test, y_train, y_test = train_test_split(features, labels_onehot_encoded, test_size=0.2, random_state=42)

	print X_train.shape, y_train.shape
	print X_test.shape, y_test.shape
	# TF Model 
	x = tf.placeholder(tf.float32, [None, 2352])
	
	W = tf.Variable(tf.zeros([2352, 43]))
	b = tf.Variable(tf.zeros([43]))

	y = tf.matmul(x, W) + b

	# Training
	y_ = tf.placeholder(tf.float32, [None, 43])
	#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	# Model evaluation 
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	# Variables initialization
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	for i in range(1000):
		batch_xs, batch_ys = next_batch(100, X_train, y_train)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

		if i % 50 == 0:
			train_acc = accuracy.eval({x: batch_xs, y_: batch_ys})
			print('step: %d, acc: %6.3f' % (i, train_acc) )
	print 'Accuracy: ', sess.run(accuracy, feed_dict={x: X_test, y_: y_test})


@challenge.command()
def le_net():
	path = 'images/FullIJCNN2013'
	img_files = [(os.path.join(root, name))
			for root, dirs, files in os.walk(path)
			for name in files
			if name.endswith((".ppm"))]
	# Resizing images size
	im_size = 32
	# reads every image, resizes them to 28x28 and flattens them, finally stacks them into a numpy array 
	features = np.vstack([(np.reshape(cv2.resize(cv2.imread(i), (im_size, im_size)),(1,im_size*im_size*3)))/255.0 for i in img_files])
	# Reads the classes from the foldar names and converts them to one hot notation
	labels = np.array([int(os.path.basename(os.path.dirname(i))) for i in img_files])
	labels = labels.reshape(len(labels), 1)

	# Split data into train and test
	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

	# Create the Estimator
	classifier = tf.estimator.Estimator(model_fn=le_net_model, model_dir="models/leNet")

	# Set up logging for predictions
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

	# Train the model
	train_input_fn = tf.estimator.inputs.numpy_input_fn(
			x={"x": X_train},
			y=y_train,
			batch_size=100,
			num_epochs=None,
			shuffle=True)

	classifier.train(
			input_fn=train_input_fn,
			steps=20000,
			hooks=[logging_hook])

	# Evaluate the model and print results
	eval_input_fn = tf.estimator.inputs.numpy_input_fn(
		x={"x": X_test},
		y=y_test,
		num_epochs=1,
		shuffle=False)
	eval_results = classifier.evaluate(input_fn=eval_input_fn)
	print(eval_results)

if __name__ == '__main__':
	challenge()