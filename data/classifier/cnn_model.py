# -*- coding: utf-8 -*-
from skimage import io,transform
import tensorflow as tf
import numpy as np
import glob
import os
from imgaug import augmenters as iaa

# 定义一个函数，按批次取数据
def mini_batches(inputs=None, targets=None, batch_size=None, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batch_size]
		else:
			excerpt = slice(start_idx, start_idx + batch_size)
		yield inputs[excerpt], targets[excerpt]

def inference(input_tensor, train, regularizer):
	# 第一个卷积层（100——>50)
	# conv1 = tf.layers.conv2d(
	# 	inputs=x,
	# 	filters=32,
	# 	kernel_size=[5, 5],
	# 	padding="same",
	# 	activation=tf.nn.relu,
	# 	kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
	# pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	with tf.variable_scope('layer1-conv1'):
		conv1_weights = tf.get_variable("weight",[5,5,3,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
		conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
		relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

	with tf.name_scope("layer2-pool1"):
		pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")

	# 第二个卷积层(50->25)
	with tf.variable_scope("layer3-conv2"):
		conv2_weights = tf.get_variable("weight",[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
		conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

	with tf.name_scope("layer4-pool2"):
		pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

	with tf.variable_scope("layer5-conv3"):
		conv3_weights = tf.get_variable("weight",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
		conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
		relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
	with tf.name_scope("layer6-pool3"):
		pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

	with tf.variable_scope("layer7-conv4"):
		conv4_weights = tf.get_variable("weight",[3,3,128,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
		conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
		relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

	with tf.name_scope("layer8-pool4"):
		pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
		nodes = 6*6*128
		reshaped = tf.reshape(pool4,[-1,nodes])

	with tf.variable_scope('layer9-fc1'):
		fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
		fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))
		fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
		if train: fc1 = tf.nn.dropout(fc1, 0.5)

	with tf.variable_scope('layer10-fc2'):
		fc2_weights = tf.get_variable("weight", [1024, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
		fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))
		fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
		if train: fc2 = tf.nn.dropout(fc2, 0.5)
	with tf.variable_scope('layer11-fc3'):
		fc3_weights = tf.get_variable("weight", [512, 5],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
		fc3_biases = tf.get_variable("bias", [5], initializer=tf.constant_initializer(0.1))
		logit = tf.matmul(fc2, fc3_weights) + fc3_biases

	return logit


def inference_2_category(input_tensor, train, regularizer):
	# 第一个卷积层（100——>50)
	# conv1 = tf.layers.conv2d(
	# 	inputs=x,
	# 	filters=32,
	# 	kernel_size=[5, 5],
	# 	padding="same",
	# 	activation=tf.nn.relu,
	# 	kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
	# pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

	with tf.variable_scope('layer1-conv1'):
		conv1_weights = tf.get_variable("weight",[5,5,3,32],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1_biases = tf.get_variable("bias", [32], initializer=tf.constant_initializer(0.0))
		conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
		relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

	with tf.name_scope("layer2-pool1"):
		pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="VALID")

	# 第二个卷积层(50->25)
	with tf.variable_scope("layer3-conv2"):
		conv2_weights = tf.get_variable("weight",[5,5,32,64],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable("bias", [64], initializer=tf.constant_initializer(0.0))
		conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

	with tf.name_scope("layer4-pool2"):
		pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

	with tf.variable_scope("layer5-conv3"):
		conv3_weights = tf.get_variable("weight",[3,3,64,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv3_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
		conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
		relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
	with tf.name_scope("layer6-pool3"):
		pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

	with tf.variable_scope("layer7-conv4"):
		conv4_weights = tf.get_variable("weight",[3,3,128,128],initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv4_biases = tf.get_variable("bias", [128], initializer=tf.constant_initializer(0.0))
		conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='SAME')
		relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))

	with tf.name_scope("layer8-pool4"):
		pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
		nodes = 6*6*128
		reshaped = tf.reshape(pool4,[-1,nodes])

	with tf.variable_scope('layer9-fc1'):
		fc1_weights = tf.get_variable("weight", [nodes, 1024],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
		fc1_biases = tf.get_variable("bias", [1024], initializer=tf.constant_initializer(0.1))
		fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
		if train: fc1 = tf.nn.dropout(fc1, 0.5)

	with tf.variable_scope('layer10-fc2'):
		fc2_weights = tf.get_variable("weight", [1024, 512],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
		fc2_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))
		fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)
		if train: fc2 = tf.nn.dropout(fc2, 0.5)
	with tf.variable_scope('layer11-fc3'):
		fc3_weights = tf.get_variable("weight", [512, 2],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None: tf.add_to_collection('losses', regularizer(fc3_weights))
		fc3_biases = tf.get_variable("bias", [2], initializer=tf.constant_initializer(0.1))
		logit = tf.matmul(fc2, fc3_weights) + fc3_biases

	return logit


def read_img(folder, w, h, augmentation=False, aug_count=10):
	# 读取图片
	print('folder#', folder)
	category = [folder + x for x in os.listdir(folder) if os.path.isdir(folder + x)]
	imgs = []
	labels = []
	# 关键步骤，保证类别有序
	category.sort()
	print('category#', category)

	seq = get_augmentation()
	for idx, folder in enumerate(category):
		print(idx, folder)
		for im in glob.glob(folder + '/*.jpg'):
			print('images: %s' % (im))
			img = io.imread(im)
			img = transform.resize(img, (w, h))
			# 避免有黑白格式的图片进入
			if img.shape != (w, h, 3):
				print(im, img.shape)
				continue
			if augmentation:
				# 复制图片
				images = np.array(
					[img for _ in range(aug_count)],
					dtype=np.uint8
				)
				images_aug = seq.augment_images(images)
				aug_count = images_aug.shape[0]
				for i in range(0, aug_count):
					imgs.append(images_aug[i])
					labels.append(idx)
			else:
			    imgs.append(img)
			    labels.append(idx)
	labels = np.asarray(labels, np.int32)
	imgs = np.asarray(imgs, np.float32)
	return imgs, labels

def get_augmentation():
	seq = iaa.Sequential([
		iaa.Fliplr(0.5),  # horizontal flips
		iaa.Crop(percent=(0, 0.1)),  # random crops
		# Small gaussian blur with random sigma between 0 and 0.5.
		# But we only blur about 50% of all images.
		iaa.Sometimes(0.5,
					  iaa.GaussianBlur(sigma=(0, 0.5))
					  ),
		# Strengthen or weaken the contrast in each image.
		iaa.ContrastNormalization((0.75, 1.5)),
		# Add gaussian noise.
		# For 50% of all images, we sample the noise once per pixel.
		# For the other 50% of all images, we sample the noise per pixel AND
		# channel. This can change the color (not only brightness) of the
		# pixels.
		iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
		# Make some images brighter and some darker.
		# In 20% of all cases, we sample the multiplier once per channel,
		# which can end up changing the color of the images.
		iaa.Multiply((0.8, 1.2), per_channel=0.2),
		# Apply affine transformations to each image.
		# Scale/zoom them, translate/move them, rotate them and shear them.
		iaa.Affine(
			scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
			translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
			rotate=(-25, 25),
			shear=(-8, 8)
		)
	], random_order=True)
	return seq