# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import time
from data.classifier.cnn_model import read_img, inference_2_category, mini_batches



def start_train_smile(root_dir = '/Users/apple/Desktop/dl_data/genki4k', w=100, h=100, c=3, ratio=0.8, augmentation=False, aug_count=10):
	files_dir = root_dir + '/files/'
	model_path = root_dir + '/models/model.ckpt'
	data, label = read_img(files_dir, w, h, augmentation, aug_count)

	# 打乱顺序
	num_example = data.shape[0]
	print('num_example', num_example)
	arr = np.arange(num_example)
	np.random.shuffle(arr)
	data = data[arr]
	label = label[arr]

	# 将所有数据分为训练集和验证集
	s = np.int(num_example * ratio)
	print('ratio', ratio)
	x_train = data[:s]
	y_train = label[:s]
	x_val = data[s:]
	y_val = label[s:]

	# -----------------构建网络----------------------
	# 占位符
	x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
	y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

	# ---------------------------网络结束---------------------------
	regularizer = tf.contrib.layers.l2_regularizer(0.0001)
	logits = inference_2_category(x, False, regularizer)

	# (小处理)将logits乘以1赋值给logits_eval，定义name，方便在后续调用模型时通过tensor名字调用输出tensor
	b = tf.constant(value=1, dtype=tf.float32)
	logits_eval = tf.multiply(logits, b, name='logits_eval')

	loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_)
	train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
	correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
	acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	n_epoch = 10
	batch_size = 64

	saver = tf.train.Saver()
	sess = tf.Session()
	sess.run(tf.global_variables_initializer())
	for epoch in range(n_epoch):
		start_time = time.time()

		train_loss, train_acc, n_batch = 0, 0, 0
		for x_train_a, y_train_a in mini_batches(x_train, y_train, batch_size, shuffle=True):
			_, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
			train_loss += err;
			train_acc += ac;
			n_batch += 1
		print("   train loss: %f" % (np.sum(train_loss) / n_batch))
		print("   train acc: %f" % (np.sum(train_acc) / n_batch))
		val_loss, val_acc, n_batch = 0, 0, 0
		for x_val_a, y_val_a in mini_batches(x_val, y_val, batch_size, shuffle=False):
			err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
			val_loss += err;
			val_acc += ac;
			n_batch += 1
		print("   validation loss: %f" % (np.sum(val_loss) / n_batch))
		print("   validation acc: %f" % (np.sum(val_acc) / n_batch))
	print("save model %s" % model_path)
	out = saver.save(sess, model_path)
	print('out#', out)
	sess.close()
	return out