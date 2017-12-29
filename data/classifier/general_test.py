# -*- coding: utf-8 -*-
from skimage import transform
import tensorflow as tf
import numpy as np
import glob
import face_recognition as FR
import os
import shutil

def read_one_image(image_file, width, height):
    # img = io.imread(image_file)
    img = FR.load_image_file(image_file)
    img = transform.resize(img,(width, height))
    # 避免有黑白格式的图片进入
    if img.shape != (width, height, 3):
        print(image_file, img.shape)
        return None
    return np.asarray(img)


def judge_category(img_file, tf_session, logits, tensor, category_dict, width, height):

    img = FR.load_image_file(img_file)
    faces = FR.face_locations(img)
    if len(faces) != 1:
		return None
	# 截取头像
	# print('    find face', faces[0], img_file)
    top, right, bottom, left = faces[0]
    face_image = img[top:bottom, left:right]
    # 处理图片的缩放，扩充到100x100
    face_array = np.asarray(face_image)
    face_array = transform.resize(face_array, (width, height))
	# 归一化
    face_array = np.asarray(face_array, np.float32)

    feed_dict = {tensor: [face_array]}
    classification_result = tf_session.run(logits, feed_dict)

	# 打印出预测矩阵
    print(classification_result)
	# 根据索引通过字典对应花的分类
    output = tf.argmax(classification_result, 1).eval()
    result = ''
    for i in range(len(output)):
        print(category_dict[output[i]], img_file)
        result = category_dict[output[i]]
	return result

def collect_results(data, dest_dir):
	if not os.path.exists(dest_dir):
		os.mkdir(dest_dir)
	for d in data:
		shutil.copy(d, dest_dir)

def test_classification(root_dir, category_dict, test_dir=None, w=100, h=100):
	model_dir = root_dir + '/models'
	if test_dir is None:
		test_dir = root_dir + '/test_files'

	with tf.Session() as sess:
		saver = tf.train.import_meta_graph(model_dir + '/model.ckpt.meta')
		saver.restore(sess, tf.train.latest_checkpoint(model_dir))
		graph = tf.get_default_graph()
		tensor = graph.get_tensor_by_name("x:0")
		logits = graph.get_tensor_by_name("logits_eval:0")
		datas = glob.glob(test_dir + '/*.jpg')
		datas.sort()
		stats = [0, 0, 0]
		smiles = []
		unsmiles = []
		unknown = []
		idx = 0
		for im in datas:
			idx += 1
			if idx < 10000:
				continue
			if idx % 50 == 0:
				print('test process#', idx, '.......')
			category = judge_category(im, sess, logits, tensor, category_dict, w, h)
			if category is None:
				stats[2] += 1
				unknown.append(im)
				continue
			if category == 'smile':
				stats[1] += 1
				smiles.append(im)
			else:
				stats[0] += 1
				unsmiles.append(im)
		print('nosmile#', stats[0])
		print('smile#', stats[1])
		print('unknown#', stats[2])
		print('nosmile', unsmiles)
		print('smile', smiles)
		collect_dir = root_dir + '/collect'
		collect_results(unsmiles, collect_dir + '/nosmile')
		collect_results(smiles, collect_dir + '/smile')

