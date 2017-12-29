# -*- coding: utf-8 -*-
from data.classifier.smile_train import start_train_smile
from data.classifier.general_test import test_classification, read_one_image

root_dir = '/Users/apple/Desktop/dl_data/flower_photos'
root_dir = '/Users/apple/Desktop/dl_data/genki4k'


# model = start_train_smile(root_dir, augmentation=True)
# print('model is#', model)

category_dict = {0: 'dasiy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
category_dict = {0: 'nosmile', 1: 'smile'}
test = root_dir + '/test_files/faces'
test = '/Users/apple/Desktop/201712/backup/8'
test_classification(root_dir, category_dict, test_dir=test)

# file = root_dir + '/files/nosmile/file2163.jpg'
# nr = read_one_image(file, 100, 100)