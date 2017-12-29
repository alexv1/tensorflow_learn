# -*- coding: utf-8 -*-
import string, os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import random
import tensorflow as tf

dir_name = '/Users/apple/Desktop/dl_data/genki4k'
# set the file path
files = os.listdir(dir_name)
for f in files:
    print (dir_name + os.sep + f)

# file_path = dir_name + os.sep + files[10]
# # get the data
# dic_mat = scipy.io.loadmat(file_path)
# data_mat = dic_mat['Face_64']
# file_path2 = dir_name + os.sep + files[15]
#
# dic_label = scipy.io.loadmat(file_path2)
# label_mat = dic_label['Label']
# file_path3 = dir_name + os.sep+files[16]
#
# # get the label
# label = label_mat.ravel()
