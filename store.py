import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from random import shuffle
import glob

shuffle_data = True  # shuffle the addresses before saving
hdf5_path = 'datasets/wechat_date.hdf5'  # address to where you want to save the hdf5 file
wechat_date_path = 'wechat-date/*.png'
# read addresses and labels from the 'train' folder
addrs = glob.glob(wechat_date_path)
labels = [0 if '0' in addr else 1 for addr in addrs]  # 0 = Cat, 1 = Dog
# to shuffle data
if shuffle_data:
    c = list(zip(addrs, labels))
    shuffle(c)
    addrs, labels = zip(*c)

# Divide the hata into 60% train, 20% validation, and 20% test
train_addrs = addrs[0:int(0.7 * len(addrs))]
train_labels = labels[0:int(0.7 * len(labels))]
test_addrs = addrs[int(0.7 * len(addrs)):]
test_labels = labels[int(0.7 * len(labels)):]

train_shape = (len(train_addrs), 60, 450, 4)
test_shape = (len(test_addrs), 60, 450, 4)

hdf5_file = h5py.File(hdf5_path, mode='w')

hdf5_file.create_dataset("train_img", train_shape, np.int8)
hdf5_file.create_dataset("test_img", test_shape, np.int8)

hdf5_file.create_dataset("train_mean", train_shape[1:], np.float32)

hdf5_file.create_dataset("train_labels", (len(train_addrs),), np.int8)
hdf5_file["train_labels"][...] = train_labels

hdf5_file.create_dataset("test_labels", (len(test_addrs),), np.int8)
hdf5_file["test_labels"][...] = test_labels

mean = np.zeros(train_shape[1:], np.float32)

for i in range(len(train_addrs)):
    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = train_addrs[i]

    img = ndimage.imread(addr, flatten=False)

    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    # save the image and calculate the mean so far
    print(addr)
    hdf5_file["train_img"][i, ...] = img[None]
    mean += img / float(len(train_labels))
# loop over test addresses
for i in range(len(test_addrs)):

    # read an image and resize to (224, 224)
    # cv2 load images as BGR, convert it to RGB
    addr = test_addrs[i]

    img = ndimage.imread(addr, flatten=False)

    # add any image pre-processing here
    # if the data order is Theano, axis orders should change
    # save the image
    hdf5_file["test_img"][i, ...] = img[None]
# save the mean and close the hdf5 file

hdf5_file["train_mean"][...] = mean
hdf5_file.close()