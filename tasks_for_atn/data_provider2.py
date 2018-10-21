import numpy as np
import os
import shutil
import glob
import cv2
import random
import matplotlib.pyplot as plt
import math

class DataProvider:

    def __init__(self, video_name, img_sz, num_channels):

        # self.config = config
        self.feed_path = "../Data"

        # load data from video
        # makedir(self.feed_path)
        # feed_size = extractImages(video_name, self.feed_path)
        #feed_size = 8900
        crop_size = 80
        s_idx = img_sz // 2 - crop_size // 2

        print('Start loading data ...')

        # prepare cropped images from all data set
        files = glob.glob(self.feed_path + "/*.jpg")
        all_data_cropped = []
        self.train = []
        self.test = []
        j = 0 # should be 0
        for img_str in files:
            if j+crop_size <= img_sz:
                I = cv2.imread(img_str)
                I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
                I = cv2.resize(I, (img_sz, img_sz))

                # I = I / 255. # test
                # I = np.atleast_3d(I) # test
                # img = np.reshape(I,(img_sz * img_sz * 3)) # test

                crop = I[s_idx:s_idx+crop_size, j:j+crop_size, :]
                #crop = I[s_idx:s_idx + crop_size,20:20 + crop_size,:] # test
                crop = crop / 255.
                # generate outliers in a low probability:
                # r = random.uniform(0, 1)
                # if r >= 0.70:
                #     r2 = int(random.uniform(0, 1)*(crop_size/2))
                #     crop = generate_outliers(crop, r2, r2+30)

                # img = np.zeros((img_sz, img_sz, 3))  # prepare the embedded image here just for testing
                # img[s_idx:s_idx+crop_size, j:j+crop_size, :] = crop
                # img = np.reshape(img, (img_sz*img_sz*3))
                # self.train.append(img)
                # self.test.append(img)

                all_data_cropped.append(crop)
                #j += 1
            else:
                j = 0

        # prepare the training and test data:
        for i in range(len(all_data_cropped)):
            I, out = embed_image_test(all_data_cropped[i], img_sz, crop_size) #  embed_image(all_data_cropped[i], img_sz, crop_size)
            I = np.reshape(out, (img_sz * img_sz * num_channels)) # change to out
            self.train.append(I)
            self.test.append(I)

        self.train = np.array(self.train)
        self.test = np.array(self.test)
        self.train_size = self.train.shape[0]
        self.test_size = self.test.shape[0]

        # plt.imshow(self.train[0, ...])
        # plt.show()
        # plt.imshow(self.train[j//2, ...])
        # plt.show()
        # plt.imshow(self.train[j-1, ...])
        # plt.show()

        print('Finished uploading data, Train data shape:', self.train.shape, '; Test data shape:', self.test.shape)

    def next_batch(self, batch_size, data_type):
        batch_img = None
        if data_type == 'train':
            idx = np.random.choice(self.train_size, batch_size)
            batch_img = self.train[idx, ...]
        elif data_type == 'test':
            idx = np.random.choice(self.test_size, batch_size)
            batch_img = self.test[idx, ...]
        return batch_img


def generate_outliers(X, s, e):
    X_o = np.reshape(X, X.shape)
    start_idx = np.array([s, s])
    end_idx = np.array([e, e])
    for i in range(start_idx[0], end_idx[0]):
        for j in range(start_idx[1], end_idx[1]):
            X_o[i][j] = np.random.random_integers(0, 1)
    return X_o


# return (img_sz x img_sz x 4)
def embed_image(I, img_sz, crop_size):
    # wrap I and W in an image of size (img_sz*img_sz*3) and (img_sz*img_sz*1) respectively.
    s_idx = img_sz // 2 - crop_size // 2
    img = np.ones((img_sz, img_sz, 3))
    img[s_idx:s_idx+crop_size, s_idx:s_idx+crop_size, :] = I
    W = np.zeros((img_sz, img_sz, 1))
    W[s_idx:s_idx+crop_size, s_idx:s_idx+crop_size, :] = 1
    # concat img and W, and create an output of size: (img_sz*img_sz*4)
    out = np.concatenate((img, W), axis=2)
    return img, out


def embed_image_test(I, img_sz, crop_size):
    i = math.floor(10*np.random.random_integers(0, 1))
    # wrap I and W in an image of size (img_sz*img_sz*3) and (img_sz*img_sz*1) respectively.
    s_idx = img_sz // 2 - crop_size // 2
    img = np.ones((img_sz, img_sz, 3))
    img[s_idx:s_idx+crop_size, i:i+crop_size, :] = I
    W = np.ones((img_sz, img_sz, 1))  # test: return to: np.zeros((img_sz, img_sz, 1))
    # concat img and W, and create an output of size: (img_sz*img_sz*4)
    out = img #np.concatenate((img, W), axis=2)
    return img, out


def makedir(folder_name):
    try:
        if os.path.exists(folder_name) and os.path.isdir(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(folder_name)
    except OSError:
        pass
    # cd into the specified directory
    # os.chdir(folder_name)