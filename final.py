#!/usr/bin/env python

import tensorflow as tf
import random
import tensorflow_datasets as tfds
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import glob
import os

class PCBSerial:

    def __init__(self):
        self.models = {}
        for i in range(0, 7):
            self.models[f"model_{i}"] = tf.keras.models.load_model(f"model_digit{i}")

    def crop(self):
        if not os.path.exists('main_output'):
            os.mkdir('main_output')

        # Initiate SIFT detector
        img1 = cv2.imread('first_batch/data/golden.jpg')
        img2 = self.image
        # Initiate SIFT detector
        sift = cv2.xfeatures2d.SIFT_create()

        (h, w, d) = img1.shape
        r = 300.0 / w
        dim = (300, int(h * r))

        # find the keypoints and descriptors with SIFT
        golden_kp1, golden_des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)

        # BFMatcher with default params
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(golden_des1,des2,k=2)

        good = []
        for m in matches:
            if len(m)<=1: continue
            elif m[0].distance < 0.7*m[1].distance:
                good.append(m[0])

        MIN_MATCH_COUNT = 15
        if len(good)<MIN_MATCH_COUNT:
            print('Error: Not enough matches')

        #
        # Transformation
        src_pts = np.float32([ golden_kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2           [m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        Minv, maskinv = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)

        timg = cv2.warpPerspective(img2, Minv, (img1.shape[1],img1.shape[0]))
        cimg = timg[260:260+560,700:700+675]
        self.cropped = cimg
        cv2.imwrite("main_output/cropped.JPG", cimg)

    def threshold(self):
    
        (b, g, r) = cv2.split(self.cropped)
        lst = [b, g, r]
        channels = ['b', 'g', 'r']
        images = []
        for c in lst:
            scale_percent = 70
            width = int(c.shape[1] * scale_percent / 100)
            height = int(c.shape[0] * scale_percent / 100)
            dim = (width, height)
            image = cv2.resize(c, dim, interpolation = cv2.INTER_AREA)
            images.append(image)

        _, b_thresh = cv2.threshold(images[0],125,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        _, g_thresh = cv2.threshold(images[1],125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        _, r_thresh = cv2.threshold(images[2],125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        rb = np.bitwise_or(r_thresh, b_thresh)
        if np.mean(rb) >= 215:
            _, r_thresh = cv2.threshold(images[2],125,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            rb = np.bitwise_or(r_thresh, b_thresh)

        self.thresh = rb
        cv2.imwrite("main_output/threshed.JPG", rb)

    def process_path(file_path):
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, (80, 80))
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img/255
        return img

    def identify(self, filename):
        self.filename = filename
        self.image = cv2.imread(self.filename)
        self.crop()
        self.threshold()
        test_ds = tf.data.Dataset.list_files("main_output/threshed.JPG")
        labelled_test_ds  = test_ds.map(PCBSerial.process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        batch_test_ds  = labelled_test_ds.batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        serial_num = ''

        for index in range(0, 7):
            model = tf.keras.models.load_model(f"model_digit{index}")
            prediction = self.models[f"model_{index}"].predict(batch_test_ds)
            predict_batch = np.argmax(prediction,axis=1)
            predict_batch = list(predict_batch)
            predict_batch = [str(x) for x in predict_batch]
            serial_num += predict_batch[0]

        print(serial_num)
        return serial_num

predictor = PCBSerial()
serial = predictor.identify("/Users/ameyakunder/pbv3_imagerec/pbv3_compvision/drive-download/IMG_3465.JPG")
