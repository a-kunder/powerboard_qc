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

###cropping

ref_point = []

class PBv3Matcher:
    def __init__(self, golden):
        #
        # Settings
        self.MIN_MATCH_COUNT = 10

        #
        # Algorithnms

        # Features
        self.algo_features = cv2.ORB_create()

        # Matching
        FLANN_INDEX_LSH = 6
        index_params = dict(algorithm = FLANN_INDEX_LSH,
                                table_number = 6, # 12
                                key_size = 12,     # 20
                                multi_probe_level = 1) #2
        search_params = dict(checks = 50)

        self.algo_matcher = cv2.FlannBasedMatcher(index_params, search_params)

        #
        # Load reference
        self.load_golden(golden)

    def load_golden(self, golden):
        """
        Load a golden image reference and prepare the
        feature descriptors
        """
        self.golden_img = cv2.imread(golden)
        self.golden_kp, self.golden_des = self.algo_features.detectAndCompute(self.golden_img, None)

    def transform(self, img):
        """
        Match img image and transform it into golden coordinate frames
        """
        #
        #
        # Features and match
        kp, des = self.algo_features.detectAndCompute(img, None)
        matches = self.algo_matcher.knnMatch(self.golden_des, des, k=2)

        print(self.golden_kp)
        # store all the good matches as per Lowe's ratio test
        good = []
        for m in matches:
            if len(m)<=1: continue
            elif m[0].distance < 0.7*m[1].distance:
                good.append(m[0])

        if len(good)<self.MIN_MATCH_COUNT:
            print('Error: Not enough matches')
            return None

        #
        # Transformation
        src_pts = np.float32([ self.golden_kp[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp            [m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        Minv, maskinv = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC,5.0)

        timg=cv2.warpPerspective(img, Minv, (self.golden_img.shape[1],self.golden_img.shape[0]))

        return timg

    def shape_select(event, x, y, flags, param):
        global ref_point

        if event == cv2.EVENT_LBUTTONDOWN:
            ref_point = [(x, y)]

        elif event == cv2.EVENT_LBUTTONUP:
            ref_point.append((x, y))

        cv2.rectangle(iimg, ref_point[0], ref_point[1], (0, 0, 0), 2)
        cv2.imshow("image", iimg)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Crop a sheildbox from a picture of a Powerboard')
    parser.add_argument('input', help='Path to images of Powerboards')

    args = parser.parse_args()
    if not os.path.exists('main_output'):
        os.mkdir('main_output')

    for x in glob.glob(f"{args.input}/*JPG"):

        iimg = cv2.imread(x)

        '''cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('test', 900, 600)
        cv2.imshow('test', iimg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()'''

        #serial = input("Enter the serial number: ")
        n =  random.randint(0, 1000000)
        cv2.imwrite(f"main_output/{n}.JPG", iimg)
        
        trf=PBv3Matcher('data/golden.jpg')
        timg=trf.transform(iimg)

        if timg is None:

            #reads and resizes the image
            (h, w, d) = iimg.shape
            r = 300.0 / w
            dim = (300, int(h * r))
            iimg = cv2.resize(iimg, dim)

            clone = iimg.copy()
            cv2.namedWindow("image")
            cv2.moveWindow("image", 20, 20)
            cv2.setMouseCallback("image", PBv3Matcher.shape_select)

            while True:
                # display the image and wait for a keypress
                cv2.imshow("image", iimg)
                key = cv2.waitKey(1) & 0xFF

                # if the 'r' key is pressed, reset the cropping region
                if key == ord("r"):
                    iimg = clone.copy()

                # if the 'c' key is pressed, break from the loop
                elif key == ord("c"):
                    break

            if len(ref_point) == 2:
                crop_img = clone[ref_point[0][1]:ref_point[1][1], ref_point[0][0]:ref_point[1][0]]
                crop_img = cv2.resize(crop_img, (600, 500))
                cv2.imshow("crop_img", crop_img)
                cv2.waitKey(0)

            write_add = f"{args.output}/{n}.JPG"
            cv2.imwrite(write_add, crop_img)
            cv2.destroyAllWindows()

        else:
            cimg=timg[260:260+560,750:750+625]
            #cv2.imshow("test",cimg)
            #cv2.waitKey(0)
            write_add = f"main_output/{n}.JPG"
            cv2.imwrite(write_add, cimg)


###thresholding

    for x in glob.glob(f"main_output/*JPG"):
        image = cv2.imread(x)
        grayscaled = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        _,threshold = cv2.threshold(grayscaled,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imwrite(x, threshold)

        #cv2.imshow('Otsu threshold',threshold)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()


###testing

    # Load and label the dataset
    # replaced args.digit with added argument to function -> index
    def get_label(file_path):
        file_name=tf.strings.split(file_path, os.path.sep)[-1]
        digit=tf.strings.substr(file_name, index, 1)
        return tf.strings.to_number(digit)

    def process_path(file_path):
        #label = get_label(file_path, index)
        # load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = tf.image.decode_png(img, channels=1)
        img = tf.image.resize(img, (80, 80))
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = img/255
        return img

    # Prepare data for training
    def show_batch(image_batch, label_batch):
        plt.figure(figsize=(10,10))
        for n in range(16):
            if n>=image_batch.shape[0]:
                continue
            ax = plt.subplot(4,4,n+1)
            plt.imshow(image_batch[n][:,:,0])
            plt.title(label_batch[n])
            plt.axis('off')
        plt.show()

    
    serial_num =[]

    for index in range(0, 2):
        test_ds = tf.data.Dataset.list_files('{}/*.JPG'.format('main_output'))
        labelled_test_ds  = test_ds .map(process_path, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        batch_test_ds  = labelled_test_ds.batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        new_model = tf.keras.models.load_model('model_digit{}'.format(index))
        new_model.summary()

        #Test

        #new_test_ds = tfds.as_numpy(test_ds)
        #print(f"Type of new_test_ds: {type(new_test_ds)}")
        #print('--------------------------------')
        #new_model.evaluate(new_test_ds , verbose=2)
        
        prediction = new_model.predict(batch_test_ds)
        predict_batch = np.argmax(prediction,axis=1)
        predict_batch = list(predict_batch)
        if serial_num:
            for i in range(len(serial_num)):
                serial_num[i] = str(serial_num[i]) + predict_batch[i]
        else:
            serial_num += predict_batch

    show_batch(image_batch, serial_num)

