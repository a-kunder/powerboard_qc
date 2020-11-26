#!/usr/bin/env python

import argparse
import numpy as np
import cv2
import glob
import random

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
            elif m[0].distance < 0.8*m[1].distance:
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

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Crop a sheildbox from a picture of a Powerboard')
    parser.add_argument('input' ,help='Path to images of Powerboards')
    parser.add_argument('output',help='Path where the results will be saved')
    args = parser.parse_args()

    read_path = f"{args.input}/*.JPG"
    for x in glob.glob(read_path):
        iimg = cv2.imread(x)
        img_name = x.split('/')[1]
        print(img_name)
        trf=PBv3Matcher('data/golden.jpg')
        timg=trf.transform(iimg)

        if timg is None:
            print('Cropping matches insufficient')

        else:
            cimg=timg[260:260+560,750:750+625]
            cv2.imshow("timg", timg)
            cv2.waitKey(0)
            cv2,destroyAllWindows()
            write_add = f"{args.output}/{img_name}"
            cv2.imwrite(write_add, cimg)
