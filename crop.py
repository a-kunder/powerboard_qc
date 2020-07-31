#!/usr/bin/env python

import argparse
import numpy as np
import cv2


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
            if len(m)==1: continue
            if m[0].distance < 0.7*m[1].distance:
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
    parser.add_argument('input' ,help='Path to image of a Powerboard')
    parser.add_argument('output',help='Path where the result will be saved')

    args = parser.parse_args()

    iimg = cv2.imread(args.input)

    cv2.imshow("test",iimg)
    cv2.imwrite(args.output, iimg)
    cv2.waitKey(0)
    
    trf=PBv3Matcher('data/golden.jpg')
    timg=trf.transform(iimg)

    cimg=timg[260:260+560,750:750+625]
    cv2.imshow("test",cimg)
    cv2.waitKey(0)

    cv2.imwrite(args.output, cimg)
