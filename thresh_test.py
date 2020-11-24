#!/usr/bin/env python
import cv2
import glob
import random
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Crop a sheildbox from a picture of a Powerboard')
parser.add_argument('input' ,help='Path to images of Powerboards')
parser.add_argument('output',help='Path where the results will be saved')

args = parser.parse_args()

read_path = f"{args.input}/*.JPG"

for x in glob.glob(read_path):

    image = cv2.imread(x)
    (b, g, r) = cv2.split(image)
    lst = [b, g, r]
    channels = ['b', 'g', 'r']
    images = []
    for c in lst:
        scale_percent = 70 # percent of original size
        width = int(c.shape[1] * scale_percent / 100)
        height = int(c.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        image = cv2.resize(c, dim, interpolation = cv2.INTER_AREA)
        #_, image = cv2.threshold(b,125,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        images.append(image)

    #for i in range(len(images)):
        #cv2.imshow(channels[i] + "_thresh", images[i])

    _, b_thresh = cv2.threshold(images[0],125,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    _, g_thresh = cv2.threshold(images[1],125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, r_thresh = cv2.threshold(images[2],125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    print(r_thresh)
    print('b_thresh:')
    print(b_thresh)
    rb = np.bitwise_or(r_thresh, b_thresh)

    #grayscaled = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    #_, threshold = cv2.threshold(grayscaled, 10, 255, cv2.THRESH_BINARY)
    #th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    #_,threshold2 = cv2.threshold(grayscaled,125,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    thresh = [b_thresh, g_thresh, r_thresh]

    #cv2.imshow('Otsu threshold',threshold2)
    #cv2.imshow("Thresh image", th)
    #cv2.imshow("OG image", image)
    
    for i in range(len(images)):
        cv2.imshow(channels[i] + "_thresh", thresh[i])

    cv2.imshow("rb_thresh", rb)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #n = random.randint(0, 1000000)
    #write_add = f"{args.output}/{n}.JPG"
    #cv2.imwrite(write_add, b_thresh)
