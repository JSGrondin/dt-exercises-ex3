#!/usr/bin/env python3

import json
import os
import cv2
import numpy as np
from skimage.io import imsave

DATASET_PATH="/home/js/PycharmProjects/dt-exercises/object_detection_ex3/dataset"
YOLO_DATASET_PATH="/home/js/PycharmProjects/dt-exercises/object_detection_ex3/yolo_dataset"

if not os.path.exists(os.path.join(YOLO_DATASET_PATH, "images")):
    os.makedirs(os.path.join(YOLO_DATASET_PATH, "images"))

if not os.path.exists(os.path.join(YOLO_DATASET_PATH, "images", "train")):
    os.makedirs(os.path.join(YOLO_DATASET_PATH, "images", "train"))

if not os.path.exists(os.path.join(YOLO_DATASET_PATH, "images", "val")):
    os.makedirs(os.path.join(YOLO_DATASET_PATH, "images", "val"))

if not os.path.exists(os.path.join(YOLO_DATASET_PATH, "labels")):
    os.makedirs(os.path.join(YOLO_DATASET_PATH, "labels"))

if not os.path.exists(os.path.join(YOLO_DATASET_PATH, "labels", "train")):
    os.makedirs(os.path.join(YOLO_DATASET_PATH, "labels", "train"))

if not os.path.exists(os.path.join(YOLO_DATASET_PATH, "labels", "val")):
    os.makedirs(os.path.join(YOLO_DATASET_PATH, "labels", "val"))

npz_index = 0
while os.path.exists(f"{DATASET_PATH}/{npz_index}.npz"):

    with np.load(f"{DATASET_PATH}/{npz_index}.npz") as data:
        img = data['arr_0']
        boxes = data['arr_1']
        classes = data['arr_2']

    if npz_index % 10 == 0:
        split = 'val'
    else:
        split = 'train'

    imsave(f"{YOLO_DATASET_PATH}/images/{split}/{npz_index}.png", img)

    with open(os.path.join(YOLO_DATASET_PATH, "labels", split, str(npz_index) + '.txt'), 'w') as yolo_file:
        for i, object in enumerate(boxes):

            xmin = object[0]
            ymin = object[1]
            xmax = object[2]
            ymax = object[3]

            # converting coordinates to yolo format (i.e. box center coordinate with width and height) and
            # normalize values between 0 and 1
            xcenter = str(((float(xmax) + float(xmin)) / 2) / float(224))
            ycenter = str(((float(ymax) + float(ymin)) / 2) / float(224))
            obj_width = str((float(xmax) - float(xmin)) / float(224))
            obj_height = str((float(ymax) - float(ymin)) / float(224))
            yolo_file.write(str(classes[i]) + ' ' + xcenter + ' ' + ycenter +
                            ' ' + obj_width + ' ' + obj_height)
            yolo_file.write("\n")

        npz_index += 1