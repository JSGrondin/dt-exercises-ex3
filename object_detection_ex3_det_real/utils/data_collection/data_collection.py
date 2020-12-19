#!/usr/bin/env python3
import cv2 as cv
import os
import numpy as np
from skimage.io import imsave
from agent import PurePursuitPolicy
from utils import launch_env, seed
from utils import launch_env, seed, makedirs, display_seg_mask, display_img_seg_mask
import imutils

DATASET_DIR="../../dataset"

npz_index = 0
while os.path.exists(f"{DATASET_DIR}/{npz_index}.npz"):
    npz_index += 1

def save_npz(img, boxes, classes):
    global npz_index
    with makedirs(DATASET_DIR):
        np.savez(f"{DATASET_DIR}/{npz_index}.npz", *(img, boxes, classes))
        npz_index += 1


def add_boxes(obs, boxes, classes):
    color_dict = {1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255), 4: (150, 255, 150)}

    for i, val in enumerate(boxes):
        cv.rectangle(obs, (val[0], val[1]), (val[2], val[3]), color_dict[classes[i]], 2)
    return obs

def clean_segmented_image(obs_ss):
    # resizing image to 224x224
    # obs_ss = cv.resize(obs_ss, (224, 224))

    orig_y, orig_x = obs_ss.shape[0], obs_ss.shape[1]
    scale_y, scale_x = 224/orig_y, 224/orig_x

    #converting image to HSV
    obs_ss = cv.cvtColor(obs_ss.copy(), cv.COLOR_RGB2HSV)

    height, width, _ = obs_ss.shape

    color_map = {'white': [0, 0, 255], 'yellow': [30, 255, 255], 'cone': [0, 150, 255],
                 'duckie': [105, 215, 215], 'truck': [0, 0, 125], 'bus': [25, 255, 255], 'background': [0, 0, 0]}

    mask_dict = {}
    mask_dict['duckie']= np.stack(
        ((cv.inRange(obs_ss, np.array([90, 100, 100]), np.array([130, 255, 255])).astype(int)),) * 3, axis=-1)
    mask_dict['bus'] = np.stack(
        ((cv.inRange(obs_ss, np.array([23, 125, 200]), np.array([25, 255, 255])).astype(int)),) * 3, axis=-1)
    mask_dict['truck'] = np.stack(
        ((cv.inRange(obs_ss, np.array([0, 0, 100]), np.array([255, 15, 120])).astype(int)),) * 3, axis=-1)
    mask_dict['cone'] = np.stack(
        ((cv.inRange(obs_ss, np.array([0, 140, 225]), np.array([10, 255, 255])).astype(int)),) * 3, axis=-1)

    obs_ss[:, :] = color_map['background']

    # create image with unified segmentation colors
    obs_ss[(mask_dict['duckie'] == 255).all(-1)] = color_map['duckie']
    obs_ss[(mask_dict['bus'] == 255).all(-1)] = color_map['bus']
    obs_ss[(mask_dict['truck'] == 255).all(-1)] = color_map['truck']
    obs_ss[(mask_dict['cone'] == 255).all(-1)] = color_map['cone']
    obs_ss2 = cv.resize(cv.cvtColor(obs_ss.copy(), cv.COLOR_HSV2RGB), (224, 224))

    # extract bounding boxes for each class
    class_list = ['duckie', 'cone', 'truck', 'bus']

    boxes = []
    classes = []

    for i, val in enumerate(class_list):
        obs_ss[:, :] = color_map['background']
        obs_ss[(mask_dict[str(val)] == 255).all(-1)] = color_map[str(val)]

        #convert to grayscale and blur slightly
        gray = cv.cvtColor(obs_ss.copy(), cv.COLOR_HSV2RGB)
        gray = cv.cvtColor(gray, cv.COLOR_RGB2GRAY)
        gray = cv.GaussianBlur(gray, (7, 7), 0)

        #remove small dots
        thresh = cv.threshold(gray, 85, 255, cv.THRESH_BINARY)[1]

        # perform edge detection, then perform a dilation + erosion to
        # close gaps in between object edges
        edged = cv.Canny(thresh, 50, 100)
        edged = cv.dilate(edged, None, iterations=1)
        edged = cv.erode(edged, None, iterations=1)

        # find contours in the edge map
        cnts = cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        for c in cnts:
            # get the bounding rect
            orig_x_min, orig_y_min, orig_w, orig_h = cv.boundingRect(c)

            # removing snow and other noise
            if orig_w * orig_h <= 120:
                continue

            x_min = int(np.round(orig_x_min * scale_x))
            y_min = int(np.round(orig_y_min * scale_y))
            x_max = x_min + int(np.round(orig_w * scale_x))
            y_max = y_min + int(np.round(orig_h * scale_y))

            boxes.append([x_min,y_min,x_max,y_max])
            classes.append(i+1)

    return obs_ss2, boxes, classes

seed(123)
environment = launch_env()

policy = PurePursuitPolicy(environment)

MAX_STEPS = 9000

while True:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = policy.predict(np.array(obs))

        obs, rew, done, misc = environment.step(action) # Gives non-segmented obs as numpy array
        segmented_obs = environment.render_obs(True)  # Gives segmented obs as numpy array

        clean_seg, boxes, classes = clean_segmented_image(segmented_obs)
        # obs_w_bbox = add_boxes(cv.resize(obs, (224, 224)), boxes, classes)
        # imsave(f"{DATASET_DIR}/{nb_of_steps}.png", clean_seg)
        # imsave(f"{DATASET_DIR}/{nb_of_steps}-bbox.png", obs_w_bbox)

        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

        if len(boxes) == 0:
            continue

        obs = cv.resize(obs, (224, 224))

        save_npz(
            obs,
            np.array(boxes),
            np.array(classes)
        )

        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break

