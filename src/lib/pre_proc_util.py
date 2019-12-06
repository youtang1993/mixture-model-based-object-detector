import cv2
import random
import numpy as np
from . import util as lib_util


def resize(img, boxes, size):
    shape = img.shape
    img = cv2.resize(img, (size[1], size[0]))

    boxes[:, [0, 2]] *= (size[1] / shape[1])
    boxes[:, [1, 3]] *= (size[0] / shape[0])
    return img, boxes


def rand_brightness(img, lower=-32, upper=32):
    if random.randint(0, 1) == 0:
        delta = random.uniform(lower, upper)
        img += delta
    return img


def rand_contrast(img, lower=0.5, upper=1.5):
    if random.randint(0, 1) == 0:
        alpha = random.uniform(lower, upper)
        img *= alpha
    return img


def expand(img, boxes, ratio_range=(1, 4)):
    if np.random.randint(2):
        return img, boxes
    else:
        h, w, c = img.shape
        ratio = np.random.uniform(*ratio_range)
        left = int(np.random.uniform(0, w * ratio - w))
        top = int(np.random.uniform(0, h * ratio - h))

        expand_h, expand_w = int(h * ratio), int(w * ratio)
        expand_img = np.zeros((expand_h, expand_w, c), dtype=img.dtype)
        expand_img[:, :, :] = np.mean(img, axis=(0, 1))
        expand_img[top:top + h, left:left + w] = img
        img = expand_img

        boxes = boxes.copy()
        boxes[:, 0:2] += (left, top)
        boxes[:, 2:4] += (left, top)
        return img, boxes


def rand_crop(
        img, boxes, labels, min_scale=0.3,
        iou_opts=(0.0, 0.1, 0.3, 0.7, 0.9, 1.0)):

    h, w, _ = img.shape
    while True:
        min_iou = np.random.choice(iou_opts)
        if min_iou >= 1.0:
            return img, boxes, labels
        else:
            max_iou = float('inf')

            _w = int(np.random.uniform(min_scale * w, w))
            _h = int(np.random.uniform(min_scale * h, h))
            if _h/_w < 0.5 or h/w > 2:
                continue
            left = int(np.random.uniform(w - _w))
            top = int(np.random.uniform(h - _h))

            rect = np.array([left, top, left+_w, top+_h])
            if len(boxes):
                overlap = lib_util.calc_jaccard_numpy(boxes, rect)
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                croped_img = img[rect[1]:rect[3], rect[0]:rect[2], :]
                centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2.0

                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                mask = m1 * m2

                if not mask.any():
                    continue

                cur_boxes = boxes[mask, :].copy()
                cur_boxes[:, 0:2] = np.maximum(cur_boxes[:, 0:2], rect[0:2])
                cur_boxes[:, 0:2] -= rect[0:2]
                cur_boxes[:, 2:4] = np.minimum(cur_boxes[:, 2:4], rect[2:4])
                cur_boxes[:, 2:4] -= rect[0:2]
                cur_labels = labels[mask]
                # import cv2
                # print(min_iou)
                # cv2.imshow('img', img)
                # cv2.imshow('crop_img', img)
                return croped_img, cur_boxes, cur_labels
            else:
                return img, boxes, labels


def rand_flip(img, boxes):
    _, w, _ = img.shape
    if np.random.randint(2):
        img = img[:, ::-1]
        boxes[:, 0::2] = w - boxes[:, 2::-2]
    return img, boxes
