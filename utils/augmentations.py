import math
import random

import cv2
import numpy as np

from utils.yolo_utils import xywhn2xyxy, xyxy2xywhn


def data_augments(imgs, labels, hyp):
    imgs = imgs if isinstance(imgs, list) else [imgs]
    imgs, labels = random_perspective(imgs,
                                      labels,
                                      degrees=hyp['degrees'],
                                      translate=hyp['translate'],
                                      scale=hyp['scale'],
                                      shear=hyp['shear'],
                                      perspective=hyp['perspective'])

    # HSV color-space
    imgs = augment_hsv(imgs,
                       hgain=hyp['hsv_h'],
                       sgain=hyp['hsv_s'],
                       vgain=hyp['hsv_v'])

    nL = len(labels)  # number of labels
    # Flip up-down
    if random.random() < hyp['flipud']:
        imgs = [np.flipud(img) for img in imgs]
        if nL:
            labels[:, 2] = 1 - labels[:, 2]

    # Flip left-right
    if random.random() < hyp['fliplr']:
        imgs = [np.fliplr(img) for img in imgs]
        if nL:
            labels[:, 1] = 1 - labels[:, 1]

    # toGray
    if random.random() < hyp['toGray']:
        imgs = [
            cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),
                         cv2.COLOR_GRAY2BGR) for img in imgs
        ]

    return imgs, labels


def augment_hsv(imgs, hgain=0.5, sgain=0.5, vgain=0.5):
    # HSV color-space augmentation
    imgs_new = []
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain
                                           ] + 1  # random gains
        for im in imgs:
            hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
            dtype = im.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat),
                                cv2.LUT(val, lut_val)))
            im = cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR)
            imgs_new.append(im)
        return imgs_new
    else:
        return imgs


def random_perspective(imgs,
                       targets=(),
                       degrees=10,
                       translate=.1,
                       scale=.1,
                       shear=10,
                       perspective=0.0,
                       border=(0, 0)):
    # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
    # targets = [cls, xyxy]

    def box_candidates(box1,
                       box2,
                       wh_thr=2,
                       ar_thr=20,
                       area_thr=0.2):  # box1(4,n), box2(4,n)
        # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (
            w2 * h2 /
            (w1 * h1 + 1e-16) > area_thr) & (ar < ar_thr)  # candidates

    imgs_new = []
    img = imgs[0]

    if targets.size:  # normalized xywh to pixel xyxy format
        targets[:, 1:] = xywhn2xyxy(targets[:, 1:],
                                    w=img.shape[1],
                                    h=img.shape[0])

    height = img.shape[0] + border[0] * 2  # shape(h,w,c)
    width = img.shape[1] + border[1] * 2

    # Center
    C = np.eye(3)
    C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective,
                             perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective,
                             perspective)  # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi /
                       180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi /
                       180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate,
                             0.5 + translate) * width  # x translation (pixels)
    T[1, 2] = random.uniform(
        0.5 - translate, 0.5 + translate) * height  # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
    if (border[0] != 0) or (border[1] !=
                            0) or (M != np.eye(3)).any():  # image changed
        for img in imgs:
            if perspective:
                img = cv2.warpPerspective(img,
                                          M,
                                          dsize=(width, height),
                                          borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img,
                                     M[:2],
                                     dsize=(width, height),
                                     borderValue=(114, 114, 114))
            imgs_new.append(img)

    # Transform label coordinates
    n = len(targets)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(
            n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        if perspective:
            xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        else:  # affine
            xy = xy[:, :2].reshape(n, 8)

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate(
            (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=xy.T)
        targets = targets[i]
        targets[:, 1:5] = xy[i]

    if targets.size:
        targets[:, 1:] = xyxy2xywhn(targets[:, 1:],
                                    w=img.shape[1],
                                    h=img.shape[0],
                                    clip=True,
                                    eps=1E-3)

    return imgs_new, targets
