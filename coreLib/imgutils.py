# -*-coding: utf-8 -
'''
    @author: (Re-Moduling) MD. Nazmuddoha Ansary 
'''
#--------------------
# imports
#--------------------
import os
import io
import typing

import cv2
import imgaug
import numpy as np
import validators
import matplotlib.pyplot as plt
from shapely import geometry
from scipy import spatial
#--------------------
# helpers:
#--------------------


def get_rotated_box(points): 
    """Obtain the parameters of a rotated box.
     -> typing.Tuple[typing.Tuple[float, float], typing.Tuple[float, float], typing.Tuple[float, float], typing.Tuple[float, float], float]
    Returns:
        The vertices of the rotated box in top-left,
        top-right, bottom-right, bottom-left order along
        with the angle of rotation about the bottom left corner.
        #https://github.com/faustomorales/keras-ocr/blob/5c87abddcf44ccfce01c74d036d498cf8f2bd18d/keras_ocr/tools.py#L458
    """
    try:
        mp = geometry.MultiPoint(points=points)
        pts = np.array(list(zip(*mp.minimum_rotated_rectangle.exterior.xy)))[:-1]  # noqa: E501
    except AttributeError:
        # There weren't enough points for the minimum rotated rectangle function
        pts = points
    # The code below is taken from
    # https://github.com/jrosebr1/imutils/blob/master/imutils/perspective.py

    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = spatial.distance.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    pts = np.array([tl, tr, br, bl], dtype="float32")

    rotation = np.arctan((tl[0] - bl[0]) / (tl[1] - bl[1]))
    return pts, rotation


def fix_line(line):
    """Given a list of (box, character) tuples, return a revised
    line with a consistent ordering of left-to-right or top-to-bottom,
    with each box provided with (top-left, top-right, bottom-right, bottom-left)
    ordering.
    Returns:
        A tuple that is the fixed line as well as a string indicating
        whether the line is horizontal or vertical.
    """
    line = [(get_rotated_box(box)[0], character) for box, character in line]
    centers = np.array([box.mean(axis=0) for box, _ in line])
    sortedx = centers[:, 0].argsort()
    sortedy = centers[:, 1].argsort()
    if np.diff(centers[sortedy][:, 1]).sum() > np.diff(centers[sortedx][:, 0]).sum():
        return [line[idx] for idx in sortedy], 'vertical'
    return [line[idx] for idx in sortedx], 'horizontal'

#--------------------
# ops:https://github.com/faustomorales/keras-ocr/blob/5c87abddcf44ccfce01c74d036d498cf8f2bd18d/keras_ocr/detection.py
#--------------------


def compute_input(image):
    # should be RGB order
    image = image.astype('float32')
    mean = np.array([0.485, 0.456, 0.406])
    variance = np.array([0.229, 0.224, 0.225])

    image -= mean * 255
    image /= variance * 255
    return image


def invert_input(X):
    X = X.copy()
    mean = np.array([0.485, 0.456, 0.406])
    variance = np.array([0.229, 0.224, 0.225])

    X *= variance * 255
    X += mean * 255
    return X.clip(0, 255).astype('uint8')


def get_gaussian_heatmap(size=512, distanceRatio=3.34):
    v = np.abs(np.linspace(-size / 2, size / 2, num=size))
    x, y = np.meshgrid(v, v)
    g = np.sqrt(x**2 + y**2)
    g *= distanceRatio / (size / 2)
    g = np.exp(-(1 / 2) * (g**2))
    g *= 255
    return g.clip(0, 255).astype('uint8')




def compute_maps(heatmap, image_height, image_width, lines):
    assert image_height % 2 == 0, 'Height must be an even number'
    assert image_width % 2 == 0, 'Width must be an even number'

    textmap = np.zeros((image_height // 2, image_width // 2)).astype('float32')
    linkmap = np.zeros((image_height // 2, image_width // 2)).astype('float32')

    src = np.array([[0, 0], [heatmap.shape[1], 0], [heatmap.shape[1], heatmap.shape[0]],
                    [0, heatmap.shape[0]]]).astype('float32')

    for line in lines:
        line, orientation = fix_line(line)
        previous_link_points = None
        for [(x1, y1), (x2, y2), (x3, y3), (x4, y4)], c in line:
            x1, y1, x2, y2, x3, y3, x4, y4 = map(lambda v: max(v, 0),
                                                 [x1, y1, x2, y2, x3, y3, x4, y4])
            if c == ' ':
                previous_link_points = None
                continue
            yc = (y4 + y1 + y3 + y2) / 4
            xc = (x1 + x2 + x3 + x4) / 4
            if orientation == 'horizontal':
                current_link_points = np.array([[
                    (xc + (x1 + x2) / 2) / 2, (yc + (y1 + y2) / 2) / 2
                ], [(xc + (x3 + x4) / 2) / 2, (yc + (y3 + y4) / 2) / 2]]) / 2
            else:
                current_link_points = np.array([[
                    (xc + (x1 + x4) / 2) / 2, (yc + (y1 + y4) / 2) / 2
                ], [(xc + (x2 + x3) / 2) / 2, (yc + (y2 + y3) / 2) / 2]]) / 2
            character_points = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]
                                         ]).astype('float32') / 2
            # pylint: disable=unsubscriptable-object
            if previous_link_points is not None:
                if orientation == 'horizontal':
                    link_points = np.array([
                        previous_link_points[0], current_link_points[0], current_link_points[1],
                        previous_link_points[1]
                    ])
                else:
                    link_points = np.array([
                        previous_link_points[0], previous_link_points[1], current_link_points[1],
                        current_link_points[0]
                    ])
                ML = cv2.getPerspectiveTransform(
                    src=src,
                    dst=link_points.astype('float32'),
                )
                linkmap += cv2.warpPerspective(heatmap,
                                               ML,
                                               dsize=(linkmap.shape[1],
                                                      linkmap.shape[0])).astype('float32')
            MA = cv2.getPerspectiveTransform(
                src=src,
                dst=character_points,
            )
            textmap += cv2.warpPerspective(heatmap, MA, dsize=(textmap.shape[1],
                                                               textmap.shape[0])).astype('float32')
            # pylint: enable=unsubscriptable-object
            previous_link_points = current_link_points
    return np.concatenate([textmap[..., np.newaxis], linkmap[..., np.newaxis]], axis=2).clip(0, 255) / 255


def map_to_rgb(y):
    return (np.concatenate([y, np.zeros(
        (y.shape[0], y.shape[1], 1))], axis=-1) * 255).astype('uint8')


def getBoxes(y_pred,
             detection_threshold=0.7,
             text_threshold=0.4,
             link_threshold=0.4,
             size_threshold=10):
    box_groups = []
    for y_pred_cur in y_pred:
        # Prepare data
        textmap = y_pred_cur[..., 0].copy()
        linkmap = y_pred_cur[..., 1].copy()
        img_h, img_w = textmap.shape

        _, text_score = cv2.threshold(textmap,
                                      thresh=text_threshold,
                                      maxval=1,
                                      type=cv2.THRESH_BINARY)
        _, link_score = cv2.threshold(linkmap,
                                      thresh=link_threshold,
                                      maxval=1,
                                      type=cv2.THRESH_BINARY)
        n_components, labels, stats, _ = cv2.connectedComponentsWithStats(np.clip(
            text_score + link_score, 0, 1).astype('uint8'),
                                                                          connectivity=4)
        boxes = []
        for component_id in range(1, n_components):
            # Filter by size
            size = stats[component_id, cv2.CC_STAT_AREA]

            if size < size_threshold:
                continue

            # If the maximum value within this connected component is less than
            # text threshold, we skip it.
            if np.max(textmap[labels == component_id]) < detection_threshold:
                continue

            # Make segmentation map. It is 255 where we find text, 0 otherwise.
            segmap = np.zeros_like(textmap)
            segmap[labels == component_id] = 255
            segmap[np.logical_and(link_score, text_score)] = 0
            x, y, w, h = [
                stats[component_id, key] for key in
                [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]
            ]

            # Expand the elements of the segmentation map
            niter = int(np.sqrt(size * min(w, h) / (w * h)) * 2)
            sx, sy = max(x - niter, 0), max(y - niter, 0)
            ex, ey = min(x + w + niter + 1, img_w), min(y + h + niter + 1, img_h)
            segmap[sy:ey, sx:ex] = cv2.dilate(
                segmap[sy:ey, sx:ex],
                cv2.getStructuringElement(cv2.MORPH_RECT, (1 + niter, 1 + niter)))

            # Make rotated box from contour
            contours = cv2.findContours(segmap.astype('uint8'),
                                        mode=cv2.RETR_TREE,
                                        method=cv2.CHAIN_APPROX_SIMPLE)[-2]
            contour = contours[0]
            box = cv2.boxPoints(cv2.minAreaRect(contour))

            # Check to see if we have a diamond
            w, h = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
            box_ratio = max(w, h) / (min(w, h) + 1e-5)
            if abs(1 - box_ratio) <= 0.1:
                l, r = contour[:, 0, 0].min(), contour[:, 0, 0].max()
                t, b = contour[:, 0, 1].min(), contour[:, 0, 1].max()
                box = np.array([[l, t], [r, t], [r, b], [l, b]], dtype=np.float32)
            else:
                # Make clock-wise order
                box = np.array(np.roll(box, 4 - box.sum(axis=1).argmin(), 0))
            boxes.append(2 * box)
        box_groups.append(np.array(boxes))
    return box_groups

