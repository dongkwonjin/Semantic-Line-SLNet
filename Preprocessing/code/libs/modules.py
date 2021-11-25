import math
import cv2
import torch

import numpy as np

def candidate_line_filtering(pts, size, thresd_boundary, thresd_length):
    ## exclude outlier -> short length, boundary
    check = 0

    pt_1 = pts[:2]
    pt_2 = pts[2:]

    length = np.sqrt(np.sum(np.square(pt_1 - pt_2)))

    # short length
    if length < thresd_length:
        check += 1

    # boundary
    if (pt_1[0] < thresd_boundary and pt_2[0] < thresd_boundary):
        check += 1
    if (pt_1[1] < thresd_boundary and pt_2[1] < thresd_boundary):
        check += 1
    if (abs(pt_1[0] - size[0]) < thresd_boundary and abs(pt_2[0] - size[0]) < thresd_boundary):
        check += 1
    if (abs(pt_1[1] - size[1]) < thresd_boundary and abs(pt_2[1] - size[1]) < thresd_boundary):
        check += 1
    return check

