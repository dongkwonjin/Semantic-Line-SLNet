import os
import cv2

import numpy as np

from libs.utils import *

class Visualize(object):

    def __init__(self, cfg):

        self.cfg = cfg

        self.mean = np.array([cfg.mean], dtype=np.float32)
        self.std = np.array([cfg.std], dtype=np.float32)

        self.line = np.zeros((cfg.height, 3, 3), dtype=np.uint8)
        self.line[:, :, :] = 255

        self.show = {}

    def display_candidates(self, data):
        '''
        :param data: candidate lines
        :return:
        '''

        img_1 = np.zeros((self.cfg.height, self.cfg.width, 3), dtype=np.uint8)
        line = np.zeros((img_1.shape[0], 3, 3), dtype=np.uint8)
        line[:, :, :] = 255

        for i in range(data.shape[0]):
            pt_1 = (int(data[i, 0]), int(data[i, 1]))
            pt_2 = (int(data[i, 2]), int(data[i, 3]))
            img_1 = cv2.line(img_1, pt_1, pt_2, (255, 0, 0), 1)

        file_path = self.cfg.dir['out'] + 'display/' + 'candidate_line.jpg'
        mkdir(self.cfg.dir['out'] + 'display/')
        cv2.imwrite(file_path, img_1)

    def display_positive(self, img, data, img_name):

        img_1 = np.ascontiguousarray(np.copy(img))

        for k in range(len(data['pos_line']['endpts'])):
            pos_line = data['pos_line']['endpts'][k]
            for i in range(pos_line.shape[0]):
                pt_1 = (int(pos_line[i, 0]), int(pos_line[i, 1]))
                pt_2 = (int(pos_line[i, 2]), int(pos_line[i, 3]))
                img_1 = cv2.line(img_1, pt_1, pt_2, (255, 0, 0), 1)

        line = np.zeros((img.shape[0], 3, 3), dtype=np.uint8)
        line[:, :, :] = 255

        mkdir(self.cfg.dir['out'] + 'display/pos/')
        cv2.imwrite(os.path.join(self.cfg.dir['out'], 'display/pos', img_name), img_1)

    def display_negative(self, img, data, img_name):

        img_1 = np.ascontiguousarray(np.copy(img))

        for i in range(data['neg_line']['endpts'].shape[0]):
            pt_1 = (int(data['neg_line']['endpts'][i, 0]), int(data['neg_line']['endpts'][i, 1]))
            pt_2 = (int(data['neg_line']['endpts'][i, 2]), int(data['neg_line']['endpts'][i, 3]))
            img_1 = cv2.line(img_1, pt_1, pt_2, (255, 0, 0), 1)

        line = np.zeros((img.shape[0], 3, 3), dtype=np.uint8)
        line[:, :, :] = 255

        mkdir(self.cfg.dir['out'] + 'display/neg/')
        cv2.imwrite(os.path.join(self.cfg.dir['out'], 'display/neg', img_name), img_1)
