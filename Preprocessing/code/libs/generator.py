import cv2

import numpy as np

from libs.utils import *
from libs.modules import *

class generate_line(object):

    def __init__(self, cfg, visualize):

        self.cfg = cfg
        self.visualize = visualize
        self.width = cfg.width
        self.height = cfg.height

        # outlier threshold
        self.thresd_l = 150  # threshold of length
        self.thresd_b = 30  # threshold of boundary
        self.interval = 15  # candidate interval 20 pixel
        self.pos_interval = 10

        # image boundary coordinates
        grid_X, grid_Y = np.linspace(0, self.width - 1, self.width), np.linspace(0, self.height - 1, self.height)
        grid_X = np.expand_dims(grid_X, axis=1)  # W
        grid_Y = np.expand_dims(grid_Y, axis=1)  # H
        self.sample_num = grid_X.shape[0]

        temp_H1 = np.zeros((self.height, 1), dtype=np.float32)
        temp_H2 = np.zeros((self.height, 1), dtype=np.float32)
        temp_W1 = np.zeros((self.width, 1), dtype=np.float32)
        temp_W2 = np.zeros((self.width, 1), dtype=np.float32)
        temp_H2[:] = self.width - 1
        temp_W2[:] = self.height - 1

        self.line = {}
        self.line[0] = np.concatenate((grid_X, temp_W1), axis=1)
        self.line[1] = np.concatenate((temp_H1, grid_Y), axis=1)
        self.line[2] = np.concatenate((grid_X, temp_W2), axis=1)
        self.line[3] = np.concatenate((temp_H2, grid_Y), axis=1)

        self.dx = np.array([1, 0, -1, 0], dtype=np.float32)
        self.dy = np.array([0, -1, 0, 1], dtype=np.float32)

    def candidate_line(self):

        cand_line = []
        outlier = {'dist': [], 'bound': [], 'vertical': [], 'positive': []}

        for i in range(4):
            for j in range(i+1, 4):

                # select two side in image boundary
                ref_1 = self.line[i]
                ref_2 = self.line[j]

                for p1 in range(1, self.height - 1, self.interval):
                    for p2 in range(1, self.width - 1, self.interval):

                        # select two endpoints in two sides respectively
                        pt_1 = ref_1[p1]
                        pt_2 = ref_2[p2]

                        endpts = np.concatenate((pt_1, pt_2), axis=0)
                        check = candidate_line_filtering(pts=endpts,
                                                         size=(self.width, self.height),
                                                         thresd_boundary=self.thresd_b,
                                                         thresd_length=self.thresd_l)

                        if check == 0:
                            cand_line.append(np.expand_dims(endpts, axis=0))

        cand_line = np.float32(np.concatenate(cand_line))
        # visualize
        self.visualize.display_candidates(cand_line)
        self.cand_line = cand_line
        return cand_line

    def negative_line(self, gt_line, gtimg=None, img_name=None):

        # not include in positive line
        num, _ = self.cand_line.shape

        neg_check = np.ones((num), dtype=np.int32)

        data = {}
        data['neg_line'] = {}
        data['neg_line']['endpts'] = []
        data['neg_line']['cls'] = []
        data['neg_line']['offset'] = []

        for i in range(num):

            pts = self.cand_line[i]

            # exclude positive line
            for k in range(gt_line.shape[0]):

                d1 = d2 = 99999
                d1 = np.minimum(d1, np.sum(np.abs(pts[:2] - gt_line[k, :2])))
                d2 = np.minimum(d2, np.sum(np.abs(pts[2:4] - gt_line[k, 2:4])))
                d1 = np.minimum(d1, np.sum(np.abs(pts[2:4] - gt_line[k, :2])))
                d2 = np.minimum(d2, np.sum(np.abs(pts[:2] - gt_line[k, 2:4])))

                if d1 <= self.pos_interval and d2 <= self.pos_interval: # positive line
                    neg_check[i] = 0
                    break

        neg_line = self.cand_line[neg_check == 1]

        neg_num = neg_line[:, :4].shape[0]

        data['neg_line']['endpts'] = np.float32(neg_line[:, :4])
        data['neg_line']['offset'] = np.repeat(np.float32(np.array([[0, 0, 0, 0]])), neg_num, axis=0)
        data['neg_line']['cls'] = np.repeat(np.float32(np.array([[1, 0]])), neg_num, axis=0)

        # visualize
        self.visualize.display_negative(gtimg, data, img_name)
        return data


    def positive_line(self, gt_line, gtimg=None, img_name=None):

        num, _ = self.cand_line.shape

        data = {}
        data['pos_line'] = {}
        data['pos_line']['endpts'] = []
        data['pos_line']['cls'] = []
        data['pos_line']['offset'] = []


        for i in range(gt_line.shape[0]):

            pos_line = []
            offset = []

            # visit table
            visit = np.zeros((self.height, self.width), dtype=np.int32)
            visit[1:self.height - 1, 1:self.width - 1] = 1

            # endpoints in side
            list_pt_1 = []
            list_pt_1.append(np.copy(gt_line[i, :2]))
            offset_1 = []
            offset_1.append(np.array([0, 0]))

            x = gt_line[i, 0]
            y = gt_line[i, 1]
            visit[int(y), int(x)] = 1

            for k in range(0, self.pos_interval):
                for l in range(4):
                    if y + self.dy[l] < 0 or x + self.dx[l] < 0 or y + self.dy[l] >= self.height or x + self.dx[l] >= self.width:
                        continue
                    if visit[int(y + self.dy[l]), int(x + self.dx[l])] == 0:
                        x += self.dx[l]
                        y += self.dy[l]
                        visit[int(y), int(x)] = 1
                        break
                list_pt_1.append(np.array([x, y]))
                offset_1.append(np.array([gt_line[i, 0] - x, gt_line[i, 1] - y]))

            x = gt_line[i, 0]
            y = gt_line[i, 1]
            for k in range(0, self.pos_interval):
                for l in range(4):
                    if y + self.dy[l] < 0 or x + self.dx[l] < 0 or y + self.dy[l] >= self.height or x + self.dx[l] >= self.width:
                        continue
                    if visit[int(y + self.dy[l]), int(x + self.dx[l])] == 0:
                        x += self.dx[l]
                        y += self.dy[l]
                        visit[int(y), int(x)] = 1
                        break
                list_pt_1.append(np.array([x, y]))
                offset_1.append(np.array([gt_line[i, 0] - x, gt_line[i, 1] - y]))

            list_pt_1 = np.concatenate([list_pt_1], axis=1)

            visit = np.zeros((self.height, self.width), dtype=np.int32)
            visit[1:self.height - 1, 1:self.width - 1] = 1

            list_pt_2 = []
            list_pt_2.append(np.copy(gt_line[i, 2:4]))
            offset_2 = []
            offset_2.append(np.array([0, 0]))

            x = gt_line[i, 2]
            y = gt_line[i, 3]
            visit[int(y), int(x)] = 1

            for k in range(0, self.pos_interval):
                for l in range(4):
                    if y + self.dy[l] < 0 or x + self.dx[l] < 0 or y + self.dy[l] >= self.height or x + self.dx[l] >= self.width:
                        continue
                    if visit[int(y + self.dy[l]), int(x + self.dx[l])] == 0:
                        x += self.dx[l]
                        y += self.dy[l]
                        visit[int(y), int(x)] = 1
                        break
                list_pt_2.append(np.array([x, y]))
                offset_2.append(np.array([gt_line[i, 2] - x, gt_line[i, 3] - y]))

            x = gt_line[i, 2]
            y = gt_line[i, 3]
            for k in range(0, self.pos_interval):
                for l in range(4):
                    if y + self.dy[l] < 0 or x + self.dx[l] < 0 or y + self.dy[l] >= self.height or x + self.dx[l] >= self.width:
                        continue
                    if visit[int(y + self.dy[l]), int(x + self.dx[l])] == 0:
                        x += self.dx[l]
                        y += self.dy[l]
                        visit[int(y), int(x)] = 1
                        break
                list_pt_2.append(np.array([x, y]))
                offset_2.append(np.array([gt_line[i, 2] - x, gt_line[i, 3] - y]))

            list_pt_2 = np.concatenate([list_pt_2], axis=1)

            for p1 in range(0, self.pos_interval * 2 + 1):
                for p2 in range(0, self.pos_interval * 2 + 1):
                    pos_line.append(np.concatenate((list_pt_1[p1], list_pt_2[p2]), axis=0))
                    offset.append(np.concatenate((offset_1[p1], offset_2[p2]), axis=0))

            pos_num = len(pos_line)

            pos_line = np.float32(pos_line)
            offset = np.float32(offset)
            cls = np.repeat(np.float32(np.array([[0, 1]])), pos_num, axis=0)

            data['pos_line']['endpts'].append(pos_line)
            data['pos_line']['offset'].append(offset)
            data['pos_line']['cls'].append(cls)


        # visualize
        self.visualize.display_positive(gtimg, data, img_name)

        # dict to array
        if data['pos_line']['endpts'] != []:
            data['pos_line']['endpts'] = np.concatenate(data['pos_line']['endpts'])
            data['pos_line']['offset'] = np.concatenate(data['pos_line']['offset'])
            data['pos_line']['cls'] = np.concatenate(data['pos_line']['cls'])

        return data

    def run(self):
        print('--------------start preprocessing--------------')
        candidates = self.candidate_line()
        save_pickle(dir_name=self.cfg.dir['out'], file_name='detector_test_candidates', data=candidates)

        # load pickle
        with open(self.cfg.dir['dataset']['SEL'] + 'data/train.pickle', 'rb') as f:
            train_data = pickle.load(f)

        result = []
        for i in range(len(train_data['img_path'])):
            print('data %d clear' % i)

            img_path = train_data['img_path'][i]
            gtline = train_data['multiple'][i]
            gtimg = cv2.imread(self.cfg.dir['dataset']['SEL'] + 'gtimg/' + img_path)

            print(img_path)

            # generate positive line and negative line
            pos_label = self.positive_line(gtline, gtimg, img_path)
            neg_label = self.negative_line(gtline, gtimg, img_path)

            candidate_line = dict(pos_label, **neg_label)
            candidate_line['img_path'] = img_path

            result.append(candidate_line)

        # save training candidate line to pickle file
        save_pickle(dir_name=self.cfg.dir['out'], file_name='detector_train_candidates', data=result)
