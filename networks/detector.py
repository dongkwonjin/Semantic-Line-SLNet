import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np

class FeatureExtraction(nn.Module):
    def __init__(self, feature_extraction_cnn='vgg16'):
        super(FeatureExtraction, self).__init__()

        if feature_extraction_cnn == 'vgg16':
            model = models.vgg16(pretrained=True)

            vgg_feature_layers = ['conv1_1', 'relu1_1', 'conv1_2', 'relu1_2',
                                   'pool1', 'conv2_1', 'relu2_1', 'conv2_2',
                                   'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
                                   'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3',
                                   'pool3', 'conv4_1', 'relu4_1', 'conv4_2',
                                   'relu4_2', 'conv4_3', 'relu4_3']

            last_layer = 'relu4_3'
            last_layer_idx = vgg_feature_layers.index(last_layer)

            self.model1 = nn.Sequential(*list(model.features.children())[:last_layer_idx+1])
            self.model2 = nn.Sequential(*list(model.features.children())[last_layer_idx+1:-1])

    def forward(self, img):
        feat1 = self.model1(img)
        feat2 = self.model2(feat1)

        return feat1, feat2


class Line_Pooling_Layer(nn.Module):
    def __init__(self, size, step=49):
        super(Line_Pooling_Layer, self).__init__()

        self.step = step
        self.f_size = int(np.sqrt(self.step))
        self.size = size


    def forward(self, feat_map, line_pts, ratio):

        b = line_pts.shape[1]

        line_pts = line_pts[:, :, :4] / (self.size - 1) * (self.size / ratio - 1)
        line_pts = (line_pts / (self.size / ratio - 1) - 0.5) * 2  # [-1, 1]

        grid_X = line_pts[:, :, [0, 2]]  # Width
        grid_Y = line_pts[:, :, [1, 3]]  # Height

        line_X = F.interpolate(grid_X, self.step, mode='linear', align_corners=True)[0]
        line_Y = F.interpolate(grid_Y, self.step, mode='linear', align_corners=True)[0]

        line_X = line_X.view(line_X.size(0), self.f_size, self.f_size, 1)
        line_Y = line_Y.view(line_Y.size(0), self.f_size, self.f_size, 1)
        line_grid = torch.cat((line_X, line_Y), dim=3)

        _, c, h, w = feat_map.shape
        feat = feat_map.expand(b, c, h, w)

        f_lp = F.grid_sample(feat, line_grid)

        return f_lp

class Fully_connected_layer(nn.Module):
    def __init__(self):
        super(Fully_connected_layer, self).__init__()

        self.linear_1 = nn.Linear(7 * 7 * 1024, 1024)
        self.linear_2 = nn.Linear(1024, 1024)


    def forward(self, x):
        x = x.view(x.size(0), -1)

        fc1 = self.linear_1(x)
        fc1 = F.relu(fc1)
        fc2 = self.linear_2(fc1)
        fc2 = F.relu(fc2)

        return fc1, fc2

class Classification(nn.Module):
    def __init__(self):
        super(Classification, self).__init__()

        self.linear = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.linear(x)
        x = F.softmax(x, dim=1)

        return x

class Regression(nn.Module):
    def __init__(self):
        super(Regression, self).__init__()

        self.linear = nn.Linear(1024, 4)

    def forward(self, x):
        x = self.linear(x)

        return x

class SLNet(nn.Module):
    def __init__(self, cfg):
        super(SLNet, self).__init__()

        self.feature_extraction = FeatureExtraction()

        self.fully_connected = Fully_connected_layer()
        self.regression = Regression()
        self.classification = Classification()

        size = torch.FloatTensor(cfg.size).cuda()
        self.line_pooling = Line_Pooling_Layer(size=size)


    def forward(self, img, line_pts, feat1=None, feat2=None):
        # Feature extraction
        if feat1 is None:
            feat1, feat2 = self.feature_extraction(img)  # 512,50,50 // 512,25,25

        # Line pooling
        lp1 = self.line_pooling(feat1, line_pts, 8)
        lp2 = self.line_pooling(feat2, line_pts, 16)

        lp_concat = torch.cat((lp1, lp2), dim=1)
        fc_out1, fc_out2 = self.fully_connected(lp_concat)  #

        # Classification & Regression
        reg_out = self.regression(fc_out2)
        cls_out = self.classification(fc_out2)

        return {'reg': reg_out, 'cls': cls_out}


