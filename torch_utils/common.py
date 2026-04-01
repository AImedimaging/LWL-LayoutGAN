import math

import torch
import torchvision
from torch import nn
import torch.nn.functional as F

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = torchvision.models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        _, C, _, _ = X.shape
        if C == 1:
            X = X.repeat(1, 3, 1, 1)
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


# class VGGLoss(torch.nn.Module):
#     def __init__(self):
#         super(VGGLoss, self).__init__()
#         self.vgg = Vgg19().cuda()
#         self.criterion = torch.nn.L1Loss()
#         self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
#
#     def forward(self, x, y):
#         x_vgg, y_vgg = self.vgg(x), self.vgg(y)
#         loss = 0
#         for i in range(len(x_vgg)):
#             loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
#         return loss


class Perceptual_loss134(torch.nn.Module):
    def __init__(self):
        super(Perceptual_loss134, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = torch.nn.MSELoss()
        # self.weights = [1.0/2.6, 1.0/16, 1.0/3.7, 1.0/5.6, 1.0]
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        loss = self.weights[0] * self.criterion(x_vgg[0], y_vgg[0].detach()) + \
               self.weights[2] * self.criterion(x_vgg[2], y_vgg[2].detach()) + \
               self.weights[3] * self.criterion(x_vgg[3], y_vgg[3].detach())
        return loss


class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.MSELoss()
        # self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.weights = [1.0, 1.0, 1.0, 1.0, 1.0]
        self.downsample = nn.AvgPool2d(2, stride=2, count_include_pad=False)

    def forward(self, x, y):
        while x.size()[3] > 4096:
            x, y = self.downsample(x), self.downsample(y)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0.0
        for iter, (x_fea, y_fea) in enumerate(zip(x_vgg, y_vgg)):
            # print(iter + 1, self.criterion(x_fea, y_fea.detach()), x_fea.size())
            loss += self.weights[iter] * self.criterion(x_fea, y_fea.detach())
        return loss

def tensor_shift(seg, shift_x, shift_y):
    h, w = seg.size()[2:]

    # y direction
    if shift_y > 0:
        seg[:, :, shift_y:h] = seg[:, :, :(h - shift_y)].clone()
        seg[:, :, :shift_y] = 0
    else:
        seg[:, :, :(h + shift_y)] = seg[:, :, abs(shift_y):h].clone()
        seg[:, :, (h + shift_y):] = 0

    # x direction
    if shift_x > 0:
        seg[..., shift_x:w] = seg[..., :(w - shift_x)].clone()
        seg[..., :shift_x] = 0
    else:
        seg[..., :(w + shift_x)] = seg[..., abs(shift_x):w].clone()
        seg[..., (w + shift_x):] = 0
    return seg

