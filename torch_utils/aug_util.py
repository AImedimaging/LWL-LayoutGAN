import random

import PIL.Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from torch_utils.custom_ops import bbox_mask
from torch_utils.ops.grid_sample_gradfix import grid_sample


def x0y0wh2x0y0x1y1(input):
    assert input.ndim == 3
    if isinstance(input, torch.Tensor):
        output = input.clone()
    else:
        output = input.copy()

    output[:, :, 2] = input[:, :, 0] + input[:, :, 2]
    output[:, :, 3] = input[:, :, 1] + input[:, :, 3]
    return output

def x0y0wh2xywh(input):
    assert input.ndim == 3
    if isinstance(input, torch.Tensor):
        output = input.clone()
    else:
        output = input.copy()
    output[:, :, 0] = input[:, :, 0] + input[:, :, 2] / 2
    output[:, :, 1] = input[:, :, 1] + input[:, :, 3] / 2
    return output


def xywh2x0y0wh(input):
    assert input.ndim == 3
    if isinstance(input, torch.Tensor):
        output = input.clone()
    else:
        output = input.copy()
    output[:, :, 0] = input[:, :, 0] - input[:, :, 2] / 2
    output[:, :, 1] = input[:, :, 1] - input[:, :, 3] / 2
    return output


def xywh2x0y0x1y1(input):
    assert input.ndim == 3
    if isinstance(input, torch.Tensor):
        output = input.clone()
    else:
        output = input.copy()
    output[:, :, 1] = input[:, :, 1] - input[:, :, 3] / 2
    output[:, :, 2] = input[:, :, 2] - input[:, :, 4] / 2
    output[:, :, 3] = input[:, :, 1] + input[:, :, 3] / 2
    output[:, :, 4] = input[:, :, 2] + input[:, :, 4] / 2
    return output


def x0y0x1y12xywh(input):
    assert input.ndim == 3
    if isinstance(input, torch.Tensor):
        output = input.clone()
    else:
        output = input.copy()
    output[:, :, 2] = input[:, :, 2] - input[:, :, 0]
    output[:, :, 3] = input[:, :, 3] - input[:, :, 1]
    output[:, :, 0] = (input[:, :, 0] + input[:, :, 2]) / 2
    output[:, :, 1] = (input[:, :, 1] + input[:, :, 3]) / 2
    return output


def HorizontalFlip(image, mask, bbox):
    bbox[:, :, 0] = 1 - bbox[:, :, 0]
    return torch.fliplr(image), torch.fliplr(mask), bbox


def VerticalFlip(image, mask, bbox):
    bbox[:, :, 1] = 1 - bbox[:, :, 1]
    return torch.flipud(image), torch.flipud(mask), bbox


def cropAndresize(image, mask, bbox, size=0.5, locate=(0.25, 0.25)):
    resolution = image.shape[-1]
    if locate == (0, 0):
        locate = (size / 2, size / 2)
    newx0, newx1 = locate[0] - size / 2, locate[0] + size / 2,
    newy0, newy1 = locate[1] - size / 2, locate[1] + size / 2,

    new_x0, new_x1 = int(newx0 * resolution), int(newx1 * resolution)
    new_y0, new_y1 = int(newy0 * resolution), int(newy1 * resolution)

    image = image[:, :, new_y0:new_y1, new_x0:new_x1]
    mask = mask[:, :, new_y0:new_y1, new_x0:new_x1]

    image = F.interpolate(image, size=(resolution, resolution), mode='bilinear', align_corners=True)
    mask = F.interpolate(mask, size=(resolution, resolution), mode='bilinear', align_corners=True)

    if image.ndim == 2:
        image = image[:, :, None]
    if mask.ndim == 2:
        mask = mask[:, :, None]

    bbox = xywh2x0y0x1y1(bbox)
    # # (x0, y0, x1, y1) , (newx0, newy0, newx1, newy1)
    bbox = bbox[(abs(bbox[:, :, 2] + bbox[:, :,  0] - newx0 - newx1) < (bbox[:, :,  2] - bbox[:, :, 0] + newx1 - newx0)) * (
                abs(bbox[:, :, 3] + bbox[:, :, 1] - newy0 - newy1) < (bbox[:, :, 3] - bbox[:, :, 1] + newy1 - newy0))]

    bbox[:, :, 0] = np.where(bbox[:, :, 0] < newx0, newx0, bbox[:, :, 0]) - newx0
    bbox[:, :, 1] = np.where(bbox[:, :, 1] < newy0, newy0, bbox[:, :, 1]) - newy0
    bbox[:, :, 2] = np.where(bbox[:, :, 2] > newx1, newx1, bbox[:, :, 2]) - newx0
    bbox[:, :, 3] = np.where(bbox[:, :, 3] > newy1, newy1, bbox[:, :, 3]) - newy0

    # # relative location
    bbox = bbox / size
    bbox = x0y0x1y12xywh(bbox)

    return image, mask, bbox


def rot90(image, mask, bbox, times: int):

    """
    逆时针90°旋转图像times次，并计算图像image中的坐标点points在旋转后的图像中的位置坐标.
    Args:
        image: 图像数组
        points: [(x, y), ...]，原图像中的坐标点集合
        times: 旋转次数
    """

    if times % 4 == 0:  # 旋转4的倍数次，相当于不旋转
        return image, mask, bbox
    else:
        times = times % 4

    image = torch.rot90(image, times, [2, 3])  # 通过numpy实现图像旋转
    mask = torch.rot90(mask, times, [2, 3])  # 通过numpy实现图像旋转
    new_bbox = bbox.clone()

    if times % 2 == 1:
        new_bbox[:, :, 2], new_bbox[:, :, 3] = bbox[:, :, 3], bbox[:, :, 2]

    if times == 1:
        new_bbox[:, :, 0], new_bbox[:, :, 1] = bbox[:, :, 1], 1 - bbox[:, :, 0]
    elif times == 2:
        new_bbox[:, :, 0], new_bbox[:, :, 1] = 1 - bbox[:, :, 0], 1 - bbox[:, :, 1]
    else:
        new_bbox[:, :, 0], new_bbox[:, :, 1] = 1 - bbox[:, :, 1], bbox[:, :, 0]

    return image, mask, new_bbox


def ReversibleAugment(image, mask, bbox, ops=None):
    if ops is None:
        ops = torch.cat([torch.floor(torch.rand(4) * 2), 0.25 + 0.0625 * (torch.randint(0, 8, [2])), torch.randint(0, 4, [1])]).numpy()

    if ops[0]:
        image, mask, bbox = HorizontalFlip(image, mask, bbox)
    if ops[1]:
        image, mask, bbox = VerticalFlip(image, mask, bbox)
    # if ops[2]:
    #    image, mask, bbox = cropAndresize(image, mask, bbox, locate=(ops[4], ops[5]))
    if ops[3]:
        image, mask, bbox = rot90(image, mask, bbox, times=int(ops[6]))

    return image, mask, bbox, ops



def _boxes_to_grid(boxes, H, W):
    """
    Input:
    - boxes: FloatTensor of shape (O, 4) giving boxes in the [x0, y0, x1, y1]
      format in the [0, 1] coordinate space
    - H, W: Scalars giving size of output
    Returns:
    - grid: FloatTensor of shape (O, H, W, 2) suitable for passing to grid_sample
    """
    O = boxes.size(0)

    boxes = boxes.view(O, 4, 1, 1)

    # All these are (O, 1, 1)
    x0, y0 = boxes[:, 0], boxes[:, 1]
    ww, hh = boxes[:, 2], boxes[:, 3]

    X = torch.linspace(0, 1, steps=W).view(1, 1, W).to(boxes)
    Y = torch.linspace(0, 1, steps=H).view(1, H, 1).to(boxes)

    X = (X - x0) / ww  # (O, 1, W)
    Y = (Y - y0) / hh  # (O, H, 1)

    # Stack does not broadcast its arguments so we need to expand explicitly
    X = X.expand(O, H, W)
    Y = Y.expand(O, H, W)
    grid = torch.stack([X, Y], dim=3)  # (O, H, W, 2)

    # Right now grid is in [0, 1] space; transform to [-1, 1]
    grid = grid.mul(2).sub(1)

    return grid

def masks_to_layout(boxes, masks, H, W=None):
    """
    Inputs:
        - boxes: Tensor of shape (b, num_o, 4) giving bounding boxes in the format
            [x0, y0, x1, y1] in the [0, 1] coordinate space
        - masks: Tensor of shape (b, num_o, M, M) giving binary masks for each object
        - H, W: Size of the output image.
    Returns:
        - out: Tensor of shape (N, num_o, H, W)
    """
    b, num_o, _ = boxes.size()
    M = masks.size(2)
    assert masks.size() == (b, num_o, M, M)
    if W is None:
        W = H

    grid = _boxes_to_grid(boxes.view(b * num_o, -1), H, W).float().to(device=masks.device)

    img_in = masks.float().view(b*num_o, 1, M, M)
    sampled = grid_sample(img_in, grid)

    return sampled.view(b, num_o, H, W)

if __name__ == "__main__":
    bbox = np.loadtxt("../../data/dataset/labels/patch0.txt")[:, 1:]
    bbox = torch.from_numpy(bbox).unsqueeze(0).to(torch.float32)

    bmask = bbox_mask(bbox.device, bbox, 16, 16)
    transforms.ToPILImage()(bmask.sum([0, 1]).clamp(0, 1)).convert('L').save("bmask.png")

    bbox = x0y0wh2x0y0x1y1(bbox)

    grid = _boxes_to_grid(bbox.view(-1, 4), 64, 64)
    mask = masks_to_layout(bbox, bmask, 64, 64)
    transforms.ToPILImage()(mask.sum([0, 1]).clamp(0, 1)).convert('L').save("bmask2.png")

