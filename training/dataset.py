# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import random

import albumentations
import albumentations as A
import cv2
import numpy as np
import zipfile
import PIL.Image
import json
import torch
from torchvision import transforms
from torch.nn import functional as F
from torchvision.ops import roi_align

import dnnlib
from torch_utils import misc

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

def x0y0wh2xywh(input):
    assert input.ndim == 2
    output = input.copy()
    output[:, 1] = input[:, 1] + input[:, 3]/2
    output[:, 2] = input[:, 2] + input[:, 4]/2
    return output

def xywh2x0y0wh(input):
    assert input.ndim == 2
    output = input.copy()
    output[:, 1] = input[:, 1] - input[:, 3]/2
    output[:, 2] = input[:, 2] - input[:, 4]/2
    return output

def xywh2x0y0x1y1(input):
    assert input.ndim == 2
    output = input.copy()
    output[:, 1] = input[:, 1] - input[:, 3]/2
    output[:, 2] = input[:, 2] - input[:, 4]/2
    output[:, 3] = input[:, 1] + input[:, 3]/2
    output[:, 4] = input[:, 2] + input[:, 4]/2
    return output

def x0y0x1y12xywh(input):
    assert input.ndim == 2
    output = input.copy()
    output[:, 3] = input[:, 3] - input[:, 1]
    output[:, 4] = input[:, 4] - input[:, 2]
    output[:, 1] = (input[:, 1] + input[:, 3])/2
    output[:, 2] = (input[:, 2] + input[:, 4])/2
    return output


def HorizontalFlip(image, mask, bbox):
    # image:HWC
    # mask :HWC
    # bbox :(box_num, 4) (xywh)
    image, mask = np.fliplr(image), np.fliplr(mask)
    bbox[:, 1] = 1 - bbox[:, 1]
    return image, mask, bbox

def VerticalFlip(image, mask, bbox):
    # image:HWC
    # mask :HWC
    # bbox :(box_num, 4) (xywh)
    image, mask = np.flipud(image), np.flipud(mask)
    bbox[:, 2] = 1 - bbox[:, 2]
    return image, mask, bbox

def cropAndresize(image, mask, bbox, size=0.5, locate=(0.25, 0.25)):
    # image:HWC
    # mask :HWC
    # bbox :(box_num, 4) (xywh)
    # size : new_resolution = image resolution // size
    # location : （0.25 * 256， 0.25 * 256） = （64， 64） => crop 中心点为（64， 64）
    resolution = image.shape[1]
    if locate == (0, 0):
        locate = (size/2, size/2)

    # bbox[:, 2], bbox[:, 3] = bbox[:, 2]*size, bbox[:, 3]*size

    newx0, newx1 = locate[0]-size/2, locate[0]+size/2,
    newy0, newy1 = locate[1]-size/2, locate[1]+size/2,

    new_x0, new_x1 = int(newx0*resolution), int(newx1*resolution)
    new_y0, new_y1 = int(newy0*resolution), int(newy1*resolution)

    image = image[new_y0:new_y1, new_x0:new_x1]
    mask = mask[new_y0:new_y1, new_x0:new_x1]

    image = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    mask  = cv2.resize(mask, (resolution, resolution), interpolation=cv2.INTER_LINEAR)

    if image.ndim == 2:
        image = image[:, :, None]
    if mask.ndim == 2:
        mask = mask[:, :, None]

    bbox1 = xywh2x0y0x1y1(bbox.copy())
    # # (x0, y0, x1, y1) , (newx0, newy0, newx1, newy1)
    bbox2 = bbox1[(abs(bbox1[:, 3]+ bbox1[:, 1]-newx0-newx1)<(bbox1[:, 3]-bbox1[:, 1]+newx1-newx0)) * (abs(bbox1[:, 4]+ bbox1[:, 2]-newy0-newy1)<(bbox1[:, 4]-bbox1[:, 2]+newy1-newy0))]

    bbox2[:, 1] = np.where(bbox2[:, 1]<newx0, newx0, bbox2[:, 1]) - newx0
    bbox2[:, 2] = np.where(bbox2[:, 2]<newy0, newy0, bbox2[:, 2]) - newy0
    bbox2[:, 3] = np.where(bbox2[:, 3]>newx1, newx1, bbox2[:, 3]) - newx0
    bbox2[:, 4] = np.where(bbox2[:, 4]>newy1, newy1, bbox2[:, 4]) - newy0
    # # relative location
    bbox2[:, 1:] = bbox2[:, 1:] / size
    bbox3 = x0y0x1y12xywh(bbox2)

    return image, mask, bbox3

def rot90(image, mask, bbox, times: int):
    """
    逆时针90°旋转图像times次，并计算图像image中的坐标点points在旋转后的图像中的位置坐标.
    Args:
        image: 图像数组
        points: [(x, y), ...]，原图像中的坐标点集合
        times: 旋转次数
    """

    if times % 4 == 0:	# 旋转4的倍数次，相当于不旋转
        return image, mask, bbox
    else:
        times = times % 4

    image = np.rot90(image, times, (0, 1))	# 通过numpy实现图像旋转
    mask = np.rot90(mask, times, (0, 1))	# 通过numpy实现图像旋转

    new_bbox = bbox.copy()
    if times%2 == 1:
        new_bbox[:, 3], new_bbox[:, 4] = bbox[:, 4], bbox[:, 3]

    if times == 1:
        new_bbox[:, 1], new_bbox[:, 2] = bbox[:, 2], 1 - bbox[:, 1]
    elif times == 2:
        new_bbox[:, 1], new_bbox[:, 2] = 1 - bbox[:, 1], 1 - bbox[:, 2]
    else:
        new_bbox[:, 1], new_bbox[:, 2] = 1 - bbox[:, 2], bbox[:, 1]

    return image, mask, new_bbox


def ReversibleAugment(image, mask, bbox):
    p = np.floor(np.random.random(4)*2)
    # p = np.array([0.0, 0.0, 1.0, 0.0])
    locate = tuple(0.25 + 0.0625 * np.random.randint(0, 8, 2))
    times = random.randint(0, 4)
    # ops = dnnlib.EasyDict(HorizontalFlip=p[0], VerticalFlip=p[1], cropAndresize=dnnlib.EasyDict(car=p[2], locate=locate), rot90=dnnlib.EasyDict(rot=p[3], times=times))
    ops = np.array([*p, *locate, times])
    if p[0]:
        image, mask, bbox = HorizontalFlip(image, mask, bbox)
    if p[1]:
        image, mask, bbox = VerticalFlip(image, mask, bbox)
        # print("p1", image.shape, mask.shape)
    if p[2]:
       locate = tuple(0.25 + 0.0625 * np.random.randint(0, 8, 2))
       image, mask, bbox = cropAndresize(image, mask, bbox, locate=locate)
       # print("p2", image.shape, mask.shape)

    if p[3]:
        times = np.random.randint(0, 4, 1)
        image, mask, bbox = rot90(image, mask, bbox, times=times)
        # print("p3", image.shape, mask.shape)

    return image, mask, bbox, ops

def bboxSort(bbox):
    new_bbox = np.zeros([bbox.shape[0], 7])
    new_bbox[:, 2:] = bbox.copy()
    new_bbox[:, 0] = np.where(new_bbox[:, 5]<new_bbox[:, 6], new_bbox[:, 5]/new_bbox[:, 6], new_bbox[:, 6]/new_bbox[:, 5])
    new_bbox = new_bbox[new_bbox[:, 0]>0.5]
    new_bbox = new_bbox[np.argsort(-new_bbox[:, 0])]
    new_bbox = new_bbox[:, 2:]
    return new_bbox

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        aug         = True,
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        self.aug = aug

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

        # self.ps = 0.0+(np.random.random(4)<=0.5)
        # self.aug_t = albumentations.Compose([
        #     A.HorizontalFlip(p=self.ps[0]),
        #     A.VerticalFlip(p=self.ps[1]),
        #     A.RandomRotate90(p=self.ps[2]),
        #     A.Compose([
        #         A.RandomCrop(self.resolution//2, self.resolution//2,p=1),
        #         A.Resize(self.resolution, self.resolution, p=1),
        #     ], p=self.ps[3])
        #     ],
        #     bbox_params=A.BboxParams(format='yolo', label_fields=["class_labels"])
        # )

        # self.aug_t2 = albumentations.Compose([
        #     A.Rotate(p=1)
        #     ],
        #     bbox_params=A.BboxParams(format='yolo', label_fields=["class_labels"])
        # )


    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_img_mask(self, path, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_mask(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_bbox(self, raw_idx):
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        while True:
            image, mask, bbox = self.__getitem2__(idx)
            if len(bbox) != 0:
                _add_nums = (self.bbox_dim - len(bbox)) if len(bbox) < self.bbox_dim else 0
                if _add_nums > 0:
                    _add_bbox = np.zeros([_add_nums, 5], dtype=float)
                    bbox = np.concatenate([bbox, _add_bbox])
                bbox = bbox[:self.bbox_dim]

                return image, mask, bbox

            idx = random.randint(0, self.__len__()-1)

    def __getitem2__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        mask  = self._load_raw_mask(self._raw_idx[idx])
        bbox  = self._load_raw_bbox(self._raw_idx[idx])
        # assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        # assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        bbox1 = x0y0wh2xywh(bbox)
        if self.aug:
           image, mask, bbox1, ops = ReversibleAugment(image, mask, bbox1)
        bbox2 = bbox1[((bbox1[:, 3]>0) * (bbox1[:, 4]>0))]
        # bbox = xywh2x0y0wh(bbox)
        bbox3 = bboxSort(bbox2)

        image = image.transpose(2, 0, 1)
        mask = mask.transpose(2, 0, 1)
        image = np.ascontiguousarray(image)
        mask = np.ascontiguousarray(mask)

        return image, mask, bbox3

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[2]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # HWC
        assert self.image_shape[0] == self.image_shape[1]
        return self.image_shape[1]

    @property
    def channel(self):
        assert len(self.image_shape) == 3  # HWC
        return self.image_shape[2]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        bbox_dim        = 128,
        type            = 'coco', # 'coco':x0, y0, w, h, 'yolo' x, y, h, w
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self.bbox_dim = bbox_dim
        self.type = type
        self._path = path
        self._img_path = os.path.join(self._path, 'images')
        self._bbox_path = os.path.join(self._path, 'labels')
        self._mask_path = os.path.join(self._path, 'masks')
        self._zipfile = None
        self._img_ext = self._file_ext(os.listdir(self._img_path)[0])
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,), inplace=True),
            ]
        )

        if os.path.isdir(self._img_path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._img_path) for root, _dirs, files in os.walk(self._img_path) for fname in files}
        elif self._file_ext(self._img_path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(os.path.splitext(fname)[0] for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[1] != resolution or raw_shape[2] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._img_path)
        return self._zipfile

    def _open_file(self, path, fname):
        if self._type == 'dir':
            return open(os.path.join(path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_img_mask(self, path, raw_idx):
        fname = self._image_fnames[raw_idx] + self._img_ext
        with self._open_file(path, fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        # image = image.transpose(2, 0, 1) # HWC => CHW
        # image = self.transform(image)
        return image

    def _load_raw_image(self, raw_idx):
        return self._load_img_mask(self._img_path, raw_idx)

    def _load_raw_mask(self, raw_idx):
        mask = self._load_img_mask(self._mask_path, raw_idx)
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # mask = cv2.dilate(mask, kernel)
        return mask

    def _load_raw_bbox(self, raw_idx):
        fname = self._image_fnames[raw_idx] + ".txt"
        bbox = np.loadtxt(os.path.join(self._bbox_path, fname))
        if bbox.ndim == 1:
            bbox = bbox[np.newaxis, :]

        if bbox.shape[1] !=5:
            bbox = np.zeros([0, 5])

        bbox2 = bbox[((bbox[:, 3]*bbox[:, 4]) > 0)]
        return bbox2

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------
def pre_bbox(bbox):
    bbox[:, :, 1:] = (bbox[:, :,  1:] * 255).clamp(0, 255).to(torch.uint8)
    label, bbox = bbox[:, :,  0], bbox[:, :, 1:]
    idx = torch.arange(start=0, end=bbox.size(0), device=bbox.device).view(bbox.size(0), 1, 1).expand(-1, bbox.size(1), -1).float()
    bbox = torch.cat((idx, bbox.float()), dim=2)
    bbox = bbox.view(-1, 5)
    label = label.view(-1)

    idx = (label != 0).nonzero().view(-1)
    bbox = bbox[idx]
    label = label[idx]
    return label, bbox

def compute_iou(bbox1, bbox2, eps=1e-8):
    """
    compute the IoU of two bounding boxes.
    :param eps: avoid
    :param bbox1: bounding box No.1.
    :param bbox2: bounding box No.2.
    :return: IoU of bbox1 and bbox2.
    """
    center = True
    if center:
        x1, y1, w1, h1 = bbox1
        xmin1, ymin1 = int(x1-w1/2.0), int(y1-h1/2.0)
        xmax1, ymax1 = int(x1+w1/2.0), int(y1+h1/2.0)
        x2, y2, w2, h2 = bbox2
        xmin2, ymin2 = int(x2-w2/2.0), int(y2-h2/2.0)
        xmax2, ymax2 = int(x2+w2/2.0), int(y2+h2/2.0)
    else:
        xmin1, ymin1, xmax1, ymax1 = bbox1
        xmin2, ymin2, xmax2, ymax2 = bbox2

    # 计算交集的对角坐标点
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    # 计算交集面积
    w = np.max([0.0, xx2 - xx1])
    h = np.max([0.0, yy2 - yy1])
    area_intersection = w * h

    # 计算并集面积（这里要记得去掉重叠的面积，避免重复计算）
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmax1) * (ymax2 - ymin2)
    area_union = area1 + area2 - area_intersection

    # 计算两个边框的交并比
    iou = area_intersection / (area_union + eps)
    print(area_intersection, area_union, iou)
    return iou

if __name__ == "__main__":
    path = "../../data/dataset"

    dataset = ImageFolderDataset(path=path, aug=False)

    for i in range(len(dataset)):
        image, mask, bbox = dataset[i]
        # bbox = bbox[:5]
        l, x, y = torch.from_numpy(bbox[:, 0]), torch.from_numpy(bbox[:, 1]), torch.from_numpy(bbox[:, 2])
        num = l.sum()
        lmetrix = l.unsqueeze(0).repeat(len(l), 1) * l.unsqueeze(1).repeat(1, len(l))
        xmetrix = x.unsqueeze(0).repeat(len(x), 1)
        ymetrix = y.unsqueeze(0).repeat(len(y), 1)

        xmetrix = torch.pow(xmetrix - xmetrix.T, 2)
        ymetrix = torch.pow(ymetrix - ymetrix.T, 2)
        metrix = torch.sqrt(xmetrix + ymetrix) * lmetrix

        self_dis = torch.sqrt(torch.pow(x, 2) + torch.pow(y, 2)) # the distance to (0, 0)

        iou = np.zeros([x.shape[0], x.shape[0]])

        for j in range(len(bbox)):
            for k in range(j+1, len(bbox)):
                iou[j, k] = compute_iou(bbox[j, 1:], bbox[k, 1:])

        print(iou)


        print(metrix.sum()/2, num)




