# Differentiable Augmentation for Data-Efficient GAN Training
# Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han
# https://arxiv.org/pdf/2006.10738
import PIL.Image
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from training.dataset import ImageFolderDataset, xywh2x0y0x1y1


def DiffAugment(x, mask=None, bbox=None, policy='', channels_first=True):
    if policy:
        if not channels_first:
            x = x.permute(0, 3, 1, 2)
            mask = mask.permute(0, 3, 1, 2) if mask is not None else mask
        for p in policy.split(','):
            for f in AUGMENT_FNS[p]:
                x, mask, bbox = f(x, mask, bbox)
        if not channels_first:
            x = x.permute(0, 2, 3, 1)
            mask = mask.permute(0, 2, 3, 1) if mask is not None else mask
        x = x.contiguous()
        mask = mask.contiguous() if mask is not None else mask
    return x, mask, bbox


def rand_brightness(x, mask=None, bbox=None):
    x = x + (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) - 0.5)
    return x, mask, bbox


def rand_saturation(x, mask=None, bbox=None):
    x_mean = x.mean(dim=1, keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) * 2) + x_mean
    return x, mask, bbox


def rand_contrast(x, mask=None, bbox=None):
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    x = (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device) + 0.5) + x_mean
    return x, mask, bbox


def rand_translation(x, mask=None, bbox=None, ratio=0.125):
    # if bbox is not None, (type, x0, y0, x1, y1)
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)

    if mask is not None:
        mask_pad = F.pad(mask, [1, 1, 1, 1, 0, 0, 0, 0])
        mask= mask_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)

    if bbox is not None:
        translation_x, translation_y = translation_x.squeeze(-1) / x.size(2), translation_y.squeeze(-1) / x.size(3)
        new_bbox = bbox.clone()

        new_bbox[:, :, 1] = (bbox[:, :, 1] - translation_y).clamp(0, 1)
        new_bbox[:, :, 3] = (bbox[:, :, 3] - translation_y).clamp(0, 1)
        new_bbox[:, :, 2] = (bbox[:, :, 2] - translation_x).clamp(0, 1)
        new_bbox[:, :, 4] = (bbox[:, :, 4] - translation_x).clamp(0, 1)

        idx = (((new_bbox[:, :, 1]<new_bbox[:, :, 3])*(new_bbox[:, :, 2]<new_bbox[:, :, 4])))
        new_bbox[~idx] = torch.zeros_like(new_bbox[~idx])

        return x, mask, new_bbox

    return x, mask, None


def rand_cutout(x, mask=None, bbox=None, ratio=0.2):
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    m = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    m[grid_batch, grid_x, grid_y] = 0
    x = x * m.unsqueeze(1)

    if mask is not None:
        mask = mask * m.unsqueeze(1)
    return x, mask, bbox


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'translation': [rand_translation],
    'cutout': [rand_cutout],
}
if __name__=="__main__":
    path = "../../data/dataset4"

    data = ImageFolderDataset(path=path)
    image, mask, bbox = data[0]
    bbox1 = xywh2x0y0x1y1(bbox)
    image = torch.from_numpy(image).unsqueeze(0)
    mask = torch.from_numpy(mask).unsqueeze(0)
    bbox2 = torch.from_numpy(bbox1).unsqueeze(0)
    image, mask, bbox3 = rand_translation(image, mask, bbox2)
    bbox3[:, :, 1:] = (bbox3[:, :, 1:]*255).clamp(0, 255)
    bbox3 = bbox3.to(torch.uint8).numpy()
    PIL.Image.fromarray(image.numpy()[0, 0]).convert('L').save("11.png")
    PIL.Image.fromarray(mask.numpy()[0, 0]).convert('L').save("12.png")
    img = image[0, 0].numpy()

    temp_bbox = bbox3[0]
    for box in temp_bbox:
        cv2.rectangle(img, (box[1], box[2]), (box[3], box[4]), (255, 0, 0))

    PIL.Image.fromarray(img).convert('L').save("13.png")
