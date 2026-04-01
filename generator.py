import glob
import os
import re
import shutil
import time
from pathlib import Path
import random

import PIL
from tqdm import tqdm

import legacy
import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch.nn.functional as F

import dnnlib

# BOX_COLOR = (255, 0, 0)  # Red
# TEXT_COLOR = (255, 255, 255)  # White
from torch_utils.aug_util import xywh2x0y0x1y1
from torch_utils.common import VGGLoss, Perceptual_loss134
from torch_utils.custom_ops import bbox_mask
from training.SimNet import SimGenerator
from training.dataset import ImageFolderDataset


class Gen(torch.nn.Module):
    def __init__(self):
        super().__init__()

        batch_size = 8
        start = 0
        length = 100
        self.init_seed = 20
        self.outdir = '../gen'

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)

        # if not os.path.exists(self.outdir+'/images'):
        #     os.makedirs(self.outdir+'/images')
        # if not os.path.exists(self.outdir+'/masks'):
        #     os.makedirs(self.outdir+'/masks')

        self.path = '../data/dataset'
        self.network_pkl = '../res/test/00000-dataset4-auto2/network-snapshot-001200.pkl'
        self.device_gpu = torch.device('cuda')
        self.device_cpu = torch.device('cpu')
        self.training_set = ImageFolderDataset(path=self.path, use_labels=False, bbox_dim=128, max_size=None,
                                          xflip=False, aug=False)
        self.dataloader = torch.utils.data.DataLoader(dataset=self.training_set, batch_size=1, num_workers=3, prefetch_factor=2)

        self.loadGenMode(self.network_pkl)

    def getBbox(self):
        bboxs = []
        for idx in range(len(self.training_set)):
            image, mask, bbox = self.training_set[idx]
            bboxs.append(bbox)
        return np.array(bboxs)

    def genSingleImgsBbox(self):
        bboxs = self.getBbox()
        for seed_idx, bbox in enumerate(bboxs):
            bbox = torch.from_numpy(bbox[None, :, :]).to(self.device_gpu)
            print('Generating image for (%d/%d) ...' % (seed_idx, len(self.training_set)))
            z = torch.from_numpy(np.random.RandomState(self.init_seed + seed_idx).randn(1, self.net.z_dim)).to(self.device_gpu)
            img, mask, _, _, _ = self.net(z, bbox)
            canvas = PIL.Image.new('L', (256 * 3, 256 * 1), 'black')
            img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            mask = (mask * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            canvas.paste(PIL.Image.fromarray(img[0, 0]).convert("L"), (256 * 0, 256 * 0))
            canvas.paste(PIL.Image.fromarray(mask[0, 0]).convert("L"), (256 * 1, 256 * 0))
            # bbox = bbox[((bbox[:, 2] > 0) * (bbox[:, 3] > 0))]
            bbox = xywh2x0y0x1y1(bbox)
            bbox = (bbox * 256).to(torch.uint8).cpu().numpy()
            # print(bbox)
            for lab in bbox[0]:
                # print((lab[0], lab[1]), (lab[2], lab[3]))
                cv2.rectangle(img[0, 0], (lab[0], lab[1]), (lab[2], lab[3]), (255, 0, 0))
            canvas.paste(PIL.Image.fromarray(img[0, 0]).convert("L"), (256 * 2, 256 * 0))
            canvas.save(f'{self.outdir}/seed{seed_idx:04d}.png')

        net = self.loadGenMode(self.network_pkl)

        save_dir = "../data/generate3_4"
        save_dir1 = f"{save_dir}/images"
        save_dir2 = f"{save_dir}/masks"
        save_dir3 = f"{save_dir}/labels"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_dir1):
            os.makedirs(save_dir1)
        if not os.path.exists(save_dir2):
            os.makedirs(save_dir2)
        if not os.path.exists(save_dir3):
            os.makedirs(save_dir3)

        sample_num = random.sample(range(0, len(self.training_set)), 890)
        num = "one"
        print("generator dataset by dataset3 and copy ...")
        for seed_idx, data in enumerate(tqdm(self.dataloader)):
            images, masks, bbox = data
            bbox = bbox.to(self.device_gpu)
            z = torch.from_numpy(np.random.RandomState(self.init_seed + seed_idx).randn(1, self.net.z_dim)).to(
                self.device_gpu)

            canvas = PIL.Image.new('L', (256 * 4, 256 * 1), 'black')
            gen_img, gen_mask, ws1, ws2, gen_mid_mask = net(z, bbox)
            img = (gen_img * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            mask = (gen_mask.sign() * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            bbox = self.xywh2x0y0wh(bbox).cpu().numpy()
            PIL.Image.fromarray(img[0, 0]).convert('L').save(f'{save_dir}/images/seed{seed_idx}.png')
            PIL.Image.fromarray(mask[0, 0]).convert('L').save(f'{save_dir}/masks/seed{seed_idx}.png')
            with open(f"{save_dir}/labels/seed{seed_idx}.txt", "w+") as f:
                for box in bbox:
                    f.write(f"1 {round(box[0], 6)} {round(box[1], 6)} {round(box[2], 6)} {round(box[3], 6)}\n")

    def gen_img(self):
        net = self.loadGenMode(self.network_pkl)

        save_dir = "../data/generate3_4"
        save_dir1 = f"{save_dir}/images"
        save_dir2 = f"{save_dir}/masks"
        save_dir3 = f"{save_dir}/labels"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_dir1):
            os.makedirs(save_dir1)
        if not os.path.exists(save_dir2):
            os.makedirs(save_dir2)
        if not os.path.exists(save_dir3):
            os.makedirs(save_dir3)

        sample_num = random.sample(range(0, len(self.training_set)), 890)
        num = "one"
        print("generator dataset by dataset3 and copy ...")
        for seed_idx, data in enumerate(tqdm(self.dataloader)):
            images, masks, bbox = data
            bbox = bbox.to(self.device_gpu)
            z = torch.from_numpy(np.random.RandomState(self.init_seed + seed_idx).randn(1, self.net.z_dim)).to(self.device_gpu)
            gen_img, gen_mask, ws1, ws2, gen_mid_mask = net(z, bbox)
            img = (gen_img * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            mask = (gen_mask.sign() * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
            bbox = self.xywh2x0y0wh(bbox).cpu().numpy()
            PIL.Image.fromarray(img[0, 0]).convert('L').save(f'{save_dir}/images/seed{seed_idx}.png')
            PIL.Image.fromarray(mask[0, 0]).convert('L').save(f'{save_dir}/masks/seed{seed_idx}.png')
            with open(f"{save_dir}/labels/seed{seed_idx}.txt", "w+") as f:
                for box in bbox:
                    f.write(f"1 {round(box[0], 6)} {round(box[1], 6)} {round(box[2], 6)} {round(box[3], 6)}\n")

            # if seed_idx in sample_num:
            #     shutil.copy(f'{save_dir}/images/seed{seed_idx}.png', f"../data/projected/{num}/images/seed{seed_idx}.png")
            #     shutil.copy(f'{save_dir}/masks/seed{seed_idx}.png', f"../data/projected/{num}/masks/seed{seed_idx}.png")
            #     shutil.copy(f'{save_dir}/labels/seed{seed_idx}.txt', f"../data/projected/{num}/labels/seed{seed_idx}.txt")

    def xywh2x0y0wh(self, bbox):
        new_bbox = bbox.clone()
        new_bbox[:, :, 0] = new_bbox[:, :, 0] - new_bbox[:, :, 2]/2
        new_bbox[:, :, 1] = new_bbox[:, :, 1] - new_bbox[:, :, 3]/2
        new_bbox = new_bbox[((new_bbox[:, :, 2]>0) & (new_bbox[:, :, 3]>0))]
        return new_bbox



    def copyfile(self):
        num = "one"
        print("copy dataset4 ...")
        for image_name in tqdm(os.listdir("../data/dataset4/images")):
            name = image_name.split(".")[0]
            shutil.copy(f'../data/dataset4/images/{name}.png', f"../data/projected/{num}/images/{name}.png")
            shutil.copy(f'../data/dataset4/masks/{name}.png', f"../data/projected/{num}/masks/{name}.png")
            shutil.copy(f'../data/dataset4/labels/{name}.txt', f"../data/projected/{num}/labels/{name}.txt")

    def loadGenMode(self, network_pkl):
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(self.device_gpu)
        self.net = G
        return G

    def loadGenModeWithNet(self, network_pkl):
        G = self.loadGenMode(network_pkl)
        common_kwargs = dnnlib.EasyDict(channel_base=16384, channel_max=512, num_fp16_res=4, conv_clamp=256)
        mapping_kwargs = dnnlib.EasyDict(num_layers=2)
        Gnet = SimGenerator(mapping_kwargs=mapping_kwargs, synthesis_kwargs=common_kwargs).eval().to(self.device_gpu)
        Gnet.load_state_dict(G.state_dict())
        self.net = Gnet

if __name__ == "__main__":
    # Gen().gen_img()
    Gen().genSingleImgsBbox()