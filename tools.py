import os
import random

import PIL
import click
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from kmeans_pytorch import kmeans

import sys
from pathlib import Path

from tqdm import tqdm

from t3 import COLOR_MAP
from torch_utils.aug_util import xywh2x0y0x1y1
from training.dataset import ImageFolderDataset
from training.training_loop2 import color_mask

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

import legacy
import dnnlib


@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', default="./results/miccai2/00000-stylegan2-dataset4-gpus1-batch16/network-snapshot.pkl")
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, default="./outdir", metavar='DIR')
@click.option('--num_iters', help='Number of iteration for visualization', type=int, default=1)
@click.option('--batch_size', help='Batch size for clustering', type=int, default=8)
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    truncation_psi: float,
    outdir: str,
    num_iters: int,
    batch_size: int,
):
    """K-means visualization of generator feature maps. Cluster the images in the same batch(So the batch size matters here)

    Usage:
        python tools/visualize_gfeat.py --outdir=out --network=your_network_path.pkl
    """
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')

    G_kwargs = dnnlib.EasyDict(class_name='training.SimNet.SimGenerator', z_dim=64, w_dim=128, bbox_dim=256, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
    G_kwargs.synthesis_kwargs.channel_base = 32768
    G_kwargs.synthesis_kwargs.channel_max = 512
    G_kwargs.synthesis_kwargs.num_fp16_res = 4
    G_kwargs.synthesis_kwargs.conv_clamp = 256
    G_kwargs.mapping_kwargs.num_layers = 2
    common_kwargs = dict(c_dim=0, img_resolution=256, img_channels=1)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).eval().to(device)
    with dnnlib.util.open_url(network_pkl) as f:
        G_pkl = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    G.load_state_dict(G_pkl.state_dict())
    os.makedirs(f'{outdir}', exist_ok=True)
    test_set = ImageFolderDataset(path='../data/dataset', use_labels=False, bbox_dim=256, max_size=None, xflip=False,
                                  aug=False)
    dataloader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, pin_memory=True,
                                             num_workers=3,
                                             prefetch_factor=2)
    diter = iter(dataloader)
    C, H, W = test_set[0][0].shape

    for i ,data in tqdm(enumerate(dataloader)):

        real_images, real_masks, bboxs = data
        bboxs = bboxs.to(device)
        # print(bboxs)
        # bmask = bbox_mask(device, bboxs[:, :, 1:], 256, 256)*2 - 1
        # PIL.Image.fromarray(color_mask(bmask)[0]).save(os.path.join(out, f'bmask/{(i * 1000)*1000 :09d}.png'))
        bbox2 = xywh2x0y0x1y1(bboxs)
        # print(bbox2)
        bbox2[:, :, 1:] = (bbox2[:, :, 1:] * 255)
        bbox2 = bbox2.to(torch.uint8).cpu().numpy()
        layout = np.zeros([batch_size, H, W, 3], dtype=np.uint8)
        for idx in range(batch_size):
            for h, box in enumerate(bbox2[idx]):
                if box[0] != 0:
                    # print((box[1], box[2]), (box[3], box[4]), COLOR_MAP[str(h//len(COLOR_MAP))])
                    cv2.rectangle(layout[idx], (box[1], box[2]), (box[3], box[4]), COLOR_MAP[str(h%len(COLOR_MAP))], 2)

        layout = torch.from_numpy(layout.transpose(0, 3, 1, 2)).to(device)

        # PIL.Image.fromarray(real_images[0, 0].cpu().numpy()).convert('L').save(
        #     os.path.join(out, f'real_images/{(i * 1000)*1000 :09d}.png'))
        # PIL.Image.fromarray(real_masks[0, 0].cpu().numpy()).convert('L').save(
        #     os.path.join(out, f'real_masks/{(i * 1000)*1000 :09d}.png'))
        # bbox_mask_ = bbox_mask(bboxs.device, bboxs[:, :, 1:], 256, 256)

        z = torch.randn(batch_size, G.z_dim).to(device)
        fake_imgs, fake_masks, fake_mid_masks, fake_feat = G(z, bboxs, isTrain=False, get_feat=True)

        target_layers = [16, 32, 64]
        num_clusters = 6

        vis_img = []

        for res in target_layers:
            img = get_cluster_vis(fake_feat[res], num_clusters=num_clusters, target_res=res)  # bnum, 256, 256
            vis_img.append(img)

        for idx, val in enumerate(vis_img):
            vis_img[idx] = F.interpolate(val, size=(256, 256))

        vis_img = torch.cat(vis_img, dim=0)  # bnum * res_num, 256, 256
        vis_img = (vis_img + 1) * 127.5 / 255.0
        fake_imgs = (fake_imgs + 1) * 127.5 / 255.0
        fake_imgs = F.interpolate(fake_imgs, size=(256, 256)).repeat(1, 3, 1 , 1)

        layout = layout / 255.0
        vis_img = torch.cat([layout, fake_imgs, vis_img], dim=0)
        vis_img = torchvision.utils.make_grid(vis_img, normalize=False, nrow=batch_size)
        torchvision.utils.save_image(vis_img, f'{outdir}/{(i*1000)*1000:09d}_semic.png')



def get_colors():
    dummy_color = np.array([
        [178, 34, 34],  # firebrick
        [0, 139, 139],  # dark cyan
        [245, 222, 179],  # wheat
        [25, 25, 112],  # midnight blue
        [255, 140, 0],  # dark orange
        [128, 128, 0],  # olive
        [50, 50, 50],  # dark grey
        [34, 139, 34],  # forest green
        [100, 149, 237],  # corn flower blue
        [153, 50, 204],  # dark orchid
        [240, 128, 128],  # light coral
    ])

    for t in (0.6, 0.3):  # just increase the number of colors for big K
        dummy_color = np.concatenate((dummy_color, dummy_color * t))

    dummy_color = (np.array(dummy_color) - 128.0) / 128.0
    dummy_color = torch.from_numpy(dummy_color)

    return dummy_color


def get_cluster_vis(feat, num_clusters=10, target_res=16):
    # feat : NCHW
    print(feat.size())
    img_num, C, H, W = feat.size()
    feat = feat.permute(0, 2, 3, 1).contiguous().view(img_num * H * W, -1)
    feat = feat.to(torch.float32).cuda()
    cluster_ids_x, cluster_centers = kmeans(
        X=feat, num_clusters=num_clusters, distance='cosine',
        tol=1e-4,
        device=torch.device("cuda:0"))

    cluster_ids_x = cluster_ids_x.cuda()
    cluster_centers = cluster_centers.cuda()
    color_rgb = get_colors().cuda()
    vis_img = []
    for idx in range(img_num):
        num_pixel = target_res * target_res
        current_res = cluster_ids_x[num_pixel * idx:num_pixel * (idx + 1)].cuda()
        color_ids = torch.index_select(color_rgb, 0, current_res)
        color_map = color_ids.permute(1, 0).view(1, 3, target_res, target_res)
        color_map = F.interpolate(color_map, size=(256, 256))
        vis_img.append(color_map.cuda())

    vis_img = torch.cat(vis_img, dim=0)

    return vis_img


if __name__ == "__main__":
    generate_images()  # pylint: disable=no-value-for-parameter