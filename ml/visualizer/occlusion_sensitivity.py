#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26
import os

import torch
import torch.nn.functional as F
from PIL import Image
from matplotlib import cm
import numpy as np
from tqdm import tqdm

from ml.visualizer.utils import save_image, apply_colormap_on_image


class OcclusionSensitivity:
    """
       Produces occlusion sensitivity map by occluding different parts of the
        image and measuring the accuracy drop
    """

    def __init__(self, model):
        # Put model in evaluation mode
        self.model = model

    def visualize(self, images, input_path, result_dir, mean=None, patch=32, stride=20, n_batches=16):
        with torch.no_grad():
            self.model.eval()
            mean = mean if mean else 0
            patch_H, patch_W = (patch, patch)
            pad_H, pad_W = patch_H // 2, patch_W // 2

            # Padded image
            padded_images = F.pad(images, (pad_W, pad_W, pad_H, pad_H), value=mean)
            B, _, H, W = images.shape
            new_H = (H - patch_H) // stride + 1
            new_W = (W - patch_W) // stride + 1

            # Prepare sampling grids
            anchors = []
            grid_h = 0
            while grid_h <= H - patch_H:
                grid_w = 0
                while grid_w <= W - patch_W:
                    grid_w += stride
                    anchors.append((grid_h, grid_w))
                grid_h += stride

            # Baseline score without occlusion
            baseline = self.model(padded_images).detach().cpu()[0]
            target_class = torch.argmax(baseline, -1)
            baseline = baseline[target_class]

            # Compute per-pixel logits
            scoremaps = []
            for i in tqdm(range(0, len(anchors), n_batches), leave=False):
                batch_images = []
                for grid_h, grid_w in anchors[i: i + n_batches]:
                    images_ = padded_images.clone()
                    images_[..., grid_h: grid_h + patch_H, grid_w: grid_w + patch_W] = mean
                    batch_images.append(images_)
                batch_images = torch.cat(batch_images, dim=0)
                scores = self.model(batch_images).detach().cpu()[:, target_class]
                scoremaps += list(torch.split(scores, B))

            diffmaps = torch.cat(scoremaps, dim=0) - baseline
            diffmaps = diffmaps.view(new_H, new_W)

        diffmaps = F.interpolate(diffmaps.unsqueeze(0).unsqueeze(0),
                                 size=(images.size(-2), images.size(-1)),
                                 mode='bilinear', align_corners=False).squeeze()

        original_image = images.cpu().detach().numpy()[0]
        original_image = np.transpose(original_image, [1, 2, 0])
        reverse_mean = np.array([-0.4432, -0.3938, -0.3764])
        reverse_std = np.array([1/0.1560, 1/0.1815, 1/0.1727])
        original_image /= reverse_std
        original_image -= reverse_mean
        original_image = np.uint8(original_image * 255)

        filename = f'{input_path.split("/")[-1].split(".")[0]}_occlusion_sensitivity'
        self.save_sensitivity(diffmaps, original_image, filename, result_dir)

    def save_sensitivity(self, maps, org_img, file_name, result_dir):
        vis_result_dir = os.path.join(result_dir, 'vis')
        if not os.path.exists(vis_result_dir):
            os.makedirs(vis_result_dir)

        maps = maps.cpu().numpy()
        maps = (maps - np.min(maps)) / (np.max(maps) - np.min(maps))  # Normalize between 0-1
        maps = np.uint8(maps * 255)  # Scale between 0-255 to visualize

        # Grayscale activation map
        heatmap, heatmap_on_image = apply_colormap_on_image(org_img, maps, 'bwr_r')
        # Save colored heatmap
        path_to_file = os.path.join(vis_result_dir, file_name + '_gradcam_heatmap.png')
        save_image(heatmap, path_to_file)
        # Save heatmap on image
        path_to_file = os.path.join(os.path.join(vis_result_dir, file_name + '_gradcam_on_image.png'))
        save_image(heatmap_on_image, path_to_file)
