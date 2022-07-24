from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import numpy as np
import torch.nn.functional as F
from kornia.filters.gaussian import gaussian_blur2d
from CAM.BaseCAM import BaseCAM
import torch
from CAM.cluster import group_sum
from pysot.core.config import cfg

blur = lambda x: gaussian_blur2d(x, kernel_size=(51, 51), sigma=(50., 50.))
# Copyright (c) SenseTime. All Rights Reserved.

class GroupCAM(BaseCAM):
    def __init__(self, model, target_layer="layer4.2", Norm=True, groups=cfg.TRAIN.NUM_CLASSES, cluster_method=None):
        super(GroupCAM, self).__init__(model, target_layer, Norm)
        self.groups = groups
        assert cluster_method in [None, 'k_means', 'agglomerate']
        self.cluster = cluster_method


    def forward(self, x, hp, retain_graph=False):
        output = self.model.track_cam(x, hp)
        cls = output["cls"]
        # idx = output["idx"]
        x_crop = output["x_crop"]
        bbox = output["bbox"]
        b, c, h, w = x_crop.size()
        self.model.model.zero_grad()
        idx = torch.argmax(cls)
        score = cls.reshape(-1)[idx]
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value'].data
        activations = self.activations['value'].data
        b, k, u, v = activations.size()

        alpha = gradients.view(b, k, -1).mean(2)
        weights = alpha.view(b, k, 1, 1)
        activations = weights * activations

        if self.cluster is None:
            saliency_map = activations.chunk(self.groups, 1)
            saliency_map = torch.cat(saliency_map, dim=0)
            saliency_map = saliency_map.sum(1, keepdim=True)
        else:
            saliency_map = group_sum(activations, n=self.groups, cluster_method=self.cluster)
            saliency_map = torch.cat(saliency_map, dim=0)

        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        norm_saliency_map = saliency_map.reshape(self.groups, -1)
        inter_min = norm_saliency_map.min(dim=-1, keepdim=True)[0]
        inter_max = norm_saliency_map.max(dim=-1, keepdim=True)[0]
        norm_saliency_map = (norm_saliency_map-inter_min) / (inter_max - inter_min)
        norm_saliency_maps = norm_saliency_map.reshape(self.groups, 1, h, w)
        outputs = torch.zeros((self.groups, 1))
        with torch.no_grad():
            x_crop = torch.cat([x_crop[:, 2, :, :][:, None, :, :],
                                     x_crop[:, 1, :, :][:, None, :, :],
                                     x_crop[:, 0, :, :][:, None, :, :]], dim=1) / 255.0
            # 归一化和反归一化中的squeeze(0)和unsqueeze(0)用于兼容不同版本的torchvision
            norm_img = self.transform_norm(x_crop.squeeze(0)).unsqueeze(0)
            blur_img = blur(norm_img)
            img = self.Nutransform(blur_img.squeeze(0)).unsqueeze(0)
            base_line = self.model.model.track(torch.cat([img[:, 2, :, :][:, None, :, :],
                                     img[:, 1, :, :][:, None, :, :],
                                     img[:, 0, :, :][:, None, :, :]], dim=1) * 255.0)["cls"][:, 1, :, :][:, None, :, :].reshape(-1)[idx]

            for index, norm_saliency_map in enumerate(norm_saliency_maps):
                blur_x = norm_img * norm_saliency_map + blur_img * (1 - norm_saliency_map)
                img = self.Nutransform(blur_x.squeeze(0)).unsqueeze(0)
                outcls = self.model.model.track(torch.cat([img[:, 2, :, :][:, None, :, :],
                                                       img[:, 1, :, :][:, None, :, :],
                                                       img[:, 0, :, :][:, None, :, :]], dim=1) * 255.0)["cls"][:, 1, :,
                     :][:, None, :, :].reshape(-1)[idx]
                outputs[index, 0] = outcls
        score = outcls - base_line.unsqueeze(0).repeat(self.groups, 1)
        score = F.relu(score).unsqueeze(-1).unsqueeze(-1)
        score_saliency_map = torch.sum(saliency_map * score, dim=0, keepdim=True)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()
        if score_saliency_map_min == score_saliency_map_max:
            return None, None, None
        score_saliency_map = (score_saliency_map - score_saliency_map_min) / (
                    score_saliency_map_max - score_saliency_map_min + self.esp).data
        return score_saliency_map.cpu().data, x_crop.cpu().numpy(), bbox

    def __call__(self, input, hp, retain_graph=True):
        return self.forward(input, hp, retain_graph)
