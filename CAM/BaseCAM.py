from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch
import torchvision.transforms as transforms
import torch.nn as nn


class UnNormalize(nn.Module):
    def __init__(self, mean, std):
        super(UnNormalize, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, tensor):
        """
        Args:
        :param tensor: tensor image of size (B,C,H,W) to be un-normalized
        :return: UnNormalized image
        """
        mean = torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device)
        tensor = tensor
        if tensor.ndim > 3:
            mean = mean.view(1, 3, 1, 1)
            std = std.view(1, 3, 1, 1)
        else:
            mean = mean.view(3, 1, 1)
            std = std.view(3, 1, 1)
        mean = mean.expand(tensor.shape)
        std = std.expand(tensor.shape)
        tensor = tensor * std + mean
        tensor = tensor.clamp(min=0, max=1)
        return tensor


class BaseCAM(object):
    def __init__(self, model, target_layer="module.layer4.2", Norm=True):
        super(BaseCAM, self).__init__()
        self.model = model
        self.gradients = dict()
        self.activations = dict()
        if Norm:
            self.transform_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            self.Nutransform = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # for module in self.model.model.named_modules():
        #     print(module[0])
        #     if module[0] == target_layer:
        #         module[1].register_forward_hook(self.forward_hook)
        #         module[1].register_backward_hook(self.backward_hook)

        if "backbone" in target_layer:
            for module in self.model.model.model.backbone.named_modules():
                if module[0] == '.'.join(target_layer.split('.')[1:]):
                    module[1].register_forward_hook(self.forward_hook)
                    module[1].register_backward_hook(self.backward_hook)
        if 'down' in target_layer:
            for module in self.model.model.model.down.named_modules():
                module[1].register_forward_hook(self.forward_hook)
                module[1].register_backward_hook(self.backward_hook)
        if "neck" in target_layer:
            for module in self.model.model.model.neck.named_modules():
                module[1].register_forward_hook(self.forward_hook)
                module[1].register_backward_hook(self.backward_hook)
        if 'car_head' in target_layer:
            for module in self.model.model.model.car_head.named_modules():
                if module[0] == '.'.join(target_layer.split('.')[1:]):
                    module[1].register_forward_hook(self.forward_hook)
                    module[1].register_backward_hook(self.backward_hook)

        if "softmax" in target_layer:
            for module in self.model.model.softmax.named_modules():
                module[1].register_forward_hook(self.forward_hook)
                module[1].register_backward_hook(self.backward_hook)

        self.esp = 1e-7

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients['value'] = grad_output[0]

    def forward_hook(self, module, input, output):
        self.activations['value'] = output

    def forward(self, x, hp, retain_graph=False):
        raise NotImplementedError

    def __call__(self, x, hp, retain_graph=False):
        return self.forward(x, hp, retain_graph)
