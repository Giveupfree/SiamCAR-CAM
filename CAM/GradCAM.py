import torch
import torch.nn.functional as F
from CAM.BaseCAM import BaseCAM


class GradCAM(BaseCAM):
    def __init__(self, model, target_layer="module.layer4.2"):
        super().__init__(model, target_layer)

    def forward(self, x, hp, retain_graph=False):
        output = self.model.track_cam(x, hp)
        cls = output["cls"]
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
        saliency_map = (weights * activations).sum(1, keepdim=True)

        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min + self.esp)

        x_crop = torch.cat([x_crop[:, 2, :, :][:, None, :, :], x_crop[:, 1, :, :][:, None, :, :], x_crop[:, 0, :, :][:, None, :, :]], dim=1) / 255.0

        return saliency_map.cpu().data, x_crop.cpu().numpy(), bbox

    def __call__(self, x, hp, retain_graph=True):
        return self.forward(x, hp, retain_graph)

class GradCAMpp(BaseCAM):
    def __init__(self, model, target_layer="module.layer4.2"):
        super().__init__(model, target_layer)
    def forward(self, x, hp, retain_graph=False):
        output = self.model.track_cam(x, hp)
        cls = output["cls"]
        x_crop = output["x_crop"]
        bbox = output["bbox"]
        b, c, h, w = x_crop.size()
        self.model.model.zero_grad()
        idx = torch.argmax(cls)
        score = cls.reshape(-1)[idx]
        score.backward(retain_graph=retain_graph)

        gradients = self.gradients['value']  # dS/dA
        activations = self.activations['value']  # A
        b, k, u, v = activations.size()

        alpha_num = gradients.pow(2)
        alpha_denom = gradients.pow(2).mul(2) + \
                      activations.mul(gradients.pow(3)).view(b, k, u * v).sum(-1, keepdim=True).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alpha = alpha_num.div(alpha_denom + self.esp)

        positive_gradients = F.relu(score.exp() * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha * positive_gradients).view(b, k, u * v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)

        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min + self.esp)
        x_crop = torch.cat([x_crop[:, 2, :, :][:, None, :, :], x_crop[:, 1, :, :][:, None, :, :], x_crop[:, 0, :, :][:, None, :, :]],
            dim=1) / 255.0

        return saliency_map.cpu().data, x_crop.cpu().numpy(), bbox

    def __call__(self, x, hp, retain_graph=True):
        return self.forward(x, hp, retain_graph)


class SmoothGradCAM(BaseCAM):
    def __init__(self, model, target_layer="module.layer4.2", stdev_spread=0.15, n_samples=20, magnitude=True):
        super().__init__(model, target_layer)
        self.stdev_spread = stdev_spread
        self.n_samples = n_samples
        self.magnitude = magnitude

    def forward(self, x, hp, retain_graph=False):
        saliency_map = 0.0
        output = self.model.track_cam(x, hp)
        cls = output["cls"]
        x_crop = output["x_crop"]
        bbox = output["bbox"]
        b, c, h, w = x_crop.size()
        self.model.model.zero_grad()
        idx = torch.argmax(cls)

        x_crop = torch.cat([x_crop[:, 2, :, :][:, None, :, :],
                            x_crop[:, 1, :, :][:, None, :, :],
                            x_crop[:, 0, :, :][:, None, :, :]], dim=1) / 255.0
        norm_img = self.transform_norm(x_crop.squeeze(0)).unsqueeze(0)

        stdev = self.stdev_spread / (norm_img.max() - norm_img.min())
        std_tensor = torch.ones_like(norm_img) * stdev

        self.model.model.zero_grad()
        for _ in range(self.n_samples):
            x_plus_noise = torch.normal(mean=norm_img, std=std_tensor)
            x_plus_noise.requires_grad_()
            x_plus_noise.cuda()
            img = self.Nutransform(x_plus_noise.squeeze(0)).unsqueeze(0)
            score = self.model.model.track(torch.cat([img[:, 2, :, :][:, None, :, :],
                                     img[:, 1, :, :][:, None, :, :],
                                     img[:, 0, :, :][:, None, :, :]], dim=1) * 255.0)["cls"][:, 1, :,
                     :][:, None, :, :].reshape(-1)[idx]
            score.backward(retain_graph=retain_graph)
            gradients = self.gradients['value']
            if self.magnitude:
                gradients = gradients * gradients
            activations = self.activations['value']
            b, k, u, v = activations.size()
            alpha = gradients.view(b, k, -1).mean(2)
            weights = alpha.view(b, k, 1, 1)
            saliency_map += (weights * activations).sum(1, keepdim=True).data
        saliency_map = F.relu(saliency_map)
        saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min + self.esp)
        return saliency_map.cpu().data, x_crop.cpu().numpy(), bbox

    def __call__(self, x, hp, retain_graph=True):
        return self.forward(x, hp, retain_graph)
