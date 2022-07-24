import torch
import torch.nn.functional as F

from CAM.BaseCAM import BaseCAM


class ScoreCAM(BaseCAM):
    """
        ScoreCAM, inherit from BaseCAM
    """

    def __init__(self, model, target_layer="module.layer4.2",Norm=True):
        super(ScoreCAM, self).__init__(model, target_layer, Norm)

    def forward(self, x, hp, retain_graph=False):
        output = self.model.track_cam(x, hp)
        cls = output["cls"]
        # idx = output["idx"]
        x_crop = output["x_crop"]
        bbox = output["bbox"]
        b, c, h, w = x_crop.size()

        idx = torch.argmax(cls)
        score = cls.reshape(-1)[idx]
        self.model.model.zero_grad()

        score.backward(retain_graph=retain_graph)
        activations = self.activations['value']
        b, k, u, v = activations.size()

        score_saliency_map = torch.zeros((1, 1, h, w))

        if torch.cuda.is_available():
            activations = activations.cuda()
            score_saliency_map = score_saliency_map.cuda()

        x_crop = torch.cat([x_crop[:, 2, :, :][:, None, :, :],
                            x_crop[:, 1, :, :][:, None, :, :],
                            x_crop[:, 0, :, :][:, None, :, :]], dim=1) / 255.0
        norm_img = self.transform_norm(x_crop.squeeze(0)).unsqueeze(0)
        with torch.no_grad():
            for i in range(k):
                # upsampling
                saliency_map = torch.unsqueeze(activations[:, i, :, :], 1)
                saliency_map = F.interpolate(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
                if saliency_map.max() == saliency_map.min():
                    continue
                # normalize to 0-1
                norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + self.esp)

                # how much increase if keeping the highlighted region
                # predication on masked x
                # 归一化和反归一化中的squeeze(0)和unsqueeze(0)用于兼容不同版本的torchvision
                img = self.Nutransform((norm_saliency_map * norm_img).squeeze(0)).unsqueeze(0)
                score = self.model.model.track(torch.cat([img[:, 2, :, :][:, None, :, :],
                                     img[:, 1, :, :][:, None, :, :],
                                     img[:, 0, :, :][:, None, :, :]], dim=1) * 255.0)["cls"][:, 1, :, :][:, None, :, :].reshape(-1)[idx]

                score_saliency_map += score * saliency_map

        score_saliency_map = F.relu(score_saliency_map)
        score_saliency_map_min, score_saliency_map_max = score_saliency_map.min(), score_saliency_map.max()

        if score_saliency_map_min == score_saliency_map_max:
            return None, None, None

        score_saliency_map = (score_saliency_map - score_saliency_map_min) / (
                score_saliency_map_max - score_saliency_map_min + self.esp).data

        return score_saliency_map.cpu().data, x_crop.cpu().numpy(), bbox

    def __call__(self, x, hp, retain_graph=True):
        return self.forward(x, hp, retain_graph)