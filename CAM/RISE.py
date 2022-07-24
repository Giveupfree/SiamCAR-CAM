import numpy as np
import torch
import torch.nn as nn
from pysot.core.config import cfg
from skimage.transform import resize  # scikit-image
from tqdm import tqdm
from torchvision.transforms import transforms
from CAM.BaseCAM import UnNormalize

class RISE(nn.Module):
    def __init__(self, model, target_layer="module.layer4.2", input_size=(cfg.TRACK.INSTANCE_SIZE, cfg.TRACK.INSTANCE_SIZE), batch_size=1, N=8000, s=7, p1=0.1):
        super(RISE, self).__init__()
        assert N % batch_size == 0
        self.model = model
        self.input_size = input_size
        self.batch_size = batch_size
        self.N = N
        self.s = s
        self.p1 = p1
        self.transform_norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.Nutransform = UnNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.generate_masks()

    def generate_masks(self):
        cell_size = np.ceil(np.array(self.input_size) / self.s)
        up_size = (self.s + 1) * cell_size

        grid = np.random.rand(self.N, self.s, self.s) < self.p1
        grid = grid.astype('float32')

        self.masks = np.empty((self.N, *self.input_size))

        for i in tqdm(range(self.N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        # np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float()
        self.N = self.masks.shape[0]

    def forward(self, x, hp):
        output = self.model.track_cam(x, hp)
        N = self.N
        cls = output["cls"]
        # idx = output["idx"]
        x_crop = output["x_crop"]
        bbox = output["bbox"]

        _, _, H, W = x_crop.size()
        # Apply array of filters to the image
        saliency = torch.zeros(1, 1, H, W).cuda()
        idx = torch.argmax(cls)
        x_crop = torch.cat([x_crop[:, 2, :, :][:, None, :, :],
                            x_crop[:, 1, :, :][:, None, :, :],
                            x_crop[:, 0, :, :][:, None, :, :]], dim=1) / 255.0
        # 归一化和反归一化中的squeeze(0)和unsqueeze(0)用于兼容不同版本的torchvision
        norm_img = self.transform_norm(x_crop.squeeze(0)).unsqueeze(0)
        for i in range(0, self.N, self.batch_size):
            mask = self.masks[i: min(i+self.batch_size, N)]
            mask = mask.cuda()
            with torch.no_grad():
                img = self.Nutransform((mask * norm_img).squeeze(0)).unsqueeze(0)
                base_line = self.model.model.track(torch.cat([img[:, 2, :, :][:, None, :, :],
                                                              img[:, 1, :, :][:, None, :, :],
                                                              img[:, 0, :, :][:, None, :, :]], dim=1) * 255.0)["cls"][:,
                            1, :, :][:, None, :, :]

            score = base_line.reshape(-1)[idx]
            saliency += (score * mask).sum(dim=0, keepdims=True)
        return (saliency / N / self.p1).cpu().data, x_crop.cpu().numpy(), bbox