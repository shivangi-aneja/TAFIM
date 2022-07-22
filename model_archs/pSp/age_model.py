# Code borrowed from https://github.com/yuval-alaluf/SAM

import torch
import numpy as np
from torch import nn
from collections import OrderedDict
from configs.common_config import device
from configs.pSp_config import *
from utils.pSp_utils.encoders import age_encoders
from model_archs.styleGan.styleGAN_model import Generator
from configs.paths_config import BASE_DIR


class AgeTransformer(object):

    def __init__(self, target_age):
        self.target_age = target_age

    def __call__(self, img):
        img = self.add_aging_channel(img)
        return img

    def add_aging_channel(self, img):
        target_age = self.__get_target_age()
        target_age = int(target_age) / 100  # normalize aging amount to be in range [-1,1]
        img = torch.cat((img.squeeze(), target_age * torch.ones((1, img.shape[2], img.shape[3])).to(device)), 0).unsqueeze(0)
        return img

    def __get_target_age(self):
        if self.target_age == "uniform_random":
            return np.random.randint(low=0., high=101, size=1)[0]
        else:
            return self.target_age


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, perturb_wt, load_pretrained=False):
        super(pSp, self).__init__()
        # Define architecture
        self.encoder = age_encoders.GradualStyleEncoder(50, 'ir_se', input_channels=4)
        self.pretrained_encoder = age_encoders.GradualStyleEncoder(50, 'ir_se', input_channels=3)
        self.decoder = Generator(output_size, 512, 8)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        # Load weights
        if load_pretrained:
            self.load_weights()

        self.loss_names = ['recon', 'perturb', 'full']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'adv_A']

    def load_weights(self):
        psp_checkpoint_path = BASE_DIR + '/model_checkpoints/pSp/psp_ffhq_encode.pt'
        sam_checkpoint_path = BASE_DIR + '/model_checkpoints/pSp/sam_ffhq_aging.pt'
        ckpt_psp = torch.load(psp_checkpoint_path, map_location='cpu')
        ckpt_sam = torch.load(sam_checkpoint_path, map_location='cpu')
        self.encoder.load_state_dict(get_keys(ckpt_sam, 'encoder'), strict=False)
        self.pretrained_encoder.load_state_dict(get_keys(ckpt_psp, 'encoder'), strict=True)
        self.decoder.load_state_dict(get_keys(ckpt_sam, 'decoder'), strict=True)
        self.__load_latent_avg(ckpt_sam)

    def set_input(self, x, y):
        self.x = x.to(device)
        self.y = y.to(device)

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def forward(self, x, age, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):
        age_trans = AgeTransformer(target_age=age)
        x_age = age_trans(x)
        if input_code:
            codes = x_age
        else:
            codes = self.encoder(x_age)
            # normalize with respect to the center of an average face
            with torch.no_grad():
                encoded_latents = self.pretrained_encoder(x_age[:, :-1, :, :])
                encoded_latents = encoded_latents + self.latent_avg
            codes = codes + encoded_latents

        if latent_mask is not None:
            for i in latent_mask:
                if inject_latent is not None:
                    if alpha is not None:
                        codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
                    else:
                        codes[:, i] = inject_latent[:, i]
                else:
                    codes[:, i] = 0

        input_is_latent = not input_code
        images, result_latent = self.decoder([codes],
                                             input_is_latent=input_is_latent,
                                             randomize_noise=randomize_noise,
                                             return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def __load_latent_avg(self, ckpt, repeat=None):
        if 'latent_avg' in ckpt:
            self.latent_avg = ckpt['latent_avg'].to(device)
            if repeat is not None:
                self.latent_avg = self.latent_avg.repeat(repeat, 1)
        else:
            self.latent_avg = None
