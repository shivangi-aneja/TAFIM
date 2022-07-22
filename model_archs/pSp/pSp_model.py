# Code borrowed from https://github.com/eladrich/pixel2style2pixel

import torch
import random
from torch import nn
from collections import OrderedDict
import torchvision.transforms as transforms
from configs.common_config import device, resize_size
from configs.pSp_config import *
from configs.paths_config import pSp_ffhq_encode_pth
from utils.pSp_utils.encoders import psp_encoders
from model_archs.styleGan.styleGAN_model import Generator
from utils.jpeg_utils.jpeg_loss_utils import quality_to_factor, diff_round
from utils.jpeg_utils.compress import compress_jpeg
from utils.jpeg_utils.decompress import decompress_jpeg


def get_keys(d, name):
    if 'state_dict' in d:
        d = d['state_dict']
    d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
    return d_filt


class pSp(nn.Module):

    def __init__(self, perturb_wt, attack_loss_type='l2'):
        super(pSp, self).__init__()
        # Define architecture
        self.encoder = self.set_encoder()
        self.decoder = Generator(output_size, 512, 8)
        self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.perturb_wt = perturb_wt

        # Load weights
        self.load_weights()

        # Loss Function
        if attack_loss_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif attack_loss_type == 'l2':
            self.criterion = torch.nn.MSELoss()

        self.rounding = {'fn': diff_round, 'type': 'sin'}
        factor = quality_to_factor(30)
        self.compress = compress_jpeg(rounding=self.rounding, factor=factor)
        self.decompress = decompress_jpeg(resize_size, resize_size, factor=factor)
        self.jpeg_img_transform = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.loss_names = ['recon', 'perturb', 'full']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'adv_A']

    def set_encoder(self):
        if encoder_type == 'GradualStyleEncoder':
            encoder = psp_encoders.GradualStyleEncoder(50, 'ir_se')
        elif encoder_type == 'BackboneEncoderUsingLastLayerIntoW':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoW(50, 'ir_se')
        elif encoder_type == 'BackboneEncoderUsingLastLayerIntoWPlus':
            encoder = psp_encoders.BackboneEncoderUsingLastLayerIntoWPlus(50, 'ir_se')
        else:
            raise Exception('{} is not a valid encoders'.format(encoder_type))
        return encoder

    def load_weights(self):
        checkpoint_path = pSp_ffhq_encode_pth
        print('Loading pSp from checkpoint: {}'.format(checkpoint_path))
        ckpt = torch.load(checkpoint_path, map_location='cpu')
        self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)
        self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
        self.__load_latent_avg(ckpt)

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

    def closure(self, model_adv_noise, global_adv_noise):
        self.y_hat_model = self.forward(torch.clamp(self.x + model_adv_noise,  -1, 1))
        self.loss_recon_model = self.criterion(self.y_hat_model, self.y)
        self.loss_perturb_model = self.criterion(torch.clamp(self.x + model_adv_noise,  -1, 1), self.x)

        self.y_hat_global = self.forward(torch.clamp(self.x + global_adv_noise, -1, 1))
        self.loss_recon_global = self.criterion(self.y_hat_global, self.y)
        self.loss_perturb_global = self.criterion(torch.clamp(self.x + global_adv_noise, -1, 1), self.x)

        self.loss_recon = self.loss_recon_model + self.loss_recon_global
        self.loss_perturb = self.loss_perturb_model + self.loss_perturb_global
        self.loss_full = self.loss_recon + self.perturb_wt * self.loss_perturb
        return self.loss_full

    def closure_jpeg_randomized(self, model_adv_noise, global_adv_noise):
        """
            Feed in the un-normalized [0, 255] RGB color image batch
        """
        rand_num = random.randint(1, 99)
        factor = quality_to_factor(rand_num)
        self.compress = compress_jpeg(rounding=self.rounding, factor=factor).to(device)
        self.decompress = decompress_jpeg(resize_size, resize_size, factor=factor).to(device)

        # Normalize b/w [0,1] for Jpeg compression
        self.adv_x_model = torch.clamp(self.x + model_adv_noise,  -1, 1)
        self.adv_x_model = ((self.adv_x_model + 1) / 2)
        self.adv_x_model[self.adv_x_model < 0] = 0
        self.adv_x_model[self.adv_x_model > 1] = 1

        self.adv_x_global = torch.clamp(self.x + global_adv_noise, -1, 1)
        self.adv_x_global = ((self.adv_x_global + 1) / 2)
        self.adv_x_global[self.adv_x_global < 0] = 0
        self.adv_x_global[self.adv_x_global > 1] = 1

        # Compress Adversarial Image (Model)
        y1, cb1, cr1 = self.compress(self.adv_x_model)
        self.adv_x_model_comp = self.decompress(y1, cb1, cr1)
        self.adv_x_model_comp = self.jpeg_img_transform(self.adv_x_model_comp)

        # Compress Adversarial Image (Global)
        y1, cb1, cr1 = self.compress(self.adv_x_global)
        self.adv_x_global_comp = self.decompress(y1, cb1, cr1)
        self.adv_x_global_comp = self.jpeg_img_transform(self.adv_x_global_comp)

        # Compute loss
        self.y_hat_model = self.forward(torch.clamp(self.adv_x_model_comp, -1, 1))
        self.loss_recon_model = self.criterion(self.y_hat_model, self.y)
        self.loss_perturb_model = self.criterion(torch.clamp(self.adv_x_model_comp, -1, 1), self.x)

        self.y_hat_global = self.forward(torch.clamp(self.adv_x_global_comp, -1, 1))
        self.loss_recon_global = self.criterion(self.y_hat_global, self.y)
        self.loss_perturb_global = self.criterion(torch.clamp(self.adv_x_global_comp, -1, 1), self.x)

        self.loss_recon = self.loss_recon_model + self.loss_recon_global
        self.loss_perturb = self.loss_perturb_model + self.loss_perturb_global
        self.loss_full = self.loss_recon + self.perturb_wt * self.loss_perturb
        return self.loss_full

    def closure_attention(self, method_specific_noise, final_noise):
        self.y_hat = self.forward(torch.clamp(self.x + final_noise,  -1, 1))
        y_hat_method = self.forward(torch.clamp(self.x + method_specific_noise,  -1, 1))
        self.loss_recon = self.criterion(self.y_hat, self.y) + self.criterion(y_hat_method, self.y)
        self.loss_perturb = self.criterion(torch.clamp(self.x + final_noise,  -1, 1), self.x)
        self.loss_full = self.loss_recon + self.perturb_wt * self.loss_perturb
        return self.loss_full

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
                inject_latent=None, return_latents=False, alpha=None):

        if input_code:
            codes = x
        else:
            codes = self.encoder(x)
            # normalize with respect to the center of an average face
            if start_from_latent_avg:
                if learn_in_w:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
                else:
                    codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)

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
        images, result_latent = self.decoder([codes], input_is_latent=input_is_latent, randomize_noise=randomize_noise, return_latents=return_latents)

        if resize:
            images = self.face_pool(images)

        if return_latents:
            return images, result_latent
        else:
            return images

    def forward_mix(self, x, y, resize=True, input_code=False, randomize_noise=True, return_latents=False):

        codes = self.encoder(x)
        codes_y = self.encoder(y)
        # normalize with respect to the center of an average face
        if start_from_latent_avg:
            if learn_in_w:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1)
            else:
                codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
                codes_y = codes_y + self.latent_avg.repeat(codes_y.shape[0], 1, 1)

        latent_mask = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        alpha = 0.8
        for i in latent_mask:
            codes[:, i] = alpha * codes_y[:, i] + (1 - alpha) * codes[:, i]

        input_is_latent = not input_code
        images, result_latent = self.decoder([codes], input_is_latent=input_is_latent, randomize_noise=randomize_noise, return_latents=return_latents)

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