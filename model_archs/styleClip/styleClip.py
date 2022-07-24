import torch
import random
from torch import nn
import torch.nn.functional as F
from argparse import Namespace
from collections import OrderedDict
from model_archs.styleClip.pSp import pSp
from configs.common_config import input_nc, output_nc, device, resize_size
from model_archs.styleClip.styleclip_mapper import StyleCLIPMapper
from configs.attack_config import no_dropout, init_type, init_gain, ngf, norm
from configs.paths_config import STYLECLIP_BASE_DIR


meta_data = {
  'afro': ['afro', False, False, True],
  'angry': ['angry', False, False, True],
  'Beyonce': ['beyonce', False, False, False],
  'bobcut': ['bobcut', False, False, True],
  'bowlcut': ['bowlcut', False, False, True],
  'curly hair': ['curly_hair', False, False, True],
  'Hilary Clinton': ['hilary_clinton', False, False, False],
  'Jhonny Depp': ['depp', False, False, False],
  'mohawk': ['mohawk', False, False, True],
  'purple hair': ['purple_hair', False, False, False],
  'surprised': ['surprised', False, False, True],
  'Taylor Swift': ['taylor_swift', False, False, False],
  'trump': ['trump', False, False, False],
  'Mark Zuckerberg': ['zuckerberg', False, False, False]
}


class StyleClip(nn.Module):

    def __init__(self, perturb_wt, attack_loss_type='l2', is_test=False):
        super(StyleClip, self).__init__()

        self.perturb_wt = perturb_wt
        self.loss_names = ['recon', 'perturb', 'full']
        self.visual_names = ['real_A', 'fake_B', 'real_B', 'adv_A']
        self.style_list = ['angry', 'Beyonce', 'bobcut', 'bowlcut', 'curly hair', 'Hilary Clinton', 'mohawk']
        if is_test:
            test_style_list = ['purple hair', 'surprised', 'Taylor Swift', 'trump', 'Mark Zuckerberg']
            edit_type = random.choice(test_style_list)
            args_styleclip = {"work_in_stylespace": False, "exp_dir": "results/", "couple_outputs": True, "mapper_type": "LevelsMapper", "no_coarse_mapper": meta_data[edit_type][1],
                              "no_medium_mapper": meta_data[edit_type][2], "no_fine_mapper": meta_data[edit_type][3], "stylegan_size": 1024, "test_batch_size": 1, "test_workers": 1}
            edit_id = meta_data[edit_type][0]
            style_path = STYLECLIP_BASE_DIR + '/' + edit_id + '.pt'
            ckpt_style = torch.load(style_path, map_location='cpu')
            opts_style = ckpt_style['opts']
            opts_style['checkpoint_path'] = style_path
            opts_style.update(args_styleclip)
            opts_style = Namespace(**opts_style)
            self.test_styleclip_mapper = StyleCLIPMapper(opts_style).to(device).eval()

        e4e_path = STYLECLIP_BASE_DIR + '/e4e_ffhq_encode.pt'
        ckpt_e4e = torch.load(e4e_path, map_location='cpu')
        opts_e4e = ckpt_e4e['opts']
        opts_e4e['checkpoint_path'] = e4e_path
        self.pSp_model = pSp(Namespace(**opts_e4e)).eval()
        # Loss Function
        if attack_loss_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif attack_loss_type == 'l2':
            self.criterion = torch.nn.MSELoss()

    def set_input(self, x, y):
        self.x = x.to(device)
        self.y = y.to(device)

    def forward(self, image):

        # For randomly on different styles
        edit_type = random.choice(self.style_list)
        args_styleclip = {"work_in_stylespace": False, "exp_dir": "results/", "couple_outputs": True,
                          "mapper_type": "LevelsMapper", "no_coarse_mapper": meta_data[edit_type][1],
                          "no_medium_mapper": meta_data[edit_type][2],
                          "no_fine_mapper": meta_data[edit_type][3],
                          "stylegan_size": 1024, "test_batch_size": 1, "test_workers": 1}
        edit_id = meta_data[edit_type][0]
        style_path = STYLECLIP_BASE_DIR + '/' + edit_id + '.pt'
        ckpt_style = torch.load(style_path, map_location='cpu')
        opts_style = ckpt_style['opts']
        opts_style['checkpoint_path'] = style_path
        opts_style.update(args_styleclip)
        opts_style = Namespace(**opts_style)
        styleclip_mapper = StyleCLIPMapper(opts_style).to(device).eval()

        _, w = self.pSp_model(image)
        w_hat = w + 0.1 * styleclip_mapper.mapper(w)
        x_hat, w_hat, _ = styleclip_mapper.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
        return x_hat

    def forward_test(self, image):
        # For randomly on different styles
        _, w = self.pSp_model(image)
        w_hat = w + 0.1 * self.test_styleclip_mapper.mapper(w)
        x_hat, w_hat, _ = self.test_styleclip_mapper.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
        return x_hat

    def closure(self, model_adv_noise, global_adv_noise):
        self.y_hat_model = self.forward(torch.clamp(self.x + model_adv_noise, -1, 1))
        self.loss_recon_model = self.criterion(self.y_hat_model, self.y)
        self.loss_perturb_model = self.criterion(torch.clamp(self.x + model_adv_noise, -1, 1), self.x)

        self.y_hat_global = self.forward(torch.clamp(self.x + global_adv_noise, -1, 1))
        self.loss_recon_global = self.criterion(self.y_hat_global, self.y)
        self.loss_perturb_global = self.criterion(torch.clamp(self.x + global_adv_noise, -1, 1), self.x)

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
                if name == 'fake_B' or name == 'real_B':
                    visual_ret['styleclip_' + name] = F.interpolate(getattr(self, name), size=resize_size)
                else:
                    visual_ret['styleclip_' + name] = getattr(self, name)
        return visual_ret