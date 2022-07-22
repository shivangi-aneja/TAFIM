import torch
from collections import OrderedDict
import torch.nn.functional as F
from configs.paths_config import simswap_ckpt, simswap_arcface_ckpt
from .fs_networks import Generator_Adain_Upsample
from .models import ResNet, IRBlock


class fsModel(torch.nn.Module):

    def __init__(self, perturb_wt,  attack_loss_type='l2'):
        super(fsModel, self).__init__()
        self.visual_names = ['real_A1', 'real_A2', 'adv_A1', 'fake_B']
        self.loss_names = ['recon', 'perturb', 'full']

        # Generator network
        self.netG = Generator_Adain_Upsample(input_nc=3, output_nc=3, latent_size=512, n_blocks=9, deep=False)
        self.netG.load_state_dict(torch.load(simswap_ckpt))
        # Id network
        netArc_checkpoint = torch.load(simswap_arcface_ckpt)
        self.netArc = ResNet(IRBlock, [3, 4, 23, 3])
        self.netArc.load_state_dict(netArc_checkpoint)
        self.perturb_wt = perturb_wt

        # Loss Function
        if attack_loss_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif attack_loss_type == 'l2':
            self.criterion = torch.nn.MSELoss()

    def swap_face(self, img_att, img_id):
        # create latent id
        img_id_downsample = F.interpolate(img_id, scale_factor=0.5)
        latent_id = self.netArc(img_id_downsample)
        latent_id = latent_id.detach()
        latent_id = latent_id / torch.linalg.norm(latent_id, axis=1, keepdims=True)
        img_fake = self.netG.forward(img_att, latent_id)
        return img_fake

    def closure(self, model_adv_noise, global_adv_noise):
        self.y_hat_model = self.swap_face(torch.clamp(self.real_A1 + model_adv_noise, 0, 1), self.real_A2)
        self.loss_recon_model = self.criterion(self.y_hat_model, self.y)
        self.loss_perturb_model = self.criterion(torch.clamp(self.real_A1 + model_adv_noise, 0, 1), self.real_A1)

        self.y_hat_global = self.swap_face(torch.clamp(self.real_A1 + global_adv_noise, 0, 1), self.real_A2)
        self.loss_recon_global = self.criterion(self.y_hat_global, self.y)
        self.loss_perturb_global = self.criterion(torch.clamp(self.real_A1 + global_adv_noise, 0, 1), self.real_A1)

        self.loss_recon = self.loss_recon_model + self.loss_recon_global
        self.loss_perturb = self.loss_perturb_model + self.loss_perturb_global
        self.loss_full = self.loss_recon + self.perturb_wt * self.loss_perturb
        return self.loss_full

    def closure_attention(self, method_specific_noise, final_noise):
        self.y_hat = self.swap_face(torch.clamp(self.real_A1 + final_noise, 0, 1), self.real_A2)
        y_hat_method = self.swap_face(torch.clamp(self.real_A1 + method_specific_noise, 0, 1), self.real_A2)
        self.loss_recon = self.criterion(self.y_hat, self.y) + self.criterion(y_hat_method, self.y)
        self.loss_perturb = self.criterion(torch.clamp(self.real_A1 + final_noise, 0, 1), self.real_A1)
        self.loss_full = self.loss_recon + self.perturb_wt * self.loss_perturb
        return self.loss_full

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(
                    getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret
