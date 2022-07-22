import torch
import torch.nn as nn
from configs.attack_config import net_noise
from model_archs.protection_net.networks import define_G as GenPix2Pix
from configs.attack_config import norm, no_dropout, init_type, init_gain, ngf


class ProtectionModelIndividual(nn.Module):
    """
        For training combined model for all manipulations simultaneously with attention
    """
    def __init__(self, optimizer, optim_args):
        super().__init__()
        self.attn_model = GenPix2Pix(4, 1, ngf, net_noise, norm, not no_dropout, init_type, init_gain, att_mode=True)
        self.fusion_model = GenPix2Pix(3, 3, ngf, net_noise, norm, not no_dropout, init_type, init_gain)
        self.model_optim = optimizer(params=list(self.attn_model.parameters()) + list(self.fusion_model.parameters()), **optim_args)
        self.tanh = nn.Tanh()

    def forward(self, noises, combined_noise_label_list):
        z_noise = torch.stack([self.attn_model(noise_label) for noise_label in combined_noise_label_list])
        alpha_noise = torch.softmax(z_noise, dim=0)
        unrefined_delta = torch.sum(noises*alpha_noise, dim=0)
        final_delta = self.fusion_model(unrefined_delta)
        return final_delta
