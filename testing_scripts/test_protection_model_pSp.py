import os
import pprint
import torch
import cv2
from tqdm import tqdm
from configs import data_config
import torchvision.utils as vutils
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils.dataset import ImagesDataset
from model_archs.pSp.pSp_model import pSp
from utils.image_utils import create_target_img, tensor2im
from configs.transforms_config import adv_img_transform
from configs.paths_config import PSP_ATTACK_BASE_DIR, ATTACK_BASE_DIR
from configs.common_config import device, dataset_type, val_imgs, resize_size
from model_archs.protection_net.networks import define_G as GenPix2Pix
from configs.attack_config import no_dropout, init_type, init_gain, ngf, net_noise, norm


if __name__ == '__main__':

    config_parser = ArgumentParser()
    config_parser.add_argument('-p', '--pSp_protection_path', default='pSp_protection.pth', type=str, help='Path to pSp pretrained model')
    config_parser.add_argument('-b', '--batch_size', default=1, type=int, help='Batch Size')
    config_parser.add_argument('-l', '--loss_type', default='l2', type=str, help='Loss Type')
    opts = config_parser.parse_args()

    pSp_protection_path = opts.pSp_protection_path
    loss_type = opts.loss_type
    batch_size = opts.batch_size

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)

    dataset_args = data_config.DATASETS[dataset_type]
    transforms_dict = dataset_args['transforms']().get_transforms()

    test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'], target_root=dataset_args['test_target_root'], source_transform=transforms_dict['transform_inference'], target_transform=transforms_dict['transform_inference'], num_imgs=val_imgs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    checkpoint = torch.load(os.path.join(PSP_ATTACK_BASE_DIR, pSp_protection_path))
    pSp_net = pSp(perturb_wt=checkpoint['perturb_wt'], attack_loss_type=loss_type)
    print("perturb Weight", checkpoint['perturb_wt'])
    pSp_net.eval().to(device)

    protection_model = GenPix2Pix(6, 3, ngf, net_noise, 'batch', not no_dropout, init_type, init_gain).to(device)
    protection_model.load_state_dict(checkpoint['protection_net'])
    protection_model.to(device)
    protection_model.eval()
    global_adv_noise = checkpoint['global_noise'].to(device)

    for idx, data in enumerate(tqdm(test_loader, desc='')):  # inner loop within one epoch
        with torch.no_grad():
            x_orig = data['A'].to(device).clone().detach()
            y_orig = data['B'].to(device).clone().detach()

            batch_size = x_orig.shape[0]
            y_target = create_target_img(batch_size=batch_size, size=resize_size,  img_transform=adv_img_transform, color=(255, 255, 255))  # Ideal output to produce

            pSp_net.set_input(x_orig, y_target)

            # Get the old output
            old_out = pSp_net(pSp_net.x)

            pSp_net.real_A = x_orig
            pSp_net.real_B = y_orig

            adv_input = torch.cat((pSp_net.x, global_adv_noise.unsqueeze(0)), 1)
            adv_noise = protection_model(adv_input)
            pSp_net.closure(model_adv_noise=adv_noise, global_adv_noise=global_adv_noise)

            pSp_net.adv_A = torch.clamp(pSp_net.x + adv_noise, -1, 1)  # For Visualization

            #  Redo Forward pass on adversarial image through the model
            pSp_net.fake_B = pSp_net(pSp_net.adv_A)
            visuals = pSp_net.get_current_visuals()  # get image results

            img_keys = list(visuals.keys())
            images = torch.cat((x_orig, pSp_net.adv_A, pSp_net.adv_A - x_orig, visuals['fake_B'], old_out, y_orig), -1)
            horizontal_grid = tensor2im(vutils.make_grid(images))
            horizontal_grid = cv2.cvtColor(horizontal_grid, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(ATTACK_BASE_DIR, 'visuals', f"{idx}.png"), horizontal_grid)
