import os
import pprint
import torch
import cv2
from tqdm import tqdm
import torch.nn.functional as F
from configs import data_config
import torchvision.utils as vutils
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils.dataset import ImagesDataset
from utils.image_utils import tensor2im
from model_archs.styleClip.styleClip import StyleClip
from configs.paths_config import MANIPULATION_TESTS_BASE_DIR
from configs.common_config import device, dataset_type, val_imgs, resize_size


if __name__ == '__main__':

    config_parser = ArgumentParser()
    config_parser.add_argument('-b', '--batch_size', default=1, type=int, help='Batch Size')

    opts = config_parser.parse_args()
    batch_size = opts.batch_size
    opts_dict = vars(opts)
    pprint.pprint(opts_dict)

    dataset_args = data_config.DATASETS[dataset_type]
    transforms_dict = dataset_args['transforms']().get_transforms()

    test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'], target_root=dataset_args['test_target_root'], source_transform=transforms_dict['transform_inference'], target_transform=transforms_dict['transform_inference'], num_imgs=val_imgs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    styleclip_net = StyleClip(perturb_wt=1, attack_loss_type='l2')
    styleclip_net.eval().to(device)
    os.makedirs(os.path.join(MANIPULATION_TESTS_BASE_DIR, 'visuals_styleClip'), exist_ok=True)

    for idx, data in enumerate(tqdm(test_loader, desc='')):  # inner loop within one epoch
        with torch.no_grad():
            x_orig = data['A'].to(device).clone().detach()
            y_orig = data['B'].to(device).clone().detach()
            styleclip_net.set_input(x_orig, y_orig)

            # Get the output
            output = styleclip_net(x_orig)

            # Visualize the results
            images = torch.cat((x_orig, y_orig, F.interpolate(output, resize_size)), -1)
            horizontal_grid = tensor2im(vutils.make_grid(images))
            horizontal_grid = cv2.cvtColor(horizontal_grid, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(MANIPULATION_TESTS_BASE_DIR, 'visuals_styleClip', f"{idx}.png"), horizontal_grid)
