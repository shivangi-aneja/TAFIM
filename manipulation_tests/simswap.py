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
from utils.image_utils import tensor2im
from model_archs.simswap.fs_model import fsModel
from configs.paths_config import MANIPULATION_TESTS_BASE_DIR
from configs.common_config import device, dataset_type, val_imgs


if __name__ == '__main__':

    config_parser = ArgumentParser()
    config_parser.add_argument('-b', '--batch_size', default=1, type=int, help='Batch Size')

    opts = config_parser.parse_args()
    batch_size = opts.batch_size
    opts_dict = vars(opts)
    pprint.pprint(opts_dict)

    dataset_args = data_config.DATASETS[dataset_type]
    transforms_dict = dataset_args['transforms']().get_transforms()

    test_dataset = ImagesDataset(source_root=dataset_args['test_source_root'], target_root=dataset_args['test_target_root'], source_transform=transforms_dict['transform_test'], target_transform=transforms_dict['transform_test'], num_imgs=val_imgs)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)
    simswap_net = fsModel(perturb_wt=1, attack_loss_type='l2')
    simswap_net.eval().to(device)
    os.makedirs(os.path.join(MANIPULATION_TESTS_BASE_DIR, 'visuals_simswap'), exist_ok=True)

    for idx, data in enumerate(tqdm(test_loader, desc='')):  # inner loop within one epoch
        with torch.no_grad():

            simswap_net.real_A1 = data['A'].to(device).clone()
            simswap_net.real_A2 = data['B'].to(device).clone()

            # Get the output
            output = simswap_net.swap_face(simswap_net.real_A1, simswap_net.real_A2)

            # Visualize the results
            images = torch.cat((simswap_net.real_A1, simswap_net.real_A2, output), -1)
            horizontal_grid = tensor2im(vutils.make_grid(images), True)
            horizontal_grid = cv2.cvtColor(horizontal_grid, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(MANIPULATION_TESTS_BASE_DIR, 'visuals_simswap', f"{idx}.png"), horizontal_grid)
