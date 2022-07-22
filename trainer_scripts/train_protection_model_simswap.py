import os
import torch
import pprint
from tqdm import tqdm
from configs import data_config
from torch.optim import SGD, Adam
from argparse import ArgumentParser
from torch.autograd import Variable
from utils.tf_logger import Logger
from torch.utils.data import DataLoader
from utils.dataset import ImagesDataset
from model_archs.simswap.fs_model import fsModel
from configs.paths_config import SIMSWAP_ATTACK_BASE_DIR, TF_BASE_DIR
from model_archs.protection_net.networks import define_G as GenPix2Pix
from configs.attack_config import no_dropout, init_type, init_gain, ngf, net_noise, norm
from configs.common_config import device, val_imgs, n_epochs, dataset_type, architecture_type, resize_size
from utils.common_utis import init_adversarial_noise
from utils.image_utils import create_target_img
from configs.transforms_config import img_transform_simswap


if __name__ == '__main__':

    config_parser = ArgumentParser()
    config_parser.add_argument('--train_imgs', default=5000, type=int, help='Number of Training Images')
    config_parser.add_argument('--perturb_wt', default=10, type=int, help='Perturbation Weight')
    config_parser.add_argument('--batch_size', default=1, type=int, help='Perturbation Weight')
    config_parser.add_argument('--loss_type', default='l2', type=str, help='Loss Type')
    config_parser.add_argument('--lr', default=0.0001, type=float, help='Learning Rate')
    opts = config_parser.parse_args()

    train_imgs = opts.train_imgs
    perturb_wt = opts.perturb_wt
    batch_size = opts.batch_size
    loss_type = opts.loss_type
    lr = opts.lr

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)

    dataset_args = data_config.DATASETS[dataset_type]
    transforms_dict = dataset_args['transforms']().get_transforms()

    train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'], target_root=dataset_args['train_target_root'], source_transform=transforms_dict['transform_train'], target_transform=transforms_dict['transform_train'], num_imgs=train_imgs)
    val_dataset = ImagesDataset(source_root=dataset_args['val_source_root'], target_root=dataset_args['val_target_root'], source_transform=transforms_dict['transform_test'], target_transform=transforms_dict['transform_test'], num_imgs=val_imgs)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    model_ckpt_pth = os.path.join(SIMSWAP_ATTACK_BASE_DIR, net_noise)
    os.makedirs(model_ckpt_pth, exist_ok=True)

    simswap_net = fsModel(perturb_wt=perturb_wt, attack_loss_type=loss_type)
    simswap_net.eval().to(device)

    dataset_size = len(train_loader)  # get the number of images in the dataset.
    print('The number of train Iterations = %d' % dataset_size)
    optim_list = [(SGD, {'lr': lr}), (Adam, {'lr': lr})]
    my_optim = optim_list[1]
    optimizer = my_optim[0]
    optim_args = my_optim[1]
    logger_train = Logger(model_name=architecture_type, data_name=dataset_type, log_path=os.path.join(TF_BASE_DIR, 'attacks/Simswap/', net_noise, 'train', optimizer.__name__ + "_perturb" + str(perturb_wt) + "_batch" + str(batch_size) + "_lr_" + str(optim_args['lr']) + "_" + str(loss_type) + "loss"))
    logger_val = Logger(model_name=architecture_type, data_name=dataset_type, log_path=os.path.join(TF_BASE_DIR, 'attacks/SimSwap/', net_noise, 'val', optimizer.__name__ + "_perturb" + str(perturb_wt) + "_batch" + str(batch_size) + "_lr_" + str(optim_args['lr']) + "_" + str(loss_type) + "loss"))

    best_loss = float("inf")
    protection_model = GenPix2Pix(6, 3, ngf, net_noise, norm, not no_dropout, init_type, init_gain).to(device)
    adv_noise = init_adversarial_noise(mode="rand").to(device)
    global_adv_noise = Variable(adv_noise.to(device), requires_grad=True)
    model_optim = optimizer(params=list(protection_model.parameters()) + list([global_adv_noise]), **optim_args)

    for epoch in range(n_epochs):  # outer loop for different epochs;
        train_current_loss = {'full': 0., 'recon': 0., 'perturb': 0.}
        for i, data in enumerate(tqdm(train_loader, desc='')):  # inner loop within one epoch

            simswap_net.real_A1 = data['A'].to(device).clone()
            simswap_net.real_A2 = data['B'].to(device).clone()

            b_size = simswap_net.real_A1.shape[0]
            simswap_net.y = create_target_img(batch_size=b_size, size=resize_size, img_transform=img_transform_simswap, color=(255, 0, 0)).to(device)  # Ideal output to produce

            # Compute the output for the original image
            with torch.no_grad():
                old_out = simswap_net.swap_face(simswap_net.real_A1, simswap_net.real_A2)

            # Perform update and generate adversarial noise
            model_optim.zero_grad()
            adv_input = torch.cat((simswap_net.real_A1, global_adv_noise.unsqueeze(0)), 1)
            adv_noise = protection_model(adv_input)
            simswap_net.closure(model_adv_noise=adv_noise, global_adv_noise=global_adv_noise)
            simswap_net.loss_full.backward()
            model_optim.step()
            simswap_net.adv_A1 = torch.clamp(simswap_net.real_A1 + adv_noise, 0, 1)  # For Visualization # For Visualization

            #  Redo Forward pass on adversarial image through the model
            simswap_net.fake_B = simswap_net.swap_face(simswap_net.adv_A1, simswap_net.real_A2)

            visuals = simswap_net.get_current_visuals()  # get image results
            train_losses = simswap_net.get_current_losses()
            train_current_loss['full'] += train_losses['full']
            train_current_loss['recon'] += train_losses['recon']
            train_current_loss['perturb'] += train_losses['perturb']

        train_current_loss['full'] /= len(train_loader)
        train_current_loss['recon'] /= len(train_loader)
        train_current_loss['perturb'] /= len(train_loader)

        logger_train.display_current_results(simswap_net.get_current_visuals(), int(epoch))
        logger_train.plot_current_losses(int(epoch), train_current_loss)

        val_current_loss = {'full': 0., 'recon': 0., 'perturb': 0.}
        for i, data in enumerate(tqdm(val_loader, desc='')):  # inner loop within one epoch
            with torch.no_grad():
                simswap_net.real_A1 = data['A'].to(device)
                simswap_net.real_A2 = data['B'].to(device)
                b_size = simswap_net.real_A1.shape[0]
                simswap_net.y = create_target_img(batch_size=b_size, size=resize_size, img_transform=img_transform_simswap, color=(255, 0, 0)).to(device)  # Ideal output to produce

                # Get the old output
                old_out = simswap_net.swap_face(simswap_net.real_A1, simswap_net.real_A2)

                adv_input = torch.cat((simswap_net.real_A1, global_adv_noise.unsqueeze(0)), 1)
                adv_noise = protection_model(adv_input)
                simswap_net.adv_A1 = torch.clamp(simswap_net.real_A1 + adv_noise, 0, 1)  # For Visualization

                #  Redo Forward pass on adversarial image through the model
                simswap_net.fake_B = simswap_net.swap_face(simswap_net.adv_A1, simswap_net.real_A2)

                simswap_net.closure(model_adv_noise=adv_noise, global_adv_noise=global_adv_noise)
                visuals = simswap_net.get_current_visuals()  # get image results
                val_losses = simswap_net.get_current_losses()
                val_current_loss['full'] += val_losses['full']
                val_current_loss['recon'] += val_losses['recon']
                val_current_loss['perturb'] += val_losses['perturb']

        val_current_loss['full'] /= len(val_loader)
        val_current_loss['recon'] /= len(val_loader)
        val_current_loss['perturb'] /= len(val_loader)

        logger_val.display_current_results(simswap_net.get_current_visuals(), int(epoch))
        logger_val.plot_current_losses(int(epoch), val_current_loss)

        # If the new loss is better than old loss, update the adversarial noise
        if val_current_loss['full'] < best_loss:
            save_filename_model = 'combined_%s_net_%sperturb_%simgs_%s_%sbatch_%s_loss_%s_lr.pth' % (optimizer.__name__, perturb_wt, train_imgs, net_noise, batch_size, loss_type, optim_args['lr'])
            save_path = os.path.join(model_ckpt_pth, save_filename_model)
            print('Updating the noise model')
            torch.save({"protection_net": protection_model.state_dict(),
                        "global_noise": adv_noise.detach()}, save_path)
            best_loss = val_current_loss['full']

        print('Epoch {} / {} \t Train Loss: {:.3f} \t Val Acc: {:.3f}'.format(epoch, n_epochs, train_current_loss['full'], val_current_loss['full']))

        save_filename_model = 'combined_%s_net_%sperturb_%simgs_%s_%sbatch_%s_loss_%s_lr_latest.pth' % (optimizer.__name__, perturb_wt, train_imgs, net_noise, batch_size, loss_type, optim_args['lr'])
        save_path = os.path.join(model_ckpt_pth, save_filename_model)
        torch.save({"protection_net": protection_model.state_dict(),
                    "global_noise": adv_noise.detach()}, save_path)
