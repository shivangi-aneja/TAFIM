import os
import pprint
import torch
from tqdm import tqdm
from torch.autograd import Variable
from configs import data_config
from torch.optim import SGD, Adam
from argparse import ArgumentParser
from utils.tf_logger import Logger
from utils.dataset import ImagesDataset
from torch.utils.data import DataLoader
from model_archs.pSp.pSp_model import pSp
from utils.image_utils import create_target_img
from utils.common_utis import init_adversarial_noise
from model_archs.protection_net.networks import define_G as GenPix2Pix
from configs.paths_config import PSP_ATTACK_BASE_DIR, TF_BASE_DIR
from configs.transforms_config import adv_img_transform
from configs.attack_config import no_dropout, init_type, init_gain, ngf, net_noise, norm
from configs.common_config import device, architecture_type, dataset_type, n_epochs, val_imgs


if __name__ == '__main__':

    config_parser = ArgumentParser()
    config_parser.add_argument('--train_imgs', default=5000, type=int, help='Number of Training Images')
    config_parser.add_argument('--perturb_wt', default=10, type=float, help='Perturbation Weight')
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

    train_dataset = ImagesDataset(source_root=dataset_args['train_source_root'], target_root=dataset_args['train_target_root'], source_transform=transforms_dict['transform_source'], target_transform=transforms_dict['transform_gt_train'], num_imgs=train_imgs)
    val_dataset = ImagesDataset(source_root=dataset_args['val_source_root'], target_root=dataset_args['val_target_root'], source_transform=transforms_dict['transform_inference'], target_transform=transforms_dict['transform_inference'], num_imgs=val_imgs)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    model_ckpt_pth = os.path.join(PSP_ATTACK_BASE_DIR, 'compressed',  net_noise)
    os.makedirs(model_ckpt_pth, exist_ok=True)

    pSp_net = pSp(perturb_wt=perturb_wt, attack_loss_type=loss_type)
    pSp_net.eval().to(device)

    dataset_size = len(train_loader)  # get the number of images in the dataset.
    print('The number of train Iterations = %d' % dataset_size)
    optim_list = [(SGD, {'lr': lr}), (Adam, {'lr': lr})]
    my_optim = optim_list[1]
    optimizer = my_optim[0]
    optim_args = my_optim[1]
    logger_train = Logger(model_name=architecture_type, data_name=dataset_type, log_path=os.path.join(TF_BASE_DIR, 'attacks/pSp_compressed/', net_noise, 'train', optimizer.__name__ + "_perturb" + str(perturb_wt) + "_batch" + str(batch_size) + "_lr_" + str(optim_args['lr']) + "_" + str(loss_type) + "loss"))
    logger_val = Logger(model_name=architecture_type, data_name=dataset_type, log_path=os.path.join(TF_BASE_DIR, 'attacks/pSp_compressed/', net_noise, 'val', optimizer.__name__ + "_perturb" + str(perturb_wt) + "_batch" + str(batch_size) + "_lr_" + str(optim_args['lr']) + "_" + str(loss_type) + "loss"))

    protection_model = GenPix2Pix(6, 3, ngf, net_noise, norm, not no_dropout, init_type, init_gain).to(device)
    adv_noise = init_adversarial_noise(mode="rand").to(device)
    global_adv_noise = Variable(adv_noise.to(device), requires_grad=True)
    pSp_net.pSp_optim = optimizer(params=list(protection_model.parameters()) + list([global_adv_noise]), **optim_args)
    best_loss = float("inf")

    for epoch in range(1, n_epochs):  # outer loop for different epochs;
        train_current_loss = {'full': 0., 'recon': 0., 'perturb': 0.}
        for i, data in enumerate(tqdm(train_loader, desc='')):  # inner loop within one epoch

            x_orig = data['A'].to(device).clone().detach()
            y_orig = data['B'].to(device).clone().detach()

            batch_size = x_orig.shape[0]
            y_target = create_target_img(batch_size=batch_size, img_transform=adv_img_transform)  # Ideal output to produce
            pSp_net.set_input(x_orig, y_target)

            # Compute the output for the original image
            with torch.no_grad():
                old_out = pSp_net(pSp_net.x)

            pSp_net.real_A = x_orig
            pSp_net.real_B = y_orig

            # Perform update and generate adversarial noise
            pSp_net.pSp_optim.zero_grad()
            adv_input = torch.cat((pSp_net.x, global_adv_noise.unsqueeze(0)), 1)
            adv_noise = protection_model(adv_input)
            pSp_net.closure_jpeg_randomized(model_adv_noise=adv_noise, global_adv_noise=global_adv_noise)
            pSp_net.loss_full.backward()
            pSp_net.pSp_optim.step()

            pSp_net.adv_A = torch.clamp(pSp_net.x + adv_noise, -1, 1)  # For Visualization

            # Apply compression on Adversarial Image
            temp = ((pSp_net.adv_A + 1) / 2)    # Normalize according to Jpeg
            temp[temp < 0] = 0
            temp[temp > 1] = 1
            y1, cb1, cr1 = pSp_net.compress(temp)    # Compress Image
            pSp_net.adv_A_comp = pSp_net.decompress(y1, cb1, cr1)     # Decompress Image
            pSp_net.adv_A_comp = pSp_net.jpeg_img_transform(pSp_net.adv_A_comp)

            #  Redo Forward pass on adversarial image through the model
            pSp_net.fake_B = pSp_net(pSp_net.adv_A_comp)

            visuals = pSp_net.get_current_visuals()  # get image results
            train_losses = pSp_net.get_current_losses()
            train_current_loss['full'] += train_losses['full']
            train_current_loss['recon'] += train_losses['recon']
            train_current_loss['perturb'] += train_losses['perturb']

        train_current_loss['full'] /= len(train_loader)
        train_current_loss['recon'] /= len(train_loader)
        train_current_loss['perturb'] /= len(train_loader)

        logger_train.display_current_results(pSp_net.get_current_visuals(), int(epoch))
        logger_train.plot_current_losses(int(epoch), train_current_loss)

        val_current_loss = {'full': 0., 'recon': 0., 'perturb': 0.}
        for i, data in enumerate(tqdm(val_loader, desc='')):  # inner loop within one epoch
            with torch.no_grad():
                x_orig = data['A'].to(device).clone().detach()
                y_orig = data['B'].to(device).clone().detach()

                batch_size = x_orig.shape[0]
                y_target = create_target_img(batch_size=batch_size, img_transform=adv_img_transform)  # Ideal output to produce

                pSp_net.set_input(x_orig, y_target)

                # Get the old output
                old_out = pSp_net(pSp_net.x)

                pSp_net.real_A = x_orig
                pSp_net.real_B = y_orig

                pSp_net.pSp_optim.zero_grad()
                adv_input = torch.cat((pSp_net.x, global_adv_noise.unsqueeze(0)), 1)
                adv_noise = protection_model(adv_input)
                pSp_net.closure_jpeg_randomized(model_adv_noise=adv_noise, global_adv_noise=global_adv_noise)

                pSp_net.adv_A = torch.clamp(pSp_net.x + adv_noise, -1, 1)  # For Visualization

                # Apply compression on Adversarial Image
                temp = ((pSp_net.adv_A + 1) / 2)  # Normalize according to Jpeg
                temp[temp < 0] = 0
                temp[temp > 1] = 1
                y1, cb1, cr1 = pSp_net.compress(temp)  # Compress Image
                pSp_net.adv_A_comp = pSp_net.decompress(y1, cb1, cr1)  # Decompress Image
                pSp_net.adv_A_comp = pSp_net.jpeg_img_transform(pSp_net.adv_A_comp)

                #  Redo Forward pass on adversarial image through the model
                pSp_net.fake_B = pSp_net(pSp_net.adv_A_comp)

                visuals = pSp_net.get_current_visuals()  # get image results
                val_losses = pSp_net.get_current_losses()
                val_current_loss['full'] += val_losses['full']
                val_current_loss['recon'] += val_losses['recon']
                val_current_loss['perturb'] += val_losses['perturb']

        val_current_loss['full'] /= len(val_loader)
        val_current_loss['recon'] /= len(val_loader)
        val_current_loss['perturb'] /= len(val_loader)

        logger_val.display_current_results(pSp_net.get_current_visuals(), int(epoch))
        logger_val.plot_current_losses(int(epoch), val_current_loss)

        # If the new loss is better than old loss, update the adversarial noise
        if val_current_loss['full'] < best_loss:
            save_filename_model = 'pSp_protection_%s_%sperturb.pth' % (net_noise, perturb_wt)
            save_path = os.path.join(model_ckpt_pth, save_filename_model)
            print('Updating the noise model')
            torch.save({"protection_net": protection_model.state_dict(), "global_noise": adv_noise.detach(), "perturb_wt": perturb_wt}, save_path)
            best_loss = val_current_loss['full']

        print('Epoch {} / {} \t Train Loss: {:.3f} \t Val Acc: {:.3f}'.format(epoch, n_epochs, train_current_loss['full'], val_current_loss['full']))

        save_filename_model = 'pSp_protection_%s_%sperturb_latest.pth' % (net_noise, perturb_wt)
        save_path = os.path.join(model_ckpt_pth, save_filename_model)
        torch.save({"protection_net": protection_model.state_dict(), "global_noise": adv_noise.detach(), "perturb_wt": perturb_wt}, save_path)

