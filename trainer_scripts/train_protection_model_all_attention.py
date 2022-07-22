import os
import pprint
import torch
import pickle
from tqdm import tqdm
from configs import data_config
from torch.optim import SGD, Adam
from argparse import ArgumentParser
from utils.tf_logger import Logger
from model_archs.pSp.pSp_model import pSp
from torch.utils.data import DataLoader
from utils.dataset import ImagesDatasetCombined
from utils.image_utils import create_target_img
from model_archs.simswap.fs_model import fsModel
from model_archs.styleClip.styleClip import StyleClip
from model_archs.attention.protection_model import ProtectionModelIndividual
from model_archs.protection_net.networks import define_G as GenPix2Pix
from configs.attack_config import no_dropout, init_type, init_gain, ngf, norm
from configs.attack_config import net_noise
from configs.common_config import device, architecture_type, dataset_type, n_epochs, val_imgs, resize_size
from configs.paths_config import ALL_ATTACK_BASE_DIR, ATTACK_BASE_DIR, TF_BASE_DIR
from configs.transforms_config import adv_img_transform, img_transform_simswap

CUDA_LAUNCH_BLOCKING = 1
torch.manual_seed(1)


if __name__ == '__main__':

    config_parser = ArgumentParser()
    config_parser.add_argument('--train_imgs', default=5000, type=int, help='Number of Training Images')
    config_parser.add_argument('--perturb_wt', default=10, type=int, help='Perturbation Weight')
    config_parser.add_argument('--batch_size', default=1, type=int, help='Perturbation Weight')
    config_parser.add_argument('--loss_type', default='l2', type=str, help='Loss Type')
    config_parser.add_argument('--pSp_protection', default='pSp/pSp_protection.pth', type=str, help='Path to pSp pretrained model')
    config_parser.add_argument('--simswap_protection', default='SimSwap/simswap_protection.pth', type=str, help='Path to SimSwap pretrained model')
    config_parser.add_argument('--styleclip_protection', default='StyleClip/styleclip_protection.pth', type=str, help='Path to StyleClip pretrained model')
    config_parser.add_argument('--lr', default=0.0001, type=float, help='Learning Rate')
    opts = config_parser.parse_args()

    train_imgs = opts.train_imgs
    perturb_wt = opts.perturb_wt
    batch_size = opts.batch_size
    loss_type = opts.loss_type
    lr = opts.lr
    pSp_protection_path = opts.pSp_protection
    simswap_protection = opts.simswap_protection
    styleclip_protection = opts.styleclip_protection

    opts_dict = vars(opts)
    pprint.pprint(opts_dict)

    dataset_args = data_config.DATASETS[dataset_type]
    transforms_pSp = dataset_args['transforms_pSp']().get_transforms()
    transforms_fs = dataset_args['transforms_fs']().get_transforms()

    train_dataset = ImagesDatasetCombined(source_root=dataset_args['train_source_root'], target_root=dataset_args['train_target_root'], pSp_source_transform=transforms_pSp['transform_source'], fs_source_transform=transforms_fs['transform_train'], pSp_target_transform=transforms_pSp['transform_source'], fs_target_transform=transforms_fs['transform_train'], num_imgs=train_imgs)
    val_dataset = ImagesDatasetCombined(source_root=dataset_args['val_source_root'], target_root=dataset_args['val_target_root'], pSp_source_transform=transforms_pSp['transform_inference'], fs_source_transform=transforms_fs['transform_test'], pSp_target_transform=transforms_pSp['transform_inference'], fs_target_transform=transforms_fs['transform_test'], num_imgs=val_imgs)
    train_loader_pSp = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
    val_loader_pSp = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    model_ckpt_pth = os.path.join(ALL_ATTACK_BASE_DIR, 'all_combined', net_noise)
    os.makedirs(model_ckpt_pth, exist_ok=True)

    pSp_target = create_target_img(batch_size=1, size=resize_size, img_transform=adv_img_transform, color=(255, 255, 255)).to(device)
    styleclip_target = create_target_img(batch_size=1, size=1024, img_transform=adv_img_transform, color=(0, 0, 255)).to(device)
    simswap_target = create_target_img(batch_size=1, size=resize_size, img_transform=img_transform_simswap, color=(255, 0, 0)).to(device)

    pSp_label = torch.full((1, resize_size, resize_size), 0).to(device)
    styleclip_label = torch.full((1, resize_size, resize_size), 1).to(device)
    simswap_label = torch.full((1, resize_size, resize_size), 2).to(device)

    optim_list = [(SGD, {'lr': lr}), (Adam, {'lr': lr})]
    my_optim = optim_list[1]
    optimizer = my_optim[0]
    optim_args = my_optim[1]
    logger_train = Logger(model_name=architecture_type, data_name=dataset_type, log_path=os.path.join(TF_BASE_DIR, 'attacks/all_attention/', net_noise, 'train', optimizer.__name__ + "_perturb" + str(perturb_wt) + "_batch" + str(batch_size) + "_lr_" + str(optim_args['lr']) + "_" + str(loss_type) + "loss"))
    logger_val = Logger(model_name=architecture_type, data_name=dataset_type, log_path=os.path.join(TF_BASE_DIR, 'attacks/all_attention/', net_noise, 'val', optimizer.__name__ + "_perturb" + str(perturb_wt) + "_batch" + str(batch_size) + "_lr_" + str(optim_args['lr']) + "_" + str(loss_type) + "loss"))

    protection_net = ProtectionModelIndividual(optimizer=optimizer, optim_args=optim_args).to(device)
    best_loss = float("inf")
    mse_criterion = torch.nn.MSELoss()

    pSp_net = pSp(perturb_wt=perturb_wt, attack_loss_type=loss_type).eval().to(device)
    styleclip_net = StyleClip(perturb_wt=perturb_wt, attack_loss_type=loss_type).eval().to(device)
    simswap_net = fsModel(perturb_wt=perturb_wt, attack_loss_type=loss_type).eval().to(device)

    pSp_noise_model = GenPix2Pix(6, 3, ngf, net_noise, norm, not no_dropout, init_type, init_gain).to(device)
    simswap_noise_model = GenPix2Pix(6, 3, ngf, net_noise, norm, not no_dropout, init_type, init_gain).to(device)
    styleclip_noise_model = GenPix2Pix(6, 3, ngf, net_noise, norm, not no_dropout, init_type, init_gain).to(device)

    # Pretrained Noise Models
    # pSp (Self-Reconstruction/Style-Mixing)
    save_path_pSp = torch.load(os.path.join(ATTACK_BASE_DIR, pSp_protection_path))
    pSp_noise_model.load_state_dict(save_path_pSp['protection_net'])
    pSp_noise_model.to(device).eval()
    pSp_global_adv_noise = save_path_pSp['global_noise'].to(device)

    # SimSwap (Face-Swapping)
    save_path_simswap = torch.load(os.path.join(ATTACK_BASE_DIR, simswap_protection))
    simswap_noise_model.load_state_dict(save_path_simswap['protection_net'])
    simswap_noise_model.to(device).eval()
    simswap_global_adv_noise = save_path_simswap['global_noise'].to(device)

    # StyleClip (Textual-Editing)
    save_path_styleclip = torch.load(os.path.join(ATTACK_BASE_DIR, styleclip_protection))
    styleclip_noise_model.load_state_dict(save_path_styleclip['protection_net'])
    styleclip_noise_model.to(device).eval()
    styleclip_global_adv_noise = save_path_styleclip['global_noise'].to(device)

    for epoch in range(n_epochs):  # outer loop for different epochs;
        train_current_loss = {'full': 0., 'recon': 0., 'perturb': 0.}
        for i, data in enumerate(tqdm(train_loader_pSp, desc='')):  # inner loop within one epoch

            pSp_net.real_A = data['A_pSp'].to(device)
            pSp_net.real_B = data['B_pSp'].to(device)
            simswap_net.real_A1 = data['A_fs'].to(device)
            simswap_net.real_A2 = data['B_fs'].to(device)
            x_orig = data['A_pSp'].to(device).clone().detach()
            styleclip_net.set_input(x_orig, styleclip_target)
            styleclip_net.real_A = x_orig.to(device)
            styleclip_net.real_B = styleclip_target.to(device)

            b_size = pSp_net.real_A.shape[0]

            # Set Input for pSp
            img_A_pSp = data['A_pSp']
            path_A_pSp = data['A_paths']
            pSp_net.set_input(img_A_pSp, pSp_target)
            simswap_net.y = simswap_target

            # Compute the output for the original image
            with torch.no_grad():
                old_out_pSp = pSp_net(pSp_net.x)
                old_out_styleclip = styleclip_net(x_orig)
                old_out_sg = simswap_net.swap_face(simswap_net.real_A1, simswap_net.real_A2)

            # Perform update and generate adversarial noise
            protection_net.model_optim.zero_grad()

            # Generate input for noise model
            adv_input_pSp = torch.cat((pSp_net.x, pSp_global_adv_noise), 1)
            adv_input_simswap = torch.cat((simswap_net.real_A1, simswap_global_adv_noise), 1)
            adv_input_styleclip = torch.cat((styleclip_net.x, styleclip_global_adv_noise), 1)

            # Feed it to noise model to generate intermediate output
            noise_pSp = pSp_noise_model(adv_input_pSp)
            noise_simswap = simswap_noise_model(adv_input_simswap)
            noise_styleclip = styleclip_noise_model(adv_input_styleclip)

            delta_label_pSp = torch.cat((noise_pSp, pSp_label.unsqueeze(0)), 1)
            delta_label_simswap = torch.cat((noise_simswap, simswap_label.unsqueeze(0)), 1)
            delta_label_styleclip = torch.cat((noise_styleclip, styleclip_label.unsqueeze(0)), 1)

            final_noise = protection_net(torch.stack([noise_pSp, noise_simswap, noise_styleclip]), [delta_label_pSp, delta_label_simswap, delta_label_styleclip])

            pSp_loss = pSp_net.closure_attention(method_specific_noise=noise_pSp, final_noise=final_noise)
            simswap_loss = simswap_net.closure_attention(method_specific_noise=noise_simswap, final_noise=final_noise)
            styleclip_loss = styleclip_net.closure_attention(method_specific_noise=noise_styleclip, final_noise=final_noise)

            total_loss = pSp_loss + styleclip_loss + simswap_loss
            total_loss.backward()
            protection_net.model_optim.step()

            pSp_net.adv_A = torch.clamp(pSp_net.x + final_noise, -1, 1)  # For Visualization
            pSp_net.adv_A_comp = pSp_net.adv_A
            simswap_net.adv_A1 = torch.clamp(simswap_net.real_A1 + final_noise, 0, 1)
            styleclip_net.adv_A = torch.clamp(styleclip_net.x + final_noise, -1, 1)

            #  Redo Forward pass on adversarial image through the model
            pSp_net.fake_B = pSp_net(pSp_net.adv_A_comp)
            simswap_net.fake_B = simswap_net.swap_face(simswap_net.adv_A1, simswap_net.real_A2)
            styleclip_net.fake_B = styleclip_net(styleclip_net.adv_A)

            visuals_pSp = pSp_net.get_current_visuals()  # get image results
            visuals_simswap = simswap_net.get_current_visuals()
            visuals_styleclip = styleclip_net.get_current_visuals()

            train_losses_pSp = pSp_net.get_current_losses()
            train_losses_simswap = simswap_net.get_current_losses()
            train_losses_styleclip = styleclip_net.get_current_losses()

            train_current_loss['full'] += train_losses_pSp['full'] + train_losses_styleclip['full'] + train_losses_simswap['full']
            train_current_loss['recon'] += train_losses_pSp['recon'] + train_losses_styleclip['recon'] + train_losses_simswap['recon']
            train_current_loss['perturb'] += train_losses_pSp['perturb'] + train_losses_styleclip['perturb'] + train_losses_simswap['perturb']

        train_current_loss['full'] /= len(train_loader_pSp)
        train_current_loss['recon'] /= len(train_loader_pSp)
        train_current_loss['perturb'] /= len(train_loader_pSp)

        logger_train.display_current_results(pSp_net.get_current_visuals(), int(epoch))
        logger_train.display_current_results(simswap_net.get_current_visuals(), int(epoch))
        logger_train.display_current_results(styleclip_net.get_current_visuals(), int(epoch))
        logger_train.plot_current_losses(int(epoch), train_current_loss)

        val_current_loss = {'full': 0., 'recon': 0., 'perturb': 0.}
        for i, data in enumerate(tqdm(val_loader_pSp, desc='')):  # inner loop within one epoch
            with torch.no_grad():
                pSp_net.real_A = data['A_pSp'].to(device)
                pSp_net.real_B = data['B_pSp'].to(device)
                simswap_net.real_A1 = data['A_fs'].to(device)
                simswap_net.real_A2 = data['B_fs'].to(device)
                x_orig = data['A_pSp'].to(device).clone().detach()
                styleclip_net.set_input(x_orig, styleclip_target)
                styleclip_net.real_A = x_orig.to(device)
                styleclip_net.real_B = styleclip_target.to(device)

                b_size = pSp_net.real_A.shape[0]

                # Set Input for pSp
                img_A_pSp = data['A_pSp']
                path_A_pSp = data['A_paths']
                pSp_net.set_input(img_A_pSp, pSp_target)
                simswap_net.y = simswap_target

                # Get the old output
                old_out_pSp = pSp_net(pSp_net.x)
                old_out_styleclip = styleclip_net(x_orig)
                old_out_sg = simswap_net.swap_face(simswap_net.real_A1, simswap_net.real_A2)

                # Generate input for noise model
                adv_input_pSp = torch.cat((pSp_net.x, pSp_global_adv_noise), 1)
                adv_input_simswap = torch.cat((simswap_net.real_A1, simswap_global_adv_noise), 1)
                adv_input_styleclip = torch.cat((styleclip_net.x, styleclip_global_adv_noise), 1)

                # Feed it to noise model to generate intermediate output
                noise_pSp = pSp_noise_model(adv_input_pSp)
                noise_simswap = simswap_noise_model(adv_input_simswap)
                noise_styleclip = styleclip_noise_model(adv_input_styleclip)

                delta_label_pSp = torch.cat((noise_pSp, pSp_label.unsqueeze(0)), 1)
                delta_label_simswap = torch.cat((noise_simswap, simswap_label.unsqueeze(0)), 1)
                delta_label_styleclip = torch.cat((noise_styleclip, styleclip_label.unsqueeze(0)), 1)

                final_noise = protection_net(torch.stack([noise_pSp, noise_simswap, noise_styleclip]), [delta_label_pSp, delta_label_simswap, delta_label_styleclip])

                pSp_net.adv_A = torch.clamp(pSp_net.x + final_noise, -1, 1)  # For Visualization
                pSp_net.adv_A_comp = pSp_net.adv_A
                simswap_net.adv_A1 = torch.clamp(simswap_net.real_A1 + final_noise, 0, 1)
                styleclip_net.adv_A = torch.clamp(styleclip_net.x + final_noise, -1, 1)

                #  Redo Forward pass on adversarial image through the model
                pSp_net.fake_B = pSp_net(pSp_net.adv_A_comp)
                simswap_net.fake_B = simswap_net.swap_face(simswap_net.adv_A1, simswap_net.real_A2)
                styleclip_net.fake_B = styleclip_net(styleclip_net.adv_A)

                visuals_pSp = pSp_net.get_current_visuals()  # get image results
                visuals_simswap = simswap_net.get_current_visuals()
                visuals_styleclip = styleclip_net.get_current_visuals()

                val_losses_pSp = pSp_net.get_current_losses()
                val_losses_simswap = simswap_net.get_current_losses()
                val_losses_styleclip = styleclip_net.get_current_losses()

                val_current_loss['full'] += val_losses_pSp['full'] + val_losses_styleclip['full'] + val_losses_simswap['full']
                val_current_loss['recon'] += val_losses_pSp['recon'] + val_losses_styleclip['recon'] + val_losses_simswap['recon']
                val_current_loss['perturb'] += val_losses_pSp['perturb'] + val_losses_styleclip['perturb'] + val_losses_simswap['perturb']

        val_current_loss['full'] /= len(val_loader_pSp)
        val_current_loss['recon'] /= len(val_loader_pSp)
        val_current_loss['perturb'] /= len(val_loader_pSp)

        logger_val.display_current_results(pSp_net.get_current_visuals(), int(epoch))
        logger_val.display_current_results(simswap_net.get_current_visuals(), int(epoch))
        logger_val.display_current_results(styleclip_net.get_current_visuals(), int(epoch))
        logger_val.plot_current_losses(int(epoch), val_current_loss)

        # If the new loss is better than old loss, update the adversarial noise
        if val_current_loss['full'] < best_loss:
            save_filename_model = 'model_%s_net_%sperturb_%simgs_%s_%sbatch_%s_loss_%s_lr.pth' % (optimizer.__name__, perturb_wt, train_imgs, net_noise, batch_size, loss_type, optim_args['lr'])
            save_path_model = os.path.join(ALL_ATTACK_BASE_DIR + '/all_attention/' + net_noise + '_noise_un/', save_filename_model)
            print('Updating the noise model')
            torch.save({'attn_net': protection_net.attn_model.state_dict(), 'refinement_net': protection_net.fusion_model.state_dict()}, save_path_model)
            best_loss = val_current_loss['full']

        save_filename_model = 'model_%s_net_%sperturb_%simgs_%s_%sbatch_%s_loss_%s_lr_latest.pth' % (optimizer.__name__, perturb_wt, train_imgs, net_noise, batch_size, loss_type, optim_args['lr'])
        save_path_model = os.path.join(ALL_ATTACK_BASE_DIR + '/all_attention/' + net_noise + '_noise_un/', save_filename_model)
        torch.save({'attn_net': protection_net.attn_model.state_dict(), 'refinement_net': protection_net.fusion_model.state_dict()}, save_path_model)
        best_loss = val_current_loss['full']
