import torch

# [ pSp | simswap | styleclip | all]
architecture_type = 'pSp'
# [ffhq_encode | ffhq_stylemix | ffhq_fs | ffhq_all]
dataset_type = 'ffhq_encode'

n_epochs = 1000
max_steps = 10000

gpu_ids = [0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Basic Parameters
# of input image channels: 3 for RGB and 1 for grayscale
input_nc = 3
label_nc = 0

# of output image channels: 3 for RGB and 1 for grayscale
output_nc = 3

val_imgs = 1000
test_imgs = 1000

resize_size = 256
