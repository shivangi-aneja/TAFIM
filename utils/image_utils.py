import cv2
import torch
import numpy as np
from configs.common_config import resize_size


def create_target_img(batch_size=1, size=resize_size, img_transform=None, color=(255, 255, 255)):
    image = np.zeros((size, size, 3), np.uint8)
    image[:] = color
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = img_transform(image)
    images = torch.repeat_interleave(image.unsqueeze(0), repeats=batch_size, dim=0)
    return images


def tensor2im(var, fs=False):
    var = var.permute(1, 2, 0).cpu().detach().numpy()
    if not fs:
        var = ((var + 1) / 2)
    var[var < 0] = 0
    var[var > 1] = 1
    var = var * 255
    return var
