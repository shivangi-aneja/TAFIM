# Standard libraries
import numpy as np
# PyTorch
import torch
import torch.nn as nn

# This is table 0 (the luminance table):
y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61],
     [12, 12, 14, 19, 26, 58, 60, 55],
     [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62],
     [18, 22, 37, 56, 68, 109, 103, 77],
     [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101],
     [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32).T

y_table = nn.Parameter(torch.from_numpy(y_table))

# This is table 1 (the chrominance table):
c_table = np.array([[17,  18,  24,  47,  99,  99,  99,  99],
             [18,  21,  26,  66,  99,  99,  99,  99],
             [24,  26,  56,  99,  99,  99,  99,  99],
             [47,  66,  99,  99,  99,  99,  99,  99],
             [99,  99,  99,  99,  99,  99,  99,  99],
             [99,  99,  99,  99,  99,  99,  99,  99],
             [99,  99,  99,  99,  99,  99,  99,  99],
             [99,  99,  99,  99,  99,  99,  99,  99]], dtype=np.float32)
c_table = nn.Parameter(torch.from_numpy(c_table))


def diff_round(x, mode='approx'):
    """ Differentiable rounding function
    Input:
        x(tensor)
    Output:
        x(tensor)
    """
    pi = torch.tensor(np.pi)
    if mode == 'approx':
        return torch.round(x) + (x - torch.round(x))**3
    elif mode == 'sin':
        return torch.subtract(x, torch.sin(2 * pi * x) / (2 * pi))
    elif mode == 'soft':
        x_ = torch.subtract(x, torch.sin(2 * pi * x) / (2 * pi))
        return torch.add((torch.round(x) - x_).detach(), x_)
    elif mode == 'identity':
        return x



def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        quality = 5000. / quality
    else:
        quality = 200. - quality*2
    return quality / 100.
