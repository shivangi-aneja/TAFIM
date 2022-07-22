import cv2
import torch
from .data_utils import make_dataset
from torch.utils.data import Dataset


class ImagesDataset(Dataset):

    def __init__(self, source_root, target_root, target_transform=None, source_transform=None, mode='train', num_imgs=1000):
        self.source_paths = sorted(make_dataset(source_root))[:num_imgs]
        self.target_paths = sorted(make_dataset(target_root))[:num_imgs]
        self.source_transform = source_transform
        self.target_transform = target_transform
        self.mode = mode

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = cv2.imread(from_path)
        from_im = cv2.cvtColor(from_im, cv2.COLOR_BGR2RGB)
        from_im = self.source_transform(from_im)
        if self.mode == 'test':
            label = torch.tensor([1, 0, 0, 0, 0])
        else:
            label = torch.randint(0, 2, (5,))

        to_path = self.target_paths[index]
        to_im = cv2.cvtColor(cv2.imread(to_path), cv2.COLOR_BGR2RGB)
        to_im = self.target_transform(to_im)
        return {'A': from_im, 'B': to_im, 'A_paths': from_path, 'B_paths': to_path, 'label': label}


class ImagesDatasetCombined(Dataset):

    def __init__(self, source_root, target_root, pSp_source_transform=None, pSp_target_transform=None, fs_source_transform=None, fs_target_transform=None, num_imgs=1000):
        self.source_paths = sorted(make_dataset(source_root))[:num_imgs]
        self.target_paths = sorted(make_dataset(target_root))[:num_imgs]
        self.pSp_source_transform = pSp_source_transform
        self.pSp_target_transform = pSp_target_transform
        self.fs_source_transform = fs_source_transform
        self.fs_target_transform = fs_target_transform

    def __len__(self):
        return len(self.source_paths)

    def __getitem__(self, index):
        from_path = self.source_paths[index]
        from_im = cv2.imread(from_path)
        from_im = cv2.cvtColor(from_im, cv2.COLOR_BGR2RGB)
        from_im_pSp = self.pSp_source_transform(from_im)
        from_im_fs = self.fs_source_transform(from_im)
        to_path = self.target_paths[index]
        to_im = cv2.cvtColor(cv2.imread(to_path), cv2.COLOR_BGR2RGB)
        to_im_pSp = self.pSp_target_transform(to_im)
        to_im_fs = self.fs_target_transform(to_im)
        return {'A_pSp': from_im_pSp, 'A_fs': from_im_fs, 'B_pSp': to_im_pSp, 'B_fs': to_im_fs, 'A_paths': from_path, 'B_paths': to_path}
