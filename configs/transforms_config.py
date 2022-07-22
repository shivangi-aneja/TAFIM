from abc import abstractmethod
import torchvision.transforms as transforms
from configs.common_config import resize_size


adv_img_transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

img_transform_simswap = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])


class TransformsConfig(object):

    def __init__(self):
        pass

    @abstractmethod
    def get_transforms(self):
        pass


class EncodeTransforms(TransformsConfig):

    def __init__(self):
        super(EncodeTransforms, self).__init__()

    def get_transforms(self):
        transforms_dict = {
            'transform_gt_train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((resize_size, resize_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_source': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]),
            'transform_inference': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((resize_size, resize_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        }
        return transforms_dict


class FaceSwapTransforms(TransformsConfig):

    def __init__(self):
        super(FaceSwapTransforms, self).__init__()

    def get_transforms(self):
        transforms_dict = {
            'transform_train': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((resize_size, resize_size)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor()]),
            'transform_test': transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((resize_size, resize_size)),
                transforms.ToTensor()]),
        }
        return transforms_dict