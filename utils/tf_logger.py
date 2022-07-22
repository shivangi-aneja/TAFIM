"""
    Tensorflow logger for monitoring train/val curves
"""

from torch.utils.tensorboard import SummaryWriter
import torch
import torchvision.utils as vutils


class Logger:

    def __init__(self, model_name, data_name, log_path):
        """
            Initializes the logger object for computing loss/accuracy curves
            Args:
                model_name (str): The name of the model for which training needs to be monitored
                data_name (str): Dataset name
                log_path (str): Base path for logging
        """
        self.model_name = model_name
        self.data_name = data_name
        self.comment = '{}_{}'.format(model_name, data_name)
        # TensorBoard
        self.writer = SummaryWriter(log_dir=log_path, comment=self.comment)

    def log_scalar(self, scalar_value, iter_num, scalar_name='error'):
        """
            Logs the scalar value passed for train and val epoch
            Args:
                scalar_value (float): loss/accuracy value to be logged
                iter_num (int): iteration number
                scalar_name (str): name of scalar to be logged
            Returns:
                None
        """

        if isinstance(scalar_value, torch.autograd.Variable):
            scalar_value = scalar_value.data.cpu().numpy()
        self.writer.add_scalar(self.comment + '_' + scalar_name, scalar_value, iter_num)

    def plot_current_losses(self, iter_num, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            iter_num (int)           -- iteration number
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """

        scalars = list(losses.keys())
        for s in scalars:
            self.log_scalar(scalar_value=losses[s], iter_num=iter_num, scalar_name=s)

    def display_current_results(self, visuals, total_iters):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            total_iters (int) - - Iteration number
        """
        img_keys = list(visuals.keys())
        img_name = '-'.join(map(str, img_keys))

        images_list = []
        for key in img_keys:
            images_list.append(visuals[key][0])

        images = torch.stack(images_list).squeeze()
        img_name = '{}_{}'.format(self.comment, img_name)

        # Make horizontal grid from image tensor
        horizontal_grid = vutils.make_grid(images, normalize=True, scale_each=True)
        self.writer.add_image(img_name, horizontal_grid, total_iters)