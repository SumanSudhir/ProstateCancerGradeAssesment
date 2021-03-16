import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F


"""
Mean and Std deviation
"""
mean = (0.5, 0.5, 0.5)
std = (0.5, 0.5, 0.5)

def log(x):
    return torch.log(x + 1e-8)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()


def get_transform(*, data, size=128):
    if data == 'train':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size, size)),
            transforms.RandomChoice([
                transforms.ColorJitter(brightness=0.5),
                transforms.ColorJitter(contrast=0.5),
                transforms.ColorJitter(saturation=0.5),
                transforms.ColorJitter(hue=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
            ]),
            transforms.RandomChoice([
                transforms.RandomRotation((0,0)),
                transforms.RandomHorizontalFlip(p=1),
                transforms.RandomVerticalFlip(p=1),
                transforms.RandomRotation((90,90)),
                transforms.RandomRotation((180,180)),
                transforms.RandomRotation((270,270)),
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.RandomRotation((90,90)),
                ]),
                transforms.Compose([
                    transforms.RandomHorizontalFlip(p=1),
                    transforms.RandomRotation((270,270)),
                ])
            ]),
            transforms.ToTensor(),
            transforms.Normalize(mean,std)])

    elif data == 'valid':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((size,size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])

def VAE_loss(recons_x, x, mu, logvar):
    MSE = F.mse_loss(recons_x, x, reduction='mean')
    # l2 = 5e-5*torch.mean(1+logvar-mu.pow(2) -logvar.exp())
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)

    return MSE+KLD

class UnNormalize(object):
    def __init__(self, mean=mean, std=std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
