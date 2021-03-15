import torch
from torchvision import transforms

"""
Mean and Std deviation
"""
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)

def log(x):
    return torch.log(x + 1e-8)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


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
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
