import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3,20,3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(20,40,3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(40,60,3,stride=2,padding=1)
        self.conv4 = nn.ConvTranspose2d(60,40,3,stride=2,padding=1,output_padding=1)
        self.conv5 = nn.ConvTranspose2d(40,20,3,stride=2,padding=1,output_padding=1)
        self.conv6 = nn.ConvTranspose2d(20,3,3,stride=2,padding=1,output_padding=1)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.relu(self.conv6(x))

        return x
