import torch
import torch.nn as nn

from efficientnet_pytorch import EfficientNet
import torchvision.models as models
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from sklearn.utils import shuffle

class EfficientModel(nn.Module):

    def __init__(self, c_out=5, tile_size=256, n_tiles=64, name='efficientnet-b0'):
        super().__init__()

        m = EfficientNet.from_pretrained(name, advprop=True, num_classes=c_out, in_channels=3)
        c_feature = m._fc.in_features
        m._fc = nn.Identity()
        self.feature_extractor = m
        self.n_tiles = n_tiles
        self.tile_size = tile_size
        self.head = AttentionMIL(c_feature, c_out, n_tiles=self.n_tiles)

    def forward(self, x):
        x = x.view(-1, 3, self.tile_size, self.tile_size)
        h = self.feature_extractor(x)
        h, w, instance_loss = self.head(h)

        return h, w, instance_loss

class AttentionPooling(nn.Module):
    def __init__(self, L, M):
        super().__init__()

        self.V = nn.Linear(L, M)
        self.w = nn.Linear(M, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def attention_weight(self, input):
        x = self.V(input)
        x = self.tanh(x)
        x = self.w(x)
        x = self.softmax(x)

        return x

    def forward(self, input):
        w = self.attention_weight(input)         # b,n,1
        aggegated_features = torch.matmul(input.transpose(1,2), w).squeeze(2)  # b,c,n x b,n,1 ---> b,c,1

        return aggegated_features, w

class AttentionMIL(nn.Module):
    def __init__(self, L, n_classes, n_tiles, k_instance=9):
        super().__init__()
        self.n_tiles = n_tiles
        self.k_instance = k_instance
        # self.instance_loss_fn = nn.CrossEntropyLoss()
        self.instance_loss_fn = nn.BCEWithLogitsLoss()
        self.attention_mil_pool = AttentionPooling(L,L//2)
        self.classifier = nn.Linear(L,n_classes)
        self.instance_classifier = nn.Linear(L,1)

    def forward(self, input):
        bn,c = input.shape
        input = input.view(-1,self.n_tiles,c)
        agg_f, w = self.attention_mil_pool(input)
        instance_loss, all_preds, all_targets = self.inst_eval(w,input)

        out = self.classifier(agg_f)

        return out, w, instance_loss

    @staticmethod
    def create_positive_targets(length, device):
        return torch.full((length, ), 1, device=device).long()
    @staticmethod
    def create_negative_targets(length, device):
        return torch.full((length, ), 0, device=device).long()

    def inst_eval(self, w, input):
        device = input.device
        w = w.squeeze()
        # input = input.squeeze(0)
        # bn,c = input.shape
        # input = input.view(-1,self.n_tiles,c)

        top_p_ids = torch.topk(w, self.k_instance)[1]
        top_n_ids = torch.topk(-w, self.k_instance)[1]

        top_p, top_n, p_targets, n_targets = [],[],[],[]
        for i in range(w.shape[0]):
            top_p.append(torch.index_select(input[i], dim=0, index=top_p_ids[i]))
            top_n.append(torch.index_select(input[i], dim=0, index=top_n_ids[i]))
            p_targets.append(self.create_positive_targets(self.k_instance, device))
            n_targets.append(self.create_negative_targets(self.k_instance, device))


        all_targets = torch.cat([torch.cat(p_targets), torch.cat(n_targets)])
        all_instances = torch.cat([torch.cat(top_p), torch.cat(top_n)])

        all_instances,all_targets = shuffle(all_instances,all_targets)

        logits = self.instance_classifier(all_instances).squeeze()
        # all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
        all_preds = logits.sigmoid().round()
        instance_loss = self.instance_loss_fn(logits.float(), all_targets.float())

        return instance_loss, all_preds, all_targets

    # def inst_eval(self, w, input):
    #     device = input.device
    #     w = w.squeeze()
    #     input = input.squeeze(0)
    #
    #     top_p_ids = torch.topk(w, self.k_instance)[1]
    #     top_p = torch.index_select(input, dim=0, index=top_p_ids)
    #     top_n_ids = torch.topk(-w, self.k_instance)[1]
    #     top_n = torch.index_select(input, dim=0, index=top_n_ids)
    #     p_targets = self.create_positive_targets(self.k_instance, device)
    #     n_targets = self.create_negative_targets(self.k_instance, device)
    #
    #     all_targets = torch.cat([p_targets, n_targets], dim=0)
    #     all_instances = torch.cat([top_p, top_n], dim=0)
    #
    #     all_instances,all_targets = shuffle(all_instances,all_targets)
    #
    #     logits = self.instance_classifier(all_instances).squeeze()
    #     # all_preds = torch.topk(logits, 1, dim = 1)[1].squeeze(1)
    #     all_preds = logits.sigmoid().round()
    #     instance_loss = self.instance_loss_fn(logits.float(), all_targets.float())
    #
    #     return instance_loss, all_preds, all_targets



class EfficientAvgModel(nn.Module):
    def __init__(self, out_dim=5):
        super().__init__()
        self.enet = EfficientNet.from_pretrained('efficientnet-b0', advprop=True, num_classes=out_dim, in_channels=3)
        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x

## Model for adapting the input image
class AdaptNet(nn.Module):
    def __init__(self, tile_size=256):
        super().__init__()
        self.tile_size = tile_size
        self.conv1 = nn.Conv2d(3,64,3,stride=2,padding=1)
        self.conv2 = nn.Conv2d(64,128,3,stride=2,padding=1)
        self.conv3 = nn.Conv2d(128,256,3,stride=2,padding=1)
        self.conv4 = nn.ConvTranspose2d(256,128,3,stride=2,padding=1,output_padding=1)
        self.conv5 = nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1)
        self.conv6 = nn.ConvTranspose2d(64,3,3,stride=2,padding=1,output_padding=1)

    def forward(self,x):
        x = x.view(-1, 3, self.tile_size, self.tile_size)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.conv6(x)

        return x


class AutoEncoder(nn.Module):
    def __init__(self,tile_size=256):
        super().__init__()
        self.tile_size = tile_size

        self.encoder = nn.Sequential(
                    nn.Conv2d(3,64,3,stride=2,padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64,128,3,stride=2,padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128,256,3,stride=2,padding=1),
                    nn.ReLU(),
                )

        self.decoder = nn.Sequential(
                    nn.ConvTranspose2d(256,128,3,stride=2,padding=1,output_padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64,3,3,stride=2,padding=1,output_padding=1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


class RoiNet(nn.Module):
    def __init__(self, c_out=5, tile_size=256, n_tiles=64, name='efficientnet-b0'):
        super().__init__()
        self.tile_size = tile_size
        self.n_tiles = n_tiles
        c_feature = 1024

        self.feature_extractor = nn.Sequential(
                    nn.Conv2d(3,64,3,stride=2,padding=1),
                    nn.ReLU(),
                    nn.Conv2d(64,128,3,stride=2,padding=1),
                    nn.ReLU(),
                    nn.Conv2d(128,256,3,stride=2,padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256,512,3,stride=2,padding=1),
                    nn.ReLU(),
                    nn.Conv2d(512,1024,3,stride=2,padding=1),
                    nn.ReLU(),
        )


        # m = EfficientNet.from_pretrained(name, advprop=True, num_classes=c_out, in_channels=3)
        # c_feature = m._fc.in_features
        # m._fc = nn.Identity()
        # self.feature_extractor = m

        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.head = AttentionMIL(c_feature, c_out, n_tiles=self.n_tiles)
        self.adapt = AdaptNet(tile_size=self.tile_size)

    def forward(self, x):
        x = x.view(-1, 3, self.tile_size, self.tile_size)
        x = self.adapt(x)
        h = self.pooling(self.feature_extractor(x)).squeeze()
        h, w, instance_loss = self.head(h)

        return h, w, x, instance_loss

class AutoRoi(nn.Module):
    def __init__(self, c_out=5, n_tiles=64):
        super().__init__()
        self.n_tiles = n_tiles
        c_feature = 512

        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.head = AttentionMIL(c_feature, c_out, n_tiles=self.n_tiles)

    def forward(self, x):
        h = self.pooling(x).squeeze()
        h, w, instance_loss = self.head(h)

        return h, w, x, instance_loss

# https://github.com/WilliBee/bigan_SRL/blob/master/models.py
# https://arxiv.org/pdf/1606.03657.pdf
class Generator(nn.Module):
    """
    Input: Vector from representation space of dimension z-dim
    Output: Vector from image space of dimension X-dim
    """
    def __init__(self, z_dim, params):
        super().__init__()

        self.input_dim = z_dim
        self.output_dim = 1
        self.slope = params['slope']
        self.dropout  = params['dropout']
        # self.num_channels = params['num_channels']

        self.generator = nn.Sequential(
            #z_dimx1x1
            nn.ConvTranspose2d(z_dim, 512, 3, stride=2, padding=0,output_padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.slope,inplace=True),
            #512x4x4
            nn.ConvTranspose2d(512, 512, 3, stride=2, padding=1,output_padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.slope,inplace=True),
            #512x8x8
            nn.ConvTranspose2d(512, 256, 3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.slope,inplace=True),
            #256x16x16
            nn.ConvTranspose2d(256, 128 ,3, stride=2, padding=1, output_padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.slope,inplace=True),
            ##128x32x32
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1,bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.slope,inplace=True),
            #64x64x64
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1,bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.slope,inplace=True),
            #32x128x128
            nn.Conv2d(32, 3, 1, stride=1, bias=True),
            nn.Tanh()
            #3x128x128
        )

        # utils.initialize_weights(self)

    def forward(self, x):
        x = self.generator(x)

        return x

class Discriminator(nn.Module):
    """
    Input: Tuple (X,z) of image vector and corresponding z vector from the Encoder.
    Output: 1-dimesnional value
    """
    def __init__(self, z_dim, h_dim, params):
        super().__init__()
        self.z_dim = z_dim
        self.h_dim = h_dim

        self.slope = params['slope']
        self.dropout = params['dropout']
        self.batch_size = params['batch_size']

        # inference over x
        self.inference_x = nn.Sequential(
            #3x128x128
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.slope, inplace=True),
            # #32x64x64
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.slope, inplace=True),
            # #64x32x32
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.slope, inplace=True),
            # #128x16x16
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.slope, inplace=True),
            # #256x8x8
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.slope, inplace=True),
            #512x4x4
            nn.Conv2d(512, 512, 3, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.slope, inplace=True),
        )


        # Inference over z
        self.inference_joint = nn.Sequential(
            nn.Linear(512 + self.z_dim, self.h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.h_dim, self.h_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.h_dim, 1),
            nn.Sigmoid()
        )

        # utils.initialize_weights(self)

    def forward(self, x, z):
        x = self.inference_x(x)
        x = x.view(x.shape[0], -1)
        z = z.view(z.shape[0], -1)
        xz = torch.cat((x,z), dim=1)
        out = self.inference_joint(xz)

        return out

class Encoder(nn.Module):
    """
    Input:
    Output:
    """
    def __init__(self, z_dim, params):
        super().__init__()

        self.z_dim = z_dim
        self.slope = params['slope']
        self.dropout = params['dropout']
        self.batch_size = params['batch_size']


        self.encoder = nn.Sequential(
            #3x128x128
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.slope, inplace=True),
            # #32x64x64
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.slope, inplace=True),
            # #64x32x32
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.slope, inplace=True),
            # #128x16x16
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.slope, inplace=True),
            # #256x8x8
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.slope, inplace=True),
            #512x4x4
            nn.Conv2d(512, 512, 3, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.slope, inplace=True),
            #512x1x1
            nn.Conv2d(512, self.z_dim, 1, stride=1, padding=0, bias=True),
            #z_dimx1x1
        )
        # utils.initialize_weights(self)

    def forward(self, x):
        x = self.encoder(x)

        return x



# z_dim = 128
# params = {'slope':0.1, 'dropout':0.2, 'batch_size':4}
# h_dim = 1024
#
# G = Generator(z_dim, params)
# D = Discriminator(z_dim, h_dim, params)
# E = Encoder(z_dim, params)
#
# X = torch.randn(4,3,128,128)
# z = torch.randn(4,z_dim,1,1)
#
# z_hat = E(X)
# X_hat = G(z)
#
# D_enc = D(X, z_hat)
# D_gen = D(X_hat,z)
#
# print(D_gen.shape)
# print(D_enc.shape)
