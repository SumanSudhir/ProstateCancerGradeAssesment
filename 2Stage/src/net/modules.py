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
    def __init__(self, L, n_classes, n_tiles, k_instance=16):
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
        # input = input.view(-1,bn,c)
        # if(self.n_tiles==None):
        #     input = input.view(-1,bn,c)
        # else:
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
    def __init__(self, out_dim=4):
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
