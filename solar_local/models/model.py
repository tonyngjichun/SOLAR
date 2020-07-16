import torch.nn as nn
from .soa_block import SOABlock 

eps_fea_norm = 1e-5
eps_l2_norm = 1e-10

def weights_init(l):
    if isinstance(l, nn.Conv2d):
        # nn.init.xavier_normal_(l.weight.data)
        nn.init.orthogonal_(l.weight.data)
    return
    
class SOLAR_LOCAL(nn.Module):
    """model definition
    """
    def __init__(self, dim_desc=128, drop_rate=0.1, soa=False, soa_layers=''):
        super().__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate
        self.soa = soa
        self.soa_layers = soa_layers

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU()

        self.layer1 = nn.Sequential(
            nn.InstanceNorm2d(1, affine=False, eps=eps_fea_norm),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,
            )
            
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,
            )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,
            )

        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,
            )

        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,
            )

        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,
            )

        self.dropout = nn.Dropout(self.drop_rate)

        self.layer7 = nn.Sequential(
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            )

        self.desc_norm = nn.Sequential(
            nn.LocalResponseNorm(2 * self.dim_desc, alpha=2 * self.dim_desc, beta=0.5, k=0)
        )

        print('SOLAR_LOCAL - SOSNet w/ SOA layers:')
        if self.soa:
            if '3' in self.soa_layers:
                print("SOA_3:")
                self.soa3 = SOABlock(64, 4)
            if '4' in self.soa_layers:
                print("SOA_4:")
                self.soa4 = SOABlock(64, 4)
            if '5' in self.soa_layers:
                print("SOA_5:")
                self.soa5 = SOABlock(128, 2)
            if '6' in self.soa_layers:
                print("SOA_6:")
                self.soa6 = SOABlock(128, 2)

        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5, self.layer6, self.layer7]:
            layer.apply(weights_init)

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        if self.soa and '3' in self.soa_layers:
             x = self.soa3(x)
        x = self.layer4(x)
        if self.soa and '4' in self.soa_layers:
             x = self.soa4(x)

        x = self.layer5(x)
        if self.soa and '5' in self.soa_layers:
            x = self.soa5(x)
        x = self.layer6(x)
        if self.soa and '6' in self.soa_layers:
            x = self.soa6(x)

        descr = self.desc_norm(self.dropout(self.layer7(x)) + eps_l2_norm)
        descr = descr.view(descr.size(0), -1)

        return descr

class SOSNet32x32(nn.Module):
    """
    128-dimensional SOSNet model definition trained on 32x32 patches
    """
    def __init__(self, dim_desc=128, drop_rate=0.1):
        super(SOSNet32x32, self).__init__()
        self.dim_desc = dim_desc
        self.drop_rate = drop_rate

        norm_layer = nn.BatchNorm2d
        activation = nn.ReLU()

        self.layers = nn.Sequential(
            nn.InstanceNorm2d(1, affine=False, eps=eps_fea_norm),
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            norm_layer(32, affine=False, eps=eps_fea_norm),
            activation,

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            norm_layer(64, affine=False, eps=eps_fea_norm),
            activation,

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
            activation,

            nn.Dropout(self.drop_rate),
            nn.Conv2d(128, self.dim_desc, kernel_size=8, bias=False),
            norm_layer(128, affine=False, eps=eps_fea_norm),
        )

        self.desc_norm = nn.Sequential(
            nn.LocalResponseNorm(2 * self.dim_desc, alpha=2 * self.dim_desc, beta=0.5, k=0)
        )

        return

    def forward(self, patch):
        descr = self.desc_norm(self.layers(patch) + eps_l2_norm)
        descr = descr.view(descr.size(0), -1)
        return descr