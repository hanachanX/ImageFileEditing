from contextlib import contextmanager
from math import sqrt, log

import torch
import torch.nn as nn

class BaseModule(nn.Module):
    def __init__(self):
        self.act_fn = None
        super(BaseModule, self).__init__()

    def selu_init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                m.weight.data.normal_(0.0, 1.0 / sqrt(m.weight.numel()))
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d) and m.weight.requires_grad:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

            elif isinstance(m, nn.Linear) and m.weight.requires_grad:
                m.weight.data.normal_(0, 1.0 / sqrt(m.weight.numel()))
                m.bias.data.zero_()

    def initialize_weights_xavier_uniform(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) and m.weight.requires_grad:
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) and m.weight.requires_grad:
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def load_state_dict(self, state_dict, strict=True, self_state=False):
        own_state = self_state if self_state else self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                try:
                    own_state[name].copy_(param.data)
                except Exception as e:
                    print("Parameter {} fails to load.".format(name))
                    print("-----------------------------------------")
                    print(e)
            else:
                print("Parameter {} is not in the model. ".format(name))

    @contextmanager
    def set_activation_inplace(self):
        if hasattr(self, 'act_fn') and hasattr(self.act_fn, 'inplace'):
            # save memory
            self.act_fn.inplace = True
            yield
            self.act_fn.inplace = False
        else:
            yield

    def total_parameters(self):
        total = sum([i.numel() for i in self.parameters()])
        trainable = sum([i.numel() for i in self.parameters() if i.requires_grad])
        print("Total parameters : {}. Trainable parameters : {}".format(total, trainable))
        return total

    def forward(self, *x):
        raise NotImplementedError


class ResidualFixBlock(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1,
                 groups=1, activation=nn.SELU(), conv=nn.Conv2d):
        super(ResidualFixBlock, self).__init__()
        self.act_fn = activation
        self.m = nn.Sequential(
            conv(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, groups=groups),
            activation,
            # conv(out_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, dilation=1, groups=groups),
            conv(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, groups=groups),
        )

    def forward(self, x):
        out = self.m(x)
        return self.act_fn(out + x)


class ConvBlock(BaseModule):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, groups=1,
                 activation=nn.SELU(), conv=nn.Conv2d):
        super(ConvBlock, self).__init__()
        self.m = nn.Sequential(conv(in_channels, out_channels, kernel_size, padding=padding,
                                    dilation=dilation, groups=groups),
                               activation)

    def forward(self, x):
        return self.m(x)


class UpSampleBlock(BaseModule):
    def __init__(self, channels, scale, activation, atrous_rate=1, conv=nn.Conv2d):
        assert scale in [2, 4, 8], "Currently UpSampleBlock supports 2, 4, 8 scaling"
        super(UpSampleBlock, self).__init__()
        m = nn.Sequential(
            conv(channels, 4 * channels, kernel_size=3, padding=atrous_rate, dilation=atrous_rate),
            activation,
            nn.PixelShuffle(2)
        )
        self.m = nn.Sequential(*[m for _ in range(int(log(scale, 2)))])

    def forward(self, x):
        return self.m(x)


class SpatialChannelSqueezeExcitation(BaseModule):
    def __init__(self, in_channel, reduction=16, activation=nn.ReLU()):
        super(SpatialChannelSqueezeExcitation, self).__init__()
        linear_nodes = max(in_channel // reduction, 4)  # avoid only 1 node case
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_excite = nn.Sequential(
            # check the paper for the number 16 in reduction. It is selected by experiment.
            nn.Linear(in_channel, linear_nodes),
            activation,
            nn.Linear(linear_nodes, in_channel),
            nn.Sigmoid()
        )
        self.spatial_excite = nn.Sequential(
            nn.Conv2d(in_channel, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        #
        channel = self.avg_pool(x).view(b, c)
        # channel = F.avg_pool2d(x, kernel_size=(h,w)).view(b,c) # used for porting to other frameworks
        cSE = self.channel_excite(channel).view(b, c, 1, 1)
        x_cSE = torch.mul(x, cSE)

        # spatial
        sSE = self.spatial_excite(x)
        x_sSE = torch.mul(x, sSE)
        # return x_sSE
        return torch.add(x_cSE, x_sSE)


