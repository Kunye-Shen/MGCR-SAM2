import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        return self.conv(x)

class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net

class MIC_2(nn.Module): # Multi-scale Information Compensation Module
    def __init__(self, in_channel, out_channel):
        super(MIC_2, self).__init__()
        self.conv1 = convbnrelu(in_channel, out_channel)
        self.conv2 = nn.Sequential(
            nn.AvgPool2d(2,2),
            convbnrelu(in_channel, out_channel)
        )

        self.conv_out = nn.Sequential(
            convbnrelu(in_channel+out_channel*2, out_channel),
            convbnrelu(out_channel, out_channel)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out2 = interpolate(out2, x.size()[2:])

        out = self.conv_out(torch.cat((x,out1,out2),1))

        return out
    
class MIC_3(nn.Module): # Multi-scale Information Compensation Module
    def __init__(self, in_channel, out_channel):
        super(MIC_3, self).__init__()
        self.conv1 = convbnrelu(in_channel, out_channel)
        self.conv2 = nn.Sequential(
            nn.AvgPool2d(2,2),
            convbnrelu(in_channel, out_channel)
        )
        self.conv3 = nn.Sequential(
            nn.AvgPool2d(4,4),
            convbnrelu(in_channel, out_channel)
        )

        self.conv_out = nn.Sequential(
            convbnrelu(in_channel+out_channel*3, out_channel),
            convbnrelu(out_channel, out_channel)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out2 = interpolate(out2, x.size()[2:])
        out3 = self.conv3(x)
        out3 = interpolate(out3, x.size()[2:])

        out = self.conv_out(torch.cat((x,out1,out2,out3),1))

        return out
    
class MIC_4(nn.Module): # Multi-scale Information Compensation Module
    def __init__(self, in_channel, out_channel):
        super(MIC_4, self).__init__()
        self.conv1 = convbnrelu(in_channel, out_channel)
        self.conv2 = nn.Sequential(
            nn.AvgPool2d(2,2),
            convbnrelu(in_channel, out_channel)
        )
        self.conv3 = nn.Sequential(
            nn.AvgPool2d(4,4),
            convbnrelu(in_channel, out_channel)
        )
        self.conv4 = nn.Sequential(
            nn.AvgPool2d(8,8),
            convbnrelu(in_channel, out_channel)
        )

        self.conv_out = nn.Sequential(
            convbnrelu(in_channel+out_channel*4, out_channel),
            convbnrelu(out_channel, out_channel)
        )

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(x)
        out2 = interpolate(out2, x.size()[2:])
        out3 = self.conv3(x)
        out3 = interpolate(out3, x.size()[2:])
        out4 = self.conv4(x)
        out4 = interpolate(out4, x.size()[2:])

        out = self.conv_out(torch.cat((x,out1,out2,out3,out4),1))

        return out

class SIC(nn.Module): # Spatial Information Compensation Module
    def __init__(self, in_channel, out_channel):
        super(SIC, self).__init__()
        self.cha = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, in_channel//16, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel//16, in_channel, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        self.reduce = convbnrelu(in_channel, out_channel, k=1, p=0)

    def forward(self, x_up, x_down, target_size):
        avg_pool = nn.AdaptiveAvgPool2d(target_size)
        x_up = avg_pool(x_up)

        out = torch.cat((x_up, x_down), 1)
        out_weight = self.cha(out)
        out = out * out_weight

        out = self.reduce(out)

        return out

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
    
class ConvOut(nn.Module):
    def __init__(self, in_channel):
        super(ConvOut, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 1, 3, stride=1, padding=1), 
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)

interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)