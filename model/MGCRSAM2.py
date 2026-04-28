import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from .block import *
from sam2.build_sam import build_sam2


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        #Encoder
        ##stage 1
        self.encoder1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            backbone.layer1
        ) # 64

        ##stage 2
        self.encoder2 = backbone.layer2 # 128

        ##stage 3
        self.encoder3 = backbone.layer3 # 256

        #Channel Enhancement
        ##stage 3
        self.cem3 = SIC(256+128, 256)

        ##stage 2
        self.cem2 = SIC(128+64, 128)

        #Decoder
        ##stage 3
        self.decoder3 = MIC_4(256, 128)

        ##stage 2
        self.decoder2 = MIC_4(128+128, 64)

        ##stage 1
        self.decoder1 = MIC_4(64+64, 64)

        #Ourput
        self.conv_out1 = ConvOut(in_channel=64)
        self.conv_out2 = ConvOut(in_channel=64+1)
        self.conv_out3 = ConvOut(in_channel=128+1)

    def sal_guide(self, feature, sal, number):
        feature_list = torch.chunk(feature, number, 1)
        feature = torch.cat((feature_list[0], sal),1)
        for i in range(1,number):
            feature = torch.cat((feature, feature_list[i], sal),1)
        
        return feature

    def forward(self, x_clear, y_raw):
        #Encoder
        ##stage 1
        score1 = self.encoder1(torch.cat((x_clear, y_raw),1))

        ##stage 2
        score2 = self.encoder2(score1)

        ##stage 3
        score3 = self.encoder3(score2)

        #Channel Enhancement
        ##stage 3
        score3 = self.cem3(score2, score3, score3.size()[2:])

        ##stage 2
        score2 = self.cem2(score1, score2, score2.size()[2:])

        #Decoder
        ##stage 3
        scored3 = self.decoder3(score3)
        t = interpolate(scored3, score2.size()[2:])

        ##stage 2
        scored2 = self.decoder2(torch.cat((score2,t),1))
        t = interpolate(scored2, score1.size()[2:])

        ##stage 1
        scored1 = self.decoder1(torch.cat((score1,t),1))

        #Output
        ##stage 1
        out1 = self.conv_out1(scored1)

        ##stage 2
        avg_pool = nn.AdaptiveAvgPool2d(scored2.size()[2:])
        out1_d2 = avg_pool(out1)
        scored2 = self.sal_guide(scored2, out1_d2, 1)
        out2 = self.conv_out2(scored2)

        ##stage 3
        avg_pool = nn.AdaptiveAvgPool2d(scored3.size()[2:])
        out1_d3 = avg_pool(out1)
        scored3 = self.sal_guide(scored3, out1_d3, 1)
        out3 = self.conv_out3(scored3)

        out1 = interpolate(out1, x_clear.size()[2:])
        out2 = interpolate(out2, x_clear.size()[2:])
        out3 = interpolate(out3, x_clear.size()[2:])

        return out1, out2, out3

class MGCRSAM2(nn.Module):
    def __init__(self, checkpoint_path=None) -> None:
        super(MGCRSAM2, self).__init__()
        # Global Localization Stage
        # Encoder    
        model_cfg = "sam2_hiera_l.yaml"
        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
        else:
            model = build_sam2(model_cfg)
        del model.sam_mask_decoder
        del model.sam_prompt_encoder
        del model.memory_encoder
        del model.memory_attention
        del model.mask_downsample
        del model.obj_ptr_tpos_proj
        del model.obj_ptr_proj
        del model.image_encoder.neck
        self.encoder = model.image_encoder.trunk

        for param in self.encoder.parameters():
            param.requires_grad = False
        blocks = []
        for block in self.encoder.blocks:
            blocks.append(
                Adapter(block)
            )
        self.encoder.blocks = nn.Sequential(
            *blocks
        )

        # Reduction Block
        self.conv_reduce1 = nn.Sequential(
            nn.Conv2d(144, 36, kernel_size=1, bias=False),
            LayerNorm2d(36),
            nn.Conv2d(36, 36, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(36)
        )
        self.conv_reduce2 = nn.Sequential(
            nn.Conv2d(288, 72, kernel_size=1, bias=False),
            LayerNorm2d(72),
            nn.Conv2d(72, 72, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(72)
        )
        self.conv_reduce3 = nn.Sequential(
            nn.Conv2d(576, 144, kernel_size=1, bias=False),
            LayerNorm2d(144),
            nn.Conv2d(144, 144, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(144)
        )
        self.conv_reduce4 = nn.Sequential(
            nn.Conv2d(1152, 288, kernel_size=1, bias=False),
            LayerNorm2d(288),
            nn.Conv2d(288, 288, kernel_size=3, padding=1, bias=False),
            LayerNorm2d(288)
        )

        # Decoder
        ##stage 4
        self.decoder4 = MIC_2(288, 144)

        ##stage 3
        self.decoder3 = MIC_3(144+144, 72)

        ##stage 2
        self.decoder2 = MIC_4(72+72, 36)

        ##stage 1
        self.decoder1 = MIC_4(36+36, 36)

        # Output
        self.conv_out1 = ConvOut(36)
        self.conv_out2 = ConvOut(36+1)
        self.conv_out3 = ConvOut(72+2)
        self.conv_out4 = ConvOut(144+3)

        # Local Refinement Stage
        self.refine = RefineNet()

    def sal_guide(self, feature, sal, number):
        feature_list = torch.chunk(feature, number, 1)
        feature = torch.cat((feature_list[0], sal),1)
        for i in range(1,number):
            feature = torch.cat((feature, feature_list[i], sal),1)
        
        return feature

    def forward(self, x):
        # Encoder
        score1, score2, score3, score4 = self.encoder(x)

        # Reduce
        score1, score2 = self.conv_reduce1(score1), self.conv_reduce2(score2)
        score3, score4 = self.conv_reduce3(score3), self.conv_reduce4(score4)

        # Decoder
        ##stage 4
        scored4 = self.decoder4(score4)
        t = interpolate(scored4, score3.size()[2:])

        ##stage 3
        scored3 = self.decoder3(torch.cat((score3,t),1))
        t = interpolate(scored3, score2.size()[2:])

        ##stage 2
        scored2 = self.decoder2(torch.cat((score2,t),1))
        t = interpolate(scored2, score1.size()[2:])
        
        ##stage 1
        scored1 = self.decoder1(torch.cat((score1,t),1))

        # Output
        ##stage 1
        out1 = self.conv_out1(scored1)
        t = interpolate(out1, x.size()[2:])
        out_r1, out_r2, out_r3 = self.refine(x, t)

        ##stage 2
        avg_pool = nn.AdaptiveAvgPool2d(scored2.size()[2:])
        out_r1_d2 = avg_pool(out_r1)
        scored2 = self.sal_guide(scored2, out_r1_d2, 1)
        out2 = self.conv_out2(scored2)

        ##stage 3
        avg_pool = nn.AdaptiveAvgPool2d(scored3.size()[2:])
        out_r1_d3 = avg_pool(out_r1)
        scored3 = self.sal_guide(scored3, out_r1_d3, 2)
        out3 = self.conv_out3(scored3)

        ##stage 4
        avg_pool = nn.AdaptiveAvgPool2d(scored4.size()[2:])
        out_r1_d4 = avg_pool(out_r1)
        scored4 = self.sal_guide(scored4, out_r1_d4, 3)
        out4 = self.conv_out4(scored4)

        out1 = interpolate(out1, x.size()[2:])
        out2 = interpolate(out2, x.size()[2:])
        out3 = interpolate(out3, x.size()[2:])
        out4 = interpolate(out4, x.size()[2:])

        return out_r1, out_r2, out_r3, out1, out2, out3, out4
    
interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)