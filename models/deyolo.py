import math

import cv2
import numpy as np
import torch
import torch.nn as nn
from utils.torch_utils import initialize_weights, is_parallel, model_info
from utils.yolo_utils import check_anchor_order


# CBL = Conv2d + BatchNormalization + LeakyReLU
class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels,
                              out_channels,
                              kernel_size,
                              stride,
                              kernel_size // 2,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# CBL + CBL + Res_Add
class Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            CBL(in_channels, in_channels // 2, kernel_size=1),
            CBL(in_channels // 2, out_channels, kernel_size=3))

    def forward(self, x):
        return x + self.block(x)  # res


# DownSample
class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.downsample = CBL(in_channels,
                              out_channels,
                              kernel_size=3,
                              stride=2)

    def forward(self, x):
        return self.downsample(x)


class ResblockX(nn.Module):
    def __init__(self, in_channels, out_channels, num_resblocks):
        super().__init__()

        # x2 downsample
        self.downsample = DownSample(in_channels, out_channels)

        blocks_conv = []
        for i in range(num_resblocks):

            blocks_conv.append(Resblock(out_channels, out_channels))

        self.blocks_conv = nn.Sequential(*blocks_conv)

    def forward(self, x):
        return self.blocks_conv(self.downsample(x))


class Darknet53(nn.Module):
    def __init__(self,
                 inplanes=32,
                 num_resblocks=[1, 2, 8, 8, 4],
                 feature_channels=[64, 128, 256, 512, 1024]):
        super().__init__()

        # input_conv: 416,416,3 -> 416,416,32
        self.input_conv = CBL(3, inplanes, kernel_size=3, stride=1)

        self.Resblock_Body = nn.ModuleList([
            # Resblock1: 416,416,32 -> 208,208,64
            ResblockX(inplanes, feature_channels[0], num_resblocks[0]),
            # Resblock2: 208,208,64 -> 104,104,128
            ResblockX(feature_channels[0], feature_channels[1],
                      num_resblocks[1]),
            # Resblock3: 104,104,128 -> 52,52,256
            ResblockX(feature_channels[1], feature_channels[2],
                      num_resblocks[2]),
            # Resblock4: 52,52,256 -> 26,26,512
            ResblockX(feature_channels[2], feature_channels[3],
                      num_resblocks[3]),
            # Resblock5: 26,26,512 -> 13,13,1024
            ResblockX(feature_channels[3], feature_channels[4],
                      num_resblocks[4])
        ])

    def forward(self, x):
        x = self.input_conv(x)  # input_conv: 416,416,3 -> 416,416,32

        x = self.Resblock_Body[0](x)  # Resblock1: 416,416,32 -> 208,208,64
        x = self.Resblock_Body[1](x)  # Resblock2: 208,208,64 -> 104,104,128
        C3 = self.Resblock_Body[2](x)  # Resblock3: 104,104,128 -> 52,52,256
        C4 = self.Resblock_Body[3](C3)  # Resblock4: 52,52,256 -> 26,26,512
        C5 = self.Resblock_Body[4](C4)  # Resblock5: 26,26,512 -> 13,13,1024

        return C3, C4, C5


# Upsample
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upsample = nn.Sequential(
            CBL(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest'))

    def forward(self, x):
        x = self.upsample(x)
        return x


# CBLX5
def CBLX5(filters_list, in_filters):
    m = nn.Sequential(
        CBL(in_filters, filters_list[0], 1),
        CBL(filters_list[0], filters_list[1], 3),
        CBL(filters_list[1], filters_list[0], 1),
        CBL(filters_list[0], filters_list[1], 3),
        CBL(filters_list[1], filters_list[0], 1),
    )
    return m


class Neck(nn.Module):
    # Neck
    def __init__(self, ):
        super().__init__()
        # SPP for C5
        self.Convx5_0 = CBLX5([512, 1024], 1024)

        # FPN
        self.Upsample1 = Upsample(512, 256)
        self.Convx5_1 = CBLX5([256, 512], 768)

        self.Upsample2 = Upsample(256, 128)
        self.Convx5_2 = CBLX5([128, 256], 384)

    def forward(self, x):
        # C3(52,52,256)、C4(26,26,512)、C5(13,13,1024)
        C3, C4, C5 = x

        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        P5 = self.Convx5_0(C5)

        # 13,13,512 -> 13,13,256 -> 26,26,256
        P5_upsample = self.Upsample1(P5)
        # 26,26,512 + 26,26,256 -> 26,26,768
        P4 = torch.cat([C4, P5_upsample], axis=1)
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        P4 = self.Convx5_1(P4)

        # 26,26,256 -> 26,26,128 -> 52,52,128
        P4_upsample = self.Upsample2(P4)
        # 52,52,128 + 52,52,256 -> 52,52,384
        P3 = torch.cat([C3, P4_upsample], axis=1)
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        P3 = self.Convx5_2(P3)

        return (P3, P4, P5)


# YOLO_Head = 3*3 CBL + 1*1 Conv2d
# 19*19*((80+4+1)*3）= 19*19*255
def YOLO_Head(channels):
    m = nn.Sequential(
        CBL(channels[0], channels[1], 3),
        nn.Conv2d(channels[1], channels[2], 1),
    )
    return m


class Detect(nn.Module):
    def __init__(
        self,
        nc=80,
        anchors=[
            [10, 13, 16, 30, 33, 23],  # P3/8
            [30, 61, 62, 45, 59, 119],  # P4/16
            [116, 90, 156, 198, 373, 326]  # P5/32
        ]):
        super().__init__()

        self.stride = None  # strides computed during build
        # self.stride = torch.tensor([8, 16, 32])  # strides
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.out_ch = self.no * self.na

        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        # shape(nl,na,2)
        self.register_buffer('anchors', a)
        # self.anchors /= self.stride.view(-1, 1, 1)
        # shape(nl,1,na,1,1,2)
        self.register_buffer('anchor_grid',
                             a.clone().view(self.nl, 1, -1, 1, 1, 2))

        # channels of yolo_head for P3、P4、P5
        head_chancels = [[128, 256, self.out_ch], [256, 512, self.out_ch],
                         [512, 1024, self.out_ch]]

        # YOLO_Head # output layers
        self.m = nn.ModuleList(YOLO_Head(ch) for ch in head_chancels)

    def forward(self, x):
        z = []  # inference output

        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny,
                             nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()

                y[...,
                  0:2] = (y[..., 0:2] * 2 - 0.5 +
                          self.grid[i].to(x[i].device)) * self.stride[i]  # xy

                y[..., 2:4] = (y[..., 2:4] * 2)**2 * self.anchor_grid[i]  # wh

                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()


class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, kernel_size=5, channels=3):
        super().__init__()

        self.num_high = num_high
        self.kernel = self.gauss_kernel(kernel_size, channels)

    def gauss_kernel(self, kernel_size, channels):
        kernel = cv2.getGaussianKernel(kernel_size, 0).dot(
            cv2.getGaussianKernel(kernel_size, 0).T)
        kernel = torch.FloatTensor(kernel).unsqueeze(0).repeat(
            channels, 1, 1, 1)
        kernel = torch.nn.Parameter(data=kernel, requires_grad=False)
        return kernel

    def conv_gauss(self, x, kernel):
        n_channels, _, kw, kh = kernel.shape
        x = torch.nn.functional.pad(x, (kw // 2, kh // 2, kw // 2, kh // 2),
                                    mode='reflect')  # replicate    # reflect
        x = torch.nn.functional.conv2d(x, kernel, groups=n_channels)
        return x

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def pyramid_down(self, x):
        return self.downsample(self.conv_gauss(x, self.kernel))

    def upsample(self, x):
        up = torch.zeros((x.size(0), x.size(1), x.size(2) * 2, x.size(3) * 2),
                         device=x.device)
        up[:, :, ::2, ::2] = x * 4

        return self.conv_gauss(up, self.kernel)

    def pyramid_decom(self, img):
        self.kernel = self.kernel.to(img.device)
        current = img
        pyr = []
        for _ in range(self.num_high):
            down = self.pyramid_down(current)
            up = self.upsample(down)
            diff = current - up
            pyr.append(diff)
            current = down
        pyr.append(current)
        return pyr

    def pyramid_recons(self, pyr):
        image = pyr[0]
        for level in pyr[1:]:
            up = self.upsample(image)
            image = up + level
        return image


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_features, in_features, 3, padding=1),
            nn.LeakyReLU(True),
            nn.Conv2d(in_features, in_features, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout) * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        assert kernel_size in (3, 5, 7), "kernel size must be 3 or 5 or 7"

        self.conv = nn.Conv2d(2,
                              1,
                              kernel_size,
                              padding=kernel_size // 2,
                              bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avgout, maxout], dim=1)
        attention = self.conv(attention)
        return self.sigmoid(attention) * x


class Trans_guide(nn.Module):
    def __init__(self, ch=16):
        super().__init__()

        self.layer = nn.Sequential(
            nn.Conv2d(6, ch, 3, padding=1),
            nn.LeakyReLU(True),
            SpatialAttention(3),
            nn.Conv2d(ch, 3, 3, padding=1),
        )

    def forward(self, x):
        return self.layer(x)


class Trans_low(nn.Module):
    def __init__(
        self,
        ch_blocks=64,
        ch_mask=16,
    ):
        super().__init__()

        self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1),
                                     nn.LeakyReLU(True),
                                     nn.Conv2d(16, ch_blocks, 3, padding=1),
                                     nn.LeakyReLU(True))

        self.mm1 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=1,
                             padding=0)
        self.mm2 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=3,
                             padding=3 // 2)
        self.mm3 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=5,
                             padding=5 // 2)
        self.mm4 = nn.Conv2d(ch_blocks,
                             ch_blocks // 4,
                             kernel_size=7,
                             padding=7 // 2)

        self.decoder = nn.Sequential(nn.Conv2d(ch_blocks, 16, 3, padding=1),
                                     nn.LeakyReLU(True),
                                     nn.Conv2d(16, 3, 3, padding=1))

        self.trans_guide = Trans_guide(ch_mask)

    def forward(self, x):
        x1 = self.encoder(x)
        x1_1 = self.mm1(x1)
        x1_2 = self.mm1(x1)
        x1_3 = self.mm1(x1)
        x1_4 = self.mm1(x1)
        x1 = torch.cat([x1_1, x1_2, x1_3, x1_4], dim=1)
        x1 = self.decoder(x1)

        out = x + x1
        out = torch.relu(out)

        mask = self.trans_guide(torch.cat([x, out], dim=1))
        return out, mask


class SFT_layer(nn.Module):
    def __init__(self, in_ch=3, inter_ch=32, out_ch=3, kernel_size=3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2),
            nn.LeakyReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(inter_ch, out_ch, kernel_size, padding=kernel_size // 2))
        self.shift_conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2))
        self.scale_conv = nn.Sequential(
            nn.Conv2d(in_ch, inter_ch, kernel_size, padding=kernel_size // 2))

    def forward(self, x, guide):
        x = self.encoder(x)
        scale = self.scale_conv(guide)
        shift = self.shift_conv(guide)
        x += x * scale + shift
        x = self.decoder(x)
        return x


class Trans_high(nn.Module):
    def __init__(self, in_ch=3, inter_ch=32, out_ch=3, kernel_size=3):
        super().__init__()

        self.sft = SFT_layer(in_ch, inter_ch, out_ch, kernel_size)

    def forward(self, x, guide):
        return x + self.sft(x, guide)


class Up_guide(nn.Module):
    def __init__(self, kernel_size=1, ch=3):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(ch,
                      ch,
                      kernel_size,
                      stride=1,
                      padding=kernel_size // 2,
                      bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


# base model for yolo
class YOLO_BASE(nn.Module):
    def __init__(self, class_names, hyp, verbose=False):
        super().__init__()
        # model config
        self.class_names = list(class_names)  # class names
        self.num_classes = len(class_names)  # number of classes
        self.hyp = hyp  # attach hyperparameters to model
        self.gr = 1.0  # box loss ratio (obj_loss = 1.0 or giou)

    def forward(self, x):
        # backbone
        C3, C4, C5 = self.backbone(x)
        # neck
        # P3(batch_size, 255, 52, 52)
        # P4(batch_size,255,26,26)
        # P5(batch_size,255,13,13)
        P3, P4, P5 = self.neck([C3, C4, C5])  # P : [P3, P4, P5]
        # head
        yolo_out = self.Detect([P3, P4, P5])

        # return
        return yolo_out

    def init_weight(self, ):
        initialize_weights(self)

    def _initialize_biases(self, cf=None):
        # initialize biases into Detect(), cf is class frequency
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.Detect  # Detect() module
        for i in range(m.nl):
            s = self.stride[i]
            mi = m.m[i][-1]
            # conv.bias(255) to (3,85)
            b = mi.bias.view(m.na, -1)
            # obj (8 objects per 640 image)
            b[:, 4] += math.log(8 / (640 / s)**2)
            # cls
            b[:, 5:] += math.log(
                0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf /
                                                                  cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def build_strides_anchors(self):
        # Build strides, anchors
        m = self.Detect  # Detect() module
        m.stride = torch.tensor([8, 16, 32])
        m.anchors /= m.stride.view(-1, 1, 1)
        check_anchor_order(m)
        self.stride = m.stride
        self._initialize_biases()  # only run once

    def _load_state_dict_(self, new_state_dict):
        state_dict = self.state_dict()
        match_state_dict = {
            k: v
            for k, v in new_state_dict.items()
            if k in state_dict.keys() and state_dict[k].numel() == v.numel()
        }
        state_dict.update(match_state_dict)
        self.load_state_dict(state_dict, strict=False)
        print("Transferred state_dict (%g / %g items) for model" %
              (len(match_state_dict), len(new_state_dict)))

    def is_parallel(self, ):
        return is_parallel(self)

    def model_info(self, verbose=False):
        model_info(model=self, verbose=verbose)

    def update_hyp(self, img_size, batch_size):
        # nominal batch size
        nbs = self.hyp["nbs"]
        # accumulate for nbs
        self.hyp["accumulate"] = max(round(nbs / batch_size), 1)
        # scale weight_decay
        self.hyp['weight_decay'] *= batch_size * self.hyp["accumulate"] / nbs
        # number of detection layers (used for scaling yolo loss weight)
        nl = self.Detect.nl
        # scale to layers
        self.hyp['box'] *= 3. / nl
        # scale to classes and layers
        self.hyp['cls'] *= self.num_classes / 80. * 3. / nl
        # scale to image size and layers
        self.hyp['obj'] *= (img_size / 640)**2 * 3. / nl
        # return
        return self.hyp


# YOLOv3
class YOLOv3(YOLO_BASE):
    def __init__(self, class_names, hyp, verbose=False):
        super().__init__(class_names, hyp, verbose)

        # backbone
        self.backbone = Darknet53()
        # Neck
        self.neck = Neck()
        # head
        self.Detect = Detect(nc=self.num_classes)

        # Build strides, anchors
        self.build_strides_anchors()

        # init model weight
        self.init_weight()

        # print model info
        self.model_info(verbose=False)


class DENet(nn.Module):
    def __init__(self,
                 num_high=3,
                 ch_blocks=32,
                 up_ksize=1,
                 high_ch=32,
                 high_ksize=3,
                 ch_mask=32,
                 gauss_kernel=5):
        super().__init__()
        self.num_high = num_high
        self.lap_pyramid = Lap_Pyramid_Conv(num_high, gauss_kernel)
        self.trans_low = Trans_low(ch_blocks, ch_mask)

        for i in range(0, self.num_high):
            self.__setattr__('up_guide_layer_{}'.format(i),
                             Up_guide(up_ksize, ch=3))
            self.__setattr__('trans_high_layer_{}'.format(i),
                             Trans_high(3, high_ch, 3, high_ksize))

    def forward(self, x):
        pyrs = self.lap_pyramid.pyramid_decom(img=x)

        trans_pyrs = []
        trans_pyr, guide = self.trans_low(pyrs[-1])
        trans_pyrs.append(trans_pyr)

        commom_guide = []
        for i in range(self.num_high):
            guide = self.__getattr__('up_guide_layer_{}'.format(i))(guide)
            commom_guide.append(guide)

        for i in range(self.num_high):
            trans_pyr = self.__getattr__('trans_high_layer_{}'.format(i))(
                pyrs[-2 - i], commom_guide[i])
            trans_pyrs.append(trans_pyr)

        out = self.lap_pyramid.pyramid_recons(trans_pyrs)

        return out


class DEYOLO(YOLO_BASE):
    def __init__(self, class_names, hyp, verbose=False):
        super().__init__(class_names, hyp, verbose)
        # enhancement
        self.enhancement = DENet()

        # backbone
        self.backbone = Darknet53()
        # Neck
        self.neck = Neck()
        # head
        self.Detect = Detect(nc=self.num_classes)

        # Build strides, anchors
        self.build_strides_anchors()

        # init model weight
        self.init_weight()

        # self.freeze_parms()

        # print model info
        self.model_info(verbose=False)

    def freeze_parms(self, backbone=True, neck=True, head=True):
        for v in self.backbone.parameters():
            v.requires_grad = False
        for v in self.neck.parameters():
            v.requires_grad = False
        for v in self.Detect.parameters():
            v.requires_grad = False

    def forward(self, x):
        x = self.enhancement(x)

        C3, C4, C5 = self.backbone(x)

        # neck
        C3, C4, C5 = self.neck([C3, C4, C5])
        # head
        yolo_out = self.Detect([C3, C4, C5])

        # return
        return yolo_out
