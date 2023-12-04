import math
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lseg_blocks import FeatureFusionBlock, Interpolate, _make_encoder, FeatureFusionBlock_custom, forward_vit
import clip
import numpy as np
import pandas as pd
import os

class depthwise_clipseg_conv(nn.Module):
    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)
    
    def depthwise_clipseg(self, x, channels):
        x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out


class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()


    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x

class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file.
        Args:
            path (str): file path
        """
        parameters = torch.load(path, map_location=torch.device("cpu"))

        if "optimizer" in parameters:
            parameters = parameters["model"]

        self.load_state_dict(parameters)

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class LSeg(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="clip_vitl16_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(LSeg, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs["arch_option"]
        if self.arch_option == 1:
            self.scratch.head_block = bottleneck_block(activation=kwargs["activation"])
            self.block_depth = kwargs['block_depth']
        elif self.arch_option == 2:
            self.scratch.head_block = depthwise_block(activation=kwargs["activation"])
            self.block_depth = kwargs['block_depth']

        self.scratch.output_conv = head

        self.text = clip.tokenize(self.labels)    
        
    def forward(self, x, labelset='',traintype="0"):
        if labelset == '':
            text = self.text[0]

        else:
            text = labelset[0] # we use here when text is not none
            # if traintype == 0:
            #     text = text
            # if traintype == 1:
            #     text = "Exist both salient and non salient objects, which is "+text
            # if traintype == 2:
            #     text = "A Non-salient object which is "+text
            # if traintype == 3:
            #     text = "A salient object which is " + text
            text = clip.tokenize(labelset).to(x.device)
        
        # print("Text in forward is: " , text)
        # print("text size ", text.size())

        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # text = text.to(x.device)
        self.logit_scale = self.logit_scale.to(x.device)
        text_features = self.clip_pretrained.encode_text(text)
        # text_features = text_features.to(x.device)
        # print(text_features.size())

        image_features = self.scratch.head1(path_1)

        imshape = image_features.shape
        image_features = image_features.permute(0,2,3,1).reshape(-1, self.out_c)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits_per_image = self.logit_scale * image_features.half() @ text_features.t()

        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)

        if self.arch_option in [1, 2]:
            for _ in range(self.block_depth - 1):
                out = self.scratch.head_block(out)
            out = self.scratch.head_block(out, False)

        out = self.scratch.output_conv(out)
            
        return out

# class LearnableGaussianSmoothing(nn.Module):
#     def __init__(self, channels, kernel_size, sigma):
#         super(LearnableGaussianSmoothing, self).__init__()
#         if kernel_size % 2 == 0:
#             raise ValueError("Kernel size must be odd")
        
#         self.kernel_size = kernel_size
#         self.sigma = sigma
#         self.channels = channels

#         # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
#         x_cord = torch.arange(self.kernel_size)
#         x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
#         y_grid = x_grid.t()
#         xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

#         mean = (self.kernel_size - 1) // 2
#         variance = sigma**2

#         # Calculate the 2-dimensional gaussian kernel
#         gaussian_kernel = (1./(2.*np.pi*variance)) * \
#                           torch.exp(
#                               -torch.sum((xy_grid - mean)**2., dim=-1) / \
#                               (2*variance)
#                           )
#         # Normalize the gaussian kernel so that the sum of all its elements equals 1
#         gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

#         # Reshape to 2d depthwise convolutional weight
#         gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
#         gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)

#         self.gaussian_kernel = nn.Parameter(gaussian_kernel, requires_grad=True)

#     def forward(self, x):
#         padding = self.kernel_size // 2
#         x = F.conv2d(x, self.gaussian_kernel, stride=1, padding=padding, groups=self.channels)
#         return x

class LearnableGaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma):
        super(LearnableGaussianSmoothing, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")
        
        self.kernel_size = kernel_size
        self.sigma = nn.Parameter(torch.full((channels,), sigma), requires_grad=True)  # Make sigma learnable
        self.channels = channels

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(self.kernel_size)
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        self.register_buffer('xy_grid', xy_grid - (self.kernel_size - 1) // 2)

    def forward(self, x):
        variance = self.sigma.unsqueeze(1).unsqueeze(1)**2
        gaussian_kernel = (1./(2.*math.pi*variance)) * \
                          torch.exp(
                              -torch.sum((self.xy_grid.unsqueeze(0) - self.xy_grid.mean())**2., dim=-1) / \
                              (2*variance)
                          )
        # Normalize the gaussian kernel so that the sum of all its elements equals 1
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel, dim=[1, 2], keepdim=True)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(self.channels, 1, self.kernel_size, self.kernel_size)

        padding = self.kernel_size // 2
        x = F.conv2d(x, gaussian_kernel, stride=1, padding=padding, groups=self.channels)
        return x

class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, upscale_factor):
        super(PixelShuffleUpsample, self).__init__()
        # 增加通道数量以准备进行Pixel Shuffle操作
        self.conv = nn.Conv2d(in_channels, in_channels * (upscale_factor ** 2),
                              kernel_size=3, stride=1, padding=1)
        self.shuffle = nn.PixelShuffle(upscale_factor)
        # 卷积后再次使用非线性激活函数和批归一化以增强特征
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class UpsampleTo512Layer(nn.Module):
    def __init__(self):
        super(UpsampleTo512Layer, self).__init__()
        # 特征提取与增强
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(64)
        # 将图像上采样到512x512
        self.upsample = PixelShuffleUpsample(64, upscale_factor=2)
        # 对上采样后的图像进行特征提炼
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(32)
        # 最后调整通道数为1
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
        self.gaussian = LearnableGaussianSmoothing(1, 5, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.upsample(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.gaussian(x)
        x = self.sigmoid(x)
        return x

# class UpsampleTo512Layer(nn.Module):
#     def __init__(self, initial_threshold=0.5):
#         super(UpsampleTo512Layer, self).__init__()
#         # Convolution to refine the features at 256x256
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
#         self.relu = nn.ReLU()
#         self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1)
#         self.relu2 = nn.ReLU()
#         # Upsample to 512x512 using bilinear interpolation
#         self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=False)
#         # Convolution to refine the features at 512x512
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1)
#         # Tanh for image data normalization
#         self.sigmoid = nn.Sigmoid()
#         self.gaussian = LearnableGaussianSmoothing(1, 5, 1)
#         # self.threshold = nn.Parameter(torch.tensor([initial_threshold]))

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu(x)
#         x= self.conv2(x)
#         x = self.relu2(x)
#         x = self.upsample(x)
#         x = self.conv3(x)
#         x = self.gaussian(x)
#         x = self.sigmoid(x)
#         # x_bin = (x>self.threshold).float()
#         return x

class LSegNet(LSeg):
    """Network for semantic segmentation."""
    def __init__(self, labels, path=None, scale_factor=0.5, crop_size=512, **kwargs): # originally 480 -> 512

        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True

        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.labels = labels

        head = nn.Sequential(
            # Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            UpsampleTo512Layer(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)


    
        
    