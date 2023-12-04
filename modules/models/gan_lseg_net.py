
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lseg_blocks import FeatureFusionBlock, Interpolate, _make_encoder, FeatureFusionBlock_custom, forward_vit
import clip
import numpy as np

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

        # self.text = clip.tokenize(self.labels)    
        
    def forward(self, x, labelset='',traintype="0"):
        if labelset == '':
            text = self.text[0]

        else:
            text = labelset[0] # we use here when text is not none
            if traintype == 0:
                text = text
            if traintype == 1:
                text = "Exist both salient and non salient objects, which is "+text
            if traintype == 2:
                text = "A Non-salient object which is "+text
            if traintype == 3:
                text = "A salient object which is " + text
            text = clip.tokenize(labelset).to(x.device)
        
        # print("Text in forward is: " , text)
        # print("text size ", text.size())

        if self.channels_last == True:
            x.contiguous(memory_format=torch.channels_last)

        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        
        if torch.isnan(layer_1.detach().cpu()).any():
            print("Nan Here") 
            exit()
        else:
            print(layer_1.detach().cpu())
        if torch.isnan(layer_1_rn.detach().cpu()).any():
            print("Nan Here") 
            exit()
        else:
            print(layer_1_rn.detach().cpu())
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        if torch.isnan(layer_2).any():
            print("Nan Here") 
            exit()
        if torch.isnan(layer_2_rn).any():
            print("Nan Here") 
            exit()
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        if torch.isnan(layer_3).any():
            print("Nan Here") 
            exit()
        if torch.isnan(layer_3_rn).any():
            print("Nan Here") 
            exit()
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        if torch.isnan(layer_4).any():
            print("Nan Here") 
            exit()
        if torch.isnan(layer_4_rn).any():
            print("Nan Here") 
            exit()


        path_4 = self.scratch.refinenet4(layer_4_rn)
        if torch.isnan(path_4.detach().cpu()).any():
            print("Nan Here") 
            exit()
        
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        if torch.isnan(path_3).any():
            print("nan here") 
            exit()
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        if torch.isnan(path_2).any():
            print("nan here") 
            exit()

        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        if torch.isnan(path_1).any():
            print("nan here") 
            exit()

        # text = text.to(x.device)
        self.logit_scale = self.logit_scale.to(x.device)
        text_features = self.clip_pretrained.encode_text(text)

        # text_features = text_features.to(x.device)
        # print(text_features.size())

        image_features = self.scratch.head1(path_1)

        if torch.isnan(image_features).any():
            print("nan here") 
            exit()
        imshape = image_features.shape
        image_features = image_features.permute(0,2,3,1).reshape(-1, self.out_c)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits_per_image = self.logit_scale * image_features.half() @ text_features.t()

        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0,3,1,2)


        # if self.arch_option in [1, 2]:
        #     for _ in range(self.block_depth - 1):
        #         out = self.scratch.head_block(out)
        #     out = self.scratch.head_block(out, False)

        # out = self.scratch.output_conv(out)
            
        return out

class LearnableGaussianSmoothing(nn.Module):
    def __init__(self, channels, kernel_size, sigma):
        super(LearnableGaussianSmoothing, self).__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")
        
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels

        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = torch.arange(self.kernel_size)
        x_grid = x_cord.repeat(self.kernel_size).view(self.kernel_size, self.kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        mean = (self.kernel_size - 1) // 2
        variance = sigma**2

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = (1./(2.*np.pi*variance)) * \
                          torch.exp(
                              -torch.sum((xy_grid - mean)**2., dim=-1) / \
                              (2*variance)
                          )
        # Normalize the gaussian kernel so that the sum of all its elements equals 1
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, self.kernel_size, self.kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(self.channels, 1, 1, 1)

        self.gaussian_kernel = nn.Parameter(gaussian_kernel, requires_grad=True)

    def forward(self, x):
        padding = self.kernel_size // 2
        x = F.conv2d(x, self.gaussian_kernel, stride=1, padding=padding, groups=self.channels)
        return x

class UpsampleTo520Layer(nn.Module):
    def __init__(self, initial_threshold=0.5):
        super(UpsampleTo520Layer, self).__init__()
        # Convolution to refine the features at 256x256
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        # self.relu = nn.ReLU()
        # Upsample to 520x520 using bilinear interpolation
        self.upsample = nn.Upsample(size=(520, 520), mode='bilinear', align_corners=False)
        # Convolution to refine the features at 520x520
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        # Tanh for image data normalization
        self.sigmoid = nn.Sigmoid()
        self.gaussian = LearnableGaussianSmoothing(1, 5, 1)
        self.threshold = nn.Parameter(torch.tensor([initial_threshold]))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # x= self.conv2(x)
        # x = self.relu(x)
        x = self.upsample(x)
        x = self.conv3(x)
        x = self.gaussian(x)
        x = self.sigmoid(x)
        # x_bin = (x>self.threshold).int()
        return  x

class LSegNet(LSeg):
    """Network for semantic segmentation."""
    def __init__(self, labels, path=None, scale_factor=0.5, crop_size=520, as_encoder=False, **kwargs): # originally 480 -> 520

        features = kwargs["features"] if "features" in kwargs else 256
        kwargs["use_bn"] = True

        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.labels = labels

        if not as_encoder:
            head = nn.Sequential(
                # Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                UpsampleTo520Layer(),
            )
        else:
            head = None

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

def make_conv_layers(cfg):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(2,2)]
        else:
            conv = nn.Conv2d(in_channels, v,3,1,1)

            layers += [conv, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'E': [64,'M', 128,  'M', 256, 'M', 512,'M', 512,],
    'D': [512, 512, 'U', 512,  512, 'U', 256, 256, 'U', 128, 128, 'U', 64, 64]
}

def make_encoder():
    return make_conv_layers(cfg['E'])

# def make_decoder():
#     return make_deconv_layers(cfg['D'])

# def make_deconv_layers(cfg):
#     layers = []
#     in_channels = 512
#     for v in cfg:
#         if v == 'U':
#             layers += [nn.Upsample(scale_factor=2)]
#         else:
#             deconv = nn.ConvTranspose2d(in_channels, v)
#             layers += [deconv]
#             in_channels = v
#     return nn.Sequential(*layers)

class GAN_Lseg(nn.Module):
    def __init__(self, labels, path=None, scale_factor=0.5, crop_size=520, as_encoder=False, **kwargs):
        super().__init__()
        self.input_layer = LSegNet(labels, path=None, scale_factor=0.5, crop_size=520, as_encoder=False, **kwargs)
        self.encoder = make_encoder()
        self.labels=None

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), 
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(), 
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), 
        # Tanh for image data normalization
            nn.Conv2d(64, 1, 1, padding=0), #, nn.Sigmoid()
            nn.Upsample(size=(520, 520), mode='bilinear', align_corners=False),
        # Convolution to refine the features at 520x520
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1),
            LearnableGaussianSmoothing(1, 5, 1)
        )

        self.classifier = nn.Sigmoid()       
        if path is not None:
            self.load(path)


    def forward(self, x, text, traintype):
        x = self.input_layer(x, text, traintype)
        print("input_layer")
        print(x.size())
        
        print(x)
        x = self.encoder(x)
        print("encoder")
        print(x.size())
        print(x)
        x = self.decoder(x)
        print("decoder")
        print(x.size())
        print(x)
        x = self.classifier(x)
        print(x.size())
        print(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv2d = nn.Sequential( 
            nn.Conv2d(4, 3, 1, padding=1), nn.ReLU(), 
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), 
            nn.MaxPool2d(2, stride=2),
            
            nn.Conv2d(32, 64, 3, padding=1),  nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(4, stride=4),
            
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(4, stride=4)
        )
        
        self.linear = nn.Sequential(
            nn.Linear(16384, 100), nn.Tanh(),
            nn.Linear(100,2), nn.Tanh(),
            nn.Linear(2,1), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv2d(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear(x)
        return x
        
class BCELossWithDownsampling():
    def __init__(self):
        self.downsample = nn.AvgPool2d(4, stride=4, count_include_pad=False)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def __call__(self, pred, y):
        return self.loss_fn(self.downsample(pred), self.downsample(y))

class WeightedMSELoss():
    def __init__(self):
        self.loss_fn = nn.MSELoss(reduction='none')
        self.alpha = 1.1  # hpyerparameter
        
    def __call__(self, pred, y):
        L = self.loss_fn(pred, y)
        w = 1 / ((self.alpha - y) ** 2)
        return (w * L).mean()
 