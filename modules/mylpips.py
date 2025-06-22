"""
LPIPS Implementation heavily inspired by 
https://github.com/richzhang/PerceptualSimilarity/blob/master/lpips/lpips.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
import warnings
warnings.filterwarnings("ignore")

class LPIPS(nn.Module):
    
    """
    VGG16 LPIPS Perceptual Loss as proposed in https://github.com/richzhang/PerceptualSimilarity

    Normally we do an L2 Reconstruction loss between our generated images and the real ground truth
    but the problem can be it leads to blurry pictures. This is why, instead of just minimizing 
    the reconstruction loss, we also want to minimize between VGG features extracted from our 
    real and fake images
    """

    def __init__(self, 
                 pretrained_backbone=True,
                 train_backbone=False, 
                 use_dropout=True,
                 img_range="minus_one_to_one",
                 pretrained_weights=None):
        
        super(LPIPS, self).__init__()

        self.pretrained_backbone = pretrained_backbone
        self.train_backbone = train_backbone
        self.use_dropout = use_dropout
        self.img_range = img_range

        ### Load a Pretrained VGG BackBone and its Channel Pattern###
        vgg_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained_backbone else None).features
        self.channels = [64,128,256,512,512]
        self.layer_groups = [(0,3), (4,8), (9,15), (16,22), (23,29)]

        ### Turn of Gradients on Backbone ###
        if not train_backbone:
            for param in vgg_model.parameters():
                param.requires_grad_(False)
        
        ### Compute the Norm Constants ###
        self.scale_constants(img_range)

        ### Slices of the Model ###
        slices = {}
        for i, (start, end) in enumerate(self.layer_groups):
            layers = []
            for j in range(start, end+1):
                layers.append(vgg_model[j])
            slices[f"slice{i+1}_layers"] = nn.Sequential(*layers)

        self.slices = nn.ModuleDict(slices)

        ### Now that VGG16 is Sliced and Stored, Delete Original Model ###
        del vgg_model

        ### Projections of Patches (B x C x H x W) -> (B x 1 x H x W) ###
        proj = {}
        for i, in_channels in enumerate(self.channels):
            layers = [nn.Dropout(),] if (use_dropout) else []
            layers += [nn.Conv2d(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=False)]
            proj[f"slice{i+1}_conv_proj"] = nn.Sequential(*layers)
        
        self.proj = nn.ModuleDict(proj)

        ### Spatial Pooling ###
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        ### Load Checkpoint ###
        if pretrained_weights is not None:
            print("Loading LPIPS Checkpoint:", pretrained_weights)
            self.load_checkpoint(pretrained_weights)

    def forward_vgg(self, x):

        ### Grab Outputs from Different Stages of VGG16 ###
        slice1_out = self.slices["slice1_layers"](x)
        slice2_out = self.slices["slice2_layers"](slice1_out)
        slice3_out = self.slices["slice3_layers"](slice2_out)
        slice4_out = self.slices["slice4_layers"](slice3_out)
        slice5_out = self.slices["slice5_layers"](slice4_out)

        return_outputs = {"slice1": slice1_out, 
                          "slice2": slice2_out, 
                          "slice3": slice3_out,
                          "slice4": slice4_out,
                          "slice5": slice5_out}
        
        return return_outputs
    
    def scale_constants(self, range="minus_one_to_one"):

        if range not in ["zero_to_one", "minus_one_to_one"]:
            raise ValueError("Indicate if images are zero_to_one [0,1] or minus_one_to_one [-1,1]")

        imagenet_mean = torch.tensor([0.485, 0.456, 0.406])
        imagenet_std = torch.tensor([0.229, 0.224, 0.225])

        ### Imagenet Mean assumed [0,1] images, if [-1,1] we have to rescale ###
        if range == "minus_one_to_one":

            ### If we double the range from [0,1] to [-1,1] we double the std ###
            imagenet_std = imagenet_std * 2

            ### If we double the range from [0,1] to [-1,1] we shift the mean ###
            imagenet_mean = (imagenet_mean * 2) - 1
        
        ### Add extra dimensions to broadcast over (B,3,H,W)
        imagenet_mean = imagenet_mean.reshape(1,3,1,1)
        imagenet_std = imagenet_std.reshape(1,3,1,1)

        self.register_buffer("mean", imagenet_mean)
        self.register_buffer("std", imagenet_std)

    def scale(self, x):
        return (x - self.mean) / self.std

    def unit_norm(self, x, eps=1e-8):

        ### Normalize across the channel for every pixel ###
        norm = torch.norm(x, p=2, dim=1, keepdim=True) + eps

        ### Unit Norm ##
        x = x / norm

        return x

    def load_checkpoint(self, path_to_checkpoint):
        self.load_state_dict(torch.load(path_to_checkpoint, weights_only=True))

    def forward(self, input, target):

        ### If our Images are [0,1], scale to [-1,1] ###
        if self.img_range != "minus_one_to_one":
            input = (input * 2) - 1
            target = (target * 2) - 1

        ### Normalize Inputs ###
        input, target = self.scale(input), self.scale(target)

        ### Grab VGG Features ###
        if not self.train_backbone:
            with torch.no_grad():
                input_vgg = self.forward_vgg(input)
                target_vgg = self.forward_vgg(target)
        else:
            input_vgg = self.forward_vgg(input)
            target_vgg = self.forward_vgg(target)
    
        
        ### Loop Through the Slices ###
        delta = {}
        pooled_outs = []
        for key in input_vgg.keys():
            
            ### Grab Cooresponding Features ###
            input_feat = input_vgg[key]
            target_feat = target_vgg[key]

            ### Unit Normalize the Tensors ###
            input_feat = self.unit_norm(input_feat)
            target_feat = self.unit_norm(target_feat)

            ### Compute the Square Error ###
            delta = (input_feat - target_feat) ** 2

            ### Pass through Corresponding Proj Layer ###
            proj_out = self.proj[f"{key}_conv_proj"](delta)

            ### Average Pooling ###
            pooled_out = self.pool(proj_out)
            pooled_outs.append(pooled_out)

        ### Accumulate the Outputs Across Layers ###
        ### pooled has shape (B,1,1,1) so we can simply use convolutions
        val = 0
        for pooled in pooled_outs:
            val = val + pooled
 
        return val

class DiffToLogits(nn.Module):

    """
    The output of LPIPS is a tensor of shape (B,1,1,1). This class takes in
    two differences from LPIPS: (ref vs p0) and (ref vs p1)

    We then compute some features from those differences:
        - diff1 (ref vs p0)
        - diff2 (ref vs p1)
        - difference: (diff_0 vs diff_1) -> Difference of Differences
        - ratio1: Ratio of diff1/diff2
        - ratio2: Ratio of diff2/diff1

    We can then concat all these features, providing us 5 features of differences

    """

    def __init__(self, middle_channels=32):

        super(DiffToLogits, self).__init__()

        self.model = nn.Sequential(

                nn.Conv2d(5, middle_channels, kernel_size=1, stride=1, padding=0), 
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(middle_channels, middle_channels, kernel_size=1, stride=1, padding=0), 
                nn.LeakyReLU(0.2, inplace=True), 
                nn.Conv2d(middle_channels, 1, kernel_size=1, stride=1, padding=0)

        )

    def forward(self, diff1, diff2, eps=0.1):
        
        ### Difference Feature ### 
        difference = diff1 - diff2

        ### Ratio Features ###
        ratio1 = diff1 / (diff2 + eps)
        ratio2 = diff2 / (diff1 + eps)

        ### Concat Features ([B,1,1,1], [B,1,1,1], ...) -> (B,5,1,1)###
        concat = torch.cat([diff1, diff2, difference, ratio1, ratio2], dim=1)

        return self.model(concat)
