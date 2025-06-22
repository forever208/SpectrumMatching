import torch
import torch.nn as nn

class NLayerDiscriminator(nn.Module):

    """
    PatchGAN discriminator as defined in Image to Image Translation w/ Conditional Adversarial Networks
    https://arxiv.org/pdf/1611.07004 
    """

    def __init__(self, 
                 input_channels=3, 
                 start_dim=64, 
                 depth=3,
                 kernel_size=4, 
                 padding=1,
                 leaky_relu_slope=0.2):
        
        super(NLayerDiscriminator, self).__init__()

        current_filters = start_dim
        layers = nn.ModuleList([])

        ### Projection from input_channels to start_dim ###
        layers.append(nn.Conv2d(input_channels, current_filters, kernel_size=kernel_size, stride=2, padding=padding))
        layers.append(nn.LeakyReLU(leaky_relu_slope))

        ### Loop For All the Next Layes ###
        for i in range(depth):

            ### Apply a stride of 2 on all convoutions except the last ###
            stride = 2 if i != depth-1 else 1 
            out_channels = current_filters * 2
            layers.append(nn.Conv2d(current_filters, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            
            ### Update Current Filters ###
            current_filters = out_channels

        # Output will have a single channel
        layers.append(nn.Conv2d(current_filters, 1, kernel_size=kernel_size, stride=1, padding=padding))
        
        self.model = nn.Sequential(*layers)

    def forward(self, input):
        return self.model(input)

def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.normal_(module.weight.data, 0.0, 0.2)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)
