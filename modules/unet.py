import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import ResidualBlock2D, DownSampleBlock2D, UpSampleBlock2D
from .transformer import TransformerBlock2D

class DownBlock2D(nn.Module):

    """
    UNet2D Encoder Block that stores the skip connections 

    Args:
        in_channels: Number of input channels to the block
        out_channels: Number of output channels from the block
        attention: Do you want a Transformer Block
        num_attention_heads: Number of attention heads in Attention computation
        cross_attention_dim: Text (context) embed dimension (if None, cross attention turned off)
        time_embed_dim: Input time embedding dimension
        dropout: What dropout probability do you want
        num_layers: How many Residual/Attention blocks do you want in this block
        transformers_per_layer: How many transformer layers do you want inside the transformer block
        attention_bias: Do you want QKV projections to have a bias?
        norm_eps: Groupnorm eps
        groupnorm_groups: How many groups in groupnorm
        add_downsample: Do you want a downsample layer?
        downsample_factor: By what factor do you want to downsample
        downsample_kernel_size: What kernel size for the downsampling convolution
    """
    
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 attention=True,
                 num_attention_heads=1,
                 cross_attention_dim=None,
                 time_embed_dim=1280, 
                 class_conditioning=False,
                 class_embed_dim=512,
                 dropout=0.0,  
                 num_layers=1, 
                 transformers_per_layer=1, 
                 transformer_dim_mult=4,
                 attention_bias=False,
                 norm_eps=1e-6, 
                 groupnorm_groups=32, 
                 add_downsample=True,
                 downsample_factor=2,
                 downsample_kernel_size=3):

        super(DownBlock2D, self).__init__()
        
        self.add_downsample = add_downsample

        self.resnets = nn.ModuleList([])
        self.attentions = nn.ModuleList([])

        for i in range(num_layers):
            
            in_channels = in_channels if i == 0 else out_channels

            self.resnets.append(

                ResidualBlock2D(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    dropout_p=dropout, 
                    groupnorm_groups=groupnorm_groups,
                    time_embed_proj=True, 
                    time_embed_dim=time_embed_dim,
                    class_embed_proj=class_conditioning,
                    class_embed_dim=class_embed_dim,
                    norm_eps=norm_eps
                )
            )

            if attention:
                self.attentions.append(

                    TransformerBlock2D(
                        num_attention_heads=num_attention_heads, 
                        attention_head_dim=out_channels // num_attention_heads,
                        in_channels=out_channels, 
                        cross_attn_dim=cross_attention_dim, 
                        num_layers=transformers_per_layer, 
                        dim_mult=transformer_dim_mult,
                        groupnorm_groups=groupnorm_groups, 
                        attn_bias=attention_bias,
                        norm_eps=norm_eps,
                        dropout_p=dropout
                    )
                )
            
            else:

                self.attentions.append(None)

        if self.add_downsample:
            self.downsample = DownSampleBlock2D(in_channels=out_channels, 
                                                kernel_size=downsample_kernel_size, 
                                                downsample_factor=downsample_factor)
            
    def forward(self, 
                x, 
                time_embed, 
                text_conditioning=None, 
                text_attention_mask=None,
                class_conditioning=None):

        skip_connection_outputs = []

        for i, (res, attn) in enumerate(zip(self.resnets, self.attentions)):

            x = res(x, time_embed, class_conditioning)

            if attn is not None:
                x = attn(x, text_conditioning, text_attention_mask)

            skip_connection_outputs.append(x)

        if self.add_downsample:
            x = self.downsample(x)
            skip_connection_outputs.append(x)

        return x, skip_connection_outputs
    
class MidBlock2D(nn.Module):

    """
    UNet2D Mid Block (BottleNeck of the UNet) 

    Args:
        in_channels: Number of input channels to the block
        out_channels: Number of output channels from the block
        attention: Do you want a Transformer Block
        num_attention_heads: Number of attention heads in Attention computation
        cross_attention_dim: Text (context) embed dimension (if None, cross attention turned off)
        time_embed_dim: Input time embedding dimension
        dropout: What dropout probability do you want
        num_layers: How many Residual/Attention blocks do you want in this block
        transformers_per_layer: How many transformer layers do you want inside the transformer block
        attention_bias: Do you want QKV projections to have a bias?
        norm_eps: Groupnorm eps
        groupnorm_groups: How many groups in groupnorm
    """

    
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 attention=True,
                 num_attention_heads=1,
                 cross_attention_dim=None,
                 time_embed_dim=1280, 
                 class_conditioning=False,
                 class_embed_dim=512,
                 dropout=0.0,  
                 num_layers=1, 
                 transformers_per_layer=1, 
                 transformer_dim_mult=4,
                 attention_bias=False,
                 norm_eps=1e-6, 
                 groupnorm_groups=32):
        
        super(MidBlock2D, self).__init__()
        
        ### Always Starts with a ResNetBlock ###
        self.resnets = nn.ModuleList(
            [
                ResidualBlock2D(
                    in_channels=in_channels, 
                    out_channels=out_channels, 
                    dropout_p=dropout, 
                    groupnorm_groups=groupnorm_groups,
                    time_embed_proj=True, 
                    time_embed_dim=time_embed_dim,
                    class_embed_proj=class_conditioning, 
                    class_embed_dim=class_embed_dim,
                    norm_eps=norm_eps
                )
            ]
        )

        self.attentions = nn.ModuleList([])

        for i in range(num_layers):
            
            in_channels = in_channels if i == 0 else out_channels

            self.resnets.append(

                ResidualBlock2D(
                    in_channels=out_channels, 
                    out_channels=out_channels, 
                    dropout_p=dropout, 
                    groupnorm_groups=groupnorm_groups,
                    time_embed_proj=True, 
                    time_embed_dim=time_embed_dim,
                    class_embed_proj=class_conditioning,
                    class_embed_dim=class_embed_dim,
                    norm_eps=norm_eps
                )
            )

            if attention:

                self.attentions.append(

                    TransformerBlock2D(
                        num_attention_heads=num_attention_heads, 
                        attention_head_dim=out_channels // num_attention_heads,
                        in_channels=out_channels, 
                        cross_attn_dim=cross_attention_dim, 
                        num_layers=transformers_per_layer, 
                        dim_mult=transformer_dim_mult,
                        groupnorm_groups=groupnorm_groups, 
                        attn_bias=attention_bias,
                        norm_eps=norm_eps,
                        dropout_p=dropout
                    )
                )
            
            else:

                self.attentions.append(None)
    
    def forward(self, 
                x, 
                time_embed, 
                text_conditioning=None, 
                text_attention_mask=None,
                class_conditioning=None):

        x = self.resnets[0](x, time_embed, class_conditioning)

        for i, (res, attn) in enumerate(zip(self.resnets[1:], self.attentions)):

            x = res(x, time_embed, class_conditioning)

            if attn is not None:
                x = attn(x, text_conditioning, text_attention_mask)

        return x

class UpBlock2D(nn.Module):

    """
    UNet2D Decoder Block that concatenates skip connections from DownBlock2D
    onto decoded tensors

    Args:
        in_channels: Number of input channels to the block
        out_channels: Number of output channels from the block
        attention: Do you want a Transformer Block
        num_attention_heads: Number of attention heads in Attention computation
        cross_attention_dim: Text (context) embed dimension (if None, cross attention turned off)
        time_embed_dim: Input time embedding dimension
        dropout: What dropout probability do you want
        num_layers: How many Residual/Attention blocks do you want in this block
        transformers_per_layer: How many transformer layers do you want inside the transformer block
        attention_bias: Do you want QKV projections to have a bias?
        norm_eps: Groupnorm eps
        groupnorm_groups: How many groups in groupnorm
        add_upsample: Do you want a upsample layer?
        upsample_factor: By what factor do you want to upsample
        upsample_kernel_size: What kernel size for the upsampling convolution
    """
    
    def __init__(self, 
                 in_channels, 
                 out_channels,
                 prev_out_channels, 
                 attention=True,
                 num_attention_heads=1,
                 cross_attention_dim=None,
                 time_embed_dim=1280, 
                 class_conditioning=False, 
                 class_embed_dim=512,
                 dropout=0.0,  
                 num_layers=1, 
                 transformers_per_layer=1, 
                 transformer_dim_mult=4,
                 attention_bias=False,
                 norm_eps=1e-6, 
                 groupnorm_groups=32, 
                 add_upsample=True,
                 upsample_factor=2,
                 upsample_kernel_size=3):
        
        super(UpBlock2D, self).__init__()
        
        self.add_upsample = add_upsample

        self.resnets = nn.ModuleList([])
        self.attentions = nn.ModuleList([])

        for i in range(num_layers):
            
            ### At the end of the block, our residual connection will have the number of
            ### channels of the next block. For example, if our channels are:
            ### (128, 256, 512, 1024). The outputs of the encoder will look like the following:
            ### This assumes we have 2 layer per block in the encoder

            # torch.Size([4, 128, 32, 32])
            # torch.Size([4, 128, 32, 32])
            # torch.Size([4, 128, 32, 32])

            # torch.Size([4, 128, 16, 16])
            # torch.Size([4, 256, 16, 16])
            # torch.Size([4, 256, 16, 16])

            # torch.Size([4, 256, 8, 8])
            # torch.Size([4, 512, 8, 8])
            # torch.Size([4, 512, 8, 8])

            # torch.Size([4, 512, 4, 4])
            # torch.Size([4, 1024, 4, 4])
            # torch.Size([4, 1024, 4, 4])

            ### Therefore, when we have our 3 layers per channel in our decoder. The first two 
            ### of those 3 channels will get out_channels (1024) channels from the encoder
            ### The third layer will get in_channels (512) channels
            ### Therefore our resnet_skip_channels will be out_channels on the first two layers
            ### and in_channels on the last layer
            resnet_skip_channels = in_channels if (i == num_layers-1) else out_channels
            
            ### The out_channels set the output of every convolution as well. The prev_out_channels
            ### give us the number of channels coming from the previous output, but after that, this 
            ### tensor is concatenated to our residual and then mapped to output_channels. Therefore, 
            ### all other tensors have output_channels as their resnet_in_channels

            ### Pattern: 

            ### Block 1)
            ### in_channels: 512 out_channels: 1024 prev_out_channels: 1024
            ### 1) Input: [4, 1024, 4, 4] + Residual: [4, 1024, 4, 4] -> conv(2048, 1024) -> [4, 1024, 4, 4]
            ### 1) Input: [4, 1024, 4, 4] + Residual: [4, 1024, 4, 4] -> conv(2048, 1024) -> [4, 1024, 4, 4]
            ### 1) Input: [4, 1024, 4, 4] + Residual: [4, 512, 4, 4] -> conv(1536, 1024) -> [4, 1024, 4, 4] -> upsample -> [4, 1024, 8, 8]

            ### Block 2)
            ### in_channels: 256 out_channels: 512 prev_out_channels: 1024
            ### 1) Input: [4, 1024, 8, 8] + Residual: [4, 512, 8, 8] -> conv(1536, 512) -> [4, 512, 8, 8]
            ### 1) Input: [4, 512, 8, 8] + Residual: [4, 512, 8, 8] -> conv(1024, 512) -> [4, 512, 8, 8]
            ### 1) Input: [4, 512, 8, 8] + Residual: [4, 256, 8, 8] -> conv(768, 512) -> [4, 512, 8, 8] -> upsample -> [4, 1024, 16, 16]

            ### Block 3)
            ### in_channels: 128 out_channels: 256 prev_out_channels: 512
            ### 1) Input: [4, 512, 16, 16] + Residual: [4, 256, 16, 16] -> conv(768, 256) -> [4, 256, 16, 16]
            ### 1) Input: [4, 256, 16, 16] + Residual: [4, 256, 16, 16] -> conv(512, 256) -> [4, 256, 16, 16]
            ### 1) Input: [4, 256, 16, 16] + Residual: [4, 128, 16, 16] -> conv(384, 256) -> [4, 256, 16, 16] -> upsample -> [4, 1024, 32, 32]

            ### Block 4)
            ### in_channels: 128 out_channels: 128 prev_out_channels: 256
            ### 1) Input: [4, 256, 32, 32] + Residual: [4, 128, 32, 32] -> conv(384, 128) -> [4, 128, 32, 32]
            ### 1) Input: [4, 128, 32, 32] + Residual: [4, 128, 32, 32] -> conv(256, 128) -> [4, 128, 32, 32]
            ### 1) Input: [4, 128, 32, 32] + Residual: [4, 128, 32, 32] -> conv(256, 128) -> [4, 128, 32, 32]
            
            resnet_in_channels = prev_out_channels if i == 0 else out_channels

            self.resnets.append(

                ResidualBlock2D(
                    in_channels=resnet_in_channels + resnet_skip_channels,
                    out_channels=out_channels, 
                    dropout_p=dropout, 
                    groupnorm_groups=groupnorm_groups, 
                    time_embed_proj=True, 
                    time_embed_dim=time_embed_dim, 
                    class_embed_proj=class_conditioning,
                    class_embed_dim=class_embed_dim,
                    norm_eps=norm_eps
                )
            )

            if attention:

                self.attentions.append(

                    TransformerBlock2D(
                        num_attention_heads=num_attention_heads, 
                        attention_head_dim=out_channels // num_attention_heads,
                        in_channels=out_channels, 
                        cross_attn_dim=cross_attention_dim, 
                        num_layers=transformers_per_layer, 
                        dim_mult=transformer_dim_mult,
                        groupnorm_groups=groupnorm_groups, 
                        attn_bias=attention_bias,
                        norm_eps=norm_eps,
                        dropout_p=dropout
                    )
                )
            
            else:

                self.attentions.append(None)
        
        if self.add_upsample:
            self.upsample = UpSampleBlock2D(in_channels=out_channels, 
                                            kernel_size=upsample_kernel_size, 
                                            upsample_factor=upsample_factor)

            
    def forward(self, 
                x, 
                residual_block_tuple, 
                time_embed, 
                text_conditioning=None, 
                text_attention_mask=None,
                class_conditioning=None):

        for res, attn in zip(self.resnets, self.attentions):

            ### Remember, Residual Connections from the Encoder have been reversed already from the Decoder ###
            ### We can use Pop to grab and remove the first element of the list simultaneously ###
            skip_connection = residual_block_tuple.pop(0)

            ### Concatenate Encoder Tensor to Decoder via the Channel Dimension ###
            x = torch.cat([x, skip_connection], dim=1)

            ### Pass through ResNet and Transformer Block ###
            x = res(x, time_embed, class_conditioning)
            
            if attn is not None:
                x = attn(x, text_conditioning, text_attention_mask)

        if self.add_upsample:

            x = self.upsample(x)

        return x


class UNet2DModel(nn.Module): 

    """
    UNet2D Model w/ Temporal Embeddings for Diffusion

    A lot of code inspiration was taken from diffusers/unet!
    https://github.com/huggingface/diffusers/tree/main/src/diffusers/models/unets


    """
    
    def __init__(self, config):
        super(UNet2DModel, self).__init__()

        self.config = config

        ### Input Convolution ###
        self.conv_in = nn.Conv2d(in_channels=config.latent_channels,
                                 out_channels=config.unet_channels_per_block[0],
                                 kernel_size=3, 
                                 padding="same") 

        ### Encoder ###
        self.down_blocks = nn.ModuleList()
        output_channels = config.unet_channels_per_block[0]
        for i, block_type in enumerate(config.down_block_types):
            input_channels = output_channels
            output_channels = config.unet_channels_per_block[i]
            is_final_block = (i == len(config.unet_channels_per_block)-1)

            ### Check if This Block Has Attention ###
            enc_use_attention = True if block_type == "AttnDown" else False

            block = DownBlock2D(in_channels=input_channels, 
                                out_channels=output_channels, 
                                attention=enc_use_attention,
                                num_attention_heads=output_channels//config.attention_head_dim,
                                cross_attention_dim=config.text_embed_dim if config.text_conditioning else None,
                                time_embed_dim=config.time_embed_proj_dim, 
                                class_conditioning=config.class_conditioning,
                                class_embed_dim=config.class_embed_dim,
                                dropout=config.dropout,  
                                num_layers=config.unet_residual_layers_per_block, 
                                transformers_per_layer=config.transformer_blocks_per_layer, 
                                transformer_dim_mult=config.transformer_dim_mult,
                                attention_bias=config.attention_bias,
                                norm_eps=config.norm_eps, 
                                groupnorm_groups=config.groupnorm_groups, 
                                add_downsample=not is_final_block,
                                downsample_factor=config.unet_up_down_factor,
                                downsample_kernel_size=config.unet_up_down_kernel_size)

            self.down_blocks.append(block)

        ### MidBlock ###
        mid_use_attention = True if "Attn" in config.mid_block_types else False 
        channels = config.unet_channels_per_block[-1]
        self.mid_block = MidBlock2D(in_channels=channels, 
                                    out_channels=channels,
                                    attention=mid_use_attention,
                                    num_attention_heads=channels//config.attention_head_dim,
                                    cross_attention_dim=config.text_embed_dim if config.text_conditioning else None,
                                    time_embed_dim=config.time_embed_proj_dim, 
                                    class_conditioning=config.class_conditioning,
                                    class_embed_dim=config.class_embed_dim,
                                    dropout=config.dropout,  
                                    transformer_dim_mult=config.transformer_dim_mult,
                                    attention_bias=config.attention_bias,
                                    norm_eps=config.norm_eps, 
                                    groupnorm_groups=config.groupnorm_groups)

        ### Decoder ###
        self.up_blocks = nn.ModuleList()
        reversed_channels_per_block = list(reversed(config.unet_channels_per_block))

        ### The initial output (coming from the midblock) has reversed_channels_per_block[0] channels ###
        output_channels = reversed_channels_per_block[0]
        for i, block_type in enumerate(config.up_block_types):

            ### Channels of the previous output (starting with the output of the midblock) ###
            prev_output_channel = output_channels

            ### Update the output channels (for every convolution in the block) to be whatever is blocks num_channels ###
            output_channels = reversed_channels_per_block[i]

            ### Compute the input channel (coming from the residual connections of the next block) ###
            ### For example, if we are on the 4th block now (1st when reversed) the input_channel will store
            ### the number of channels of the 3rd block (2nd reversed)
            ### Remember, we repeat the first channels_per_block twice in the encoder ###
            ### So we repeat it twice again in the decoder (just at the end instead of the beginning) ### 
            input_channel = reversed_channels_per_block[min(i+1, len(reversed_channels_per_block)-1)]

            ### Check if this is the final block in our stack of blocks (no upsample on the last one) ###
            is_final_block = (i == len(reversed_channels_per_block) - 1)

            dec_use_attention = (block_type == "AttnUp")

            block = UpBlock2D(in_channels=input_channel, 
                              out_channels=output_channels,
                              prev_out_channels=prev_output_channel, 
                              attention=dec_use_attention,
                              num_attention_heads=output_channels//config.attention_head_dim,
                              cross_attention_dim=config.text_embed_dim if config.text_conditioning else None,
                              time_embed_dim=config.time_embed_proj_dim, 
                              class_conditioning=config.class_conditioning,
                              class_embed_dim=config.class_embed_dim,
                              dropout=config.dropout,  
                              num_layers=config.unet_residual_layers_per_block + 1, 
                              transformers_per_layer=config.transformer_blocks_per_layer, 
                              transformer_dim_mult=config.transformer_dim_mult,
                              attention_bias=config.attention_bias,
                              norm_eps=config.norm_eps, 
                              groupnorm_groups=config.groupnorm_groups, 
                              add_upsample=not is_final_block,
                              upsample_factor=config.unet_up_down_factor,
                              upsample_kernel_size=config.unet_up_down_kernel_size)

            self.up_blocks.append(block)

        ### Final Output Convolution ###
        self.norm_out = nn.GroupNorm(num_groups=config.groupnorm_groups, 
                                     num_channels=config.unet_channels_per_block[0],
                                     eps=config.norm_eps)
        
        self.conv_out = nn.Conv2d(config.unet_channels_per_block[0], 
                                  config.latent_channels, 
                                  kernel_size=3,
                                  stride=1, 
                                  padding="same")


    def forward(self, 
                x, 
                time_embed, 
                text_conditioning=None, 
                text_attention_mask=None,
                class_conditioning=None):
        
        ### Store the Residuals from Every Layers ###
        residuals = ()

        ### Pass through Input Convolution and Store Skip ###
        x = self.conv_in(x)
        residuals += (x,)
        
        ### Pass Through Encoder Blocks ###
        for block in self.down_blocks:
            x, skip_connection = block(x, 
                                       time_embed, 
                                       text_conditioning, 
                                       text_attention_mask,
                                       class_conditioning)
            
            residuals += tuple(skip_connection)

        ### Pass Through Mid Block (no skip connection) ###
        x = self.mid_block(x, 
                           time_embed, 
                           text_conditioning, 
                           text_attention_mask,
                           class_conditioning)

        ### Reverse Skip Connections for Decoder ###
        residuals = list(reversed(residuals))

        ### Pass through Decoder Blocks ###
        for idx, block in enumerate(self.up_blocks):
            
            start_idx = idx*(self.config.unet_residual_layers_per_block+1)
            end_idx = start_idx + self.config.unet_residual_layers_per_block+1
            selected_skips = residuals[start_idx:end_idx]

            x = block(x, 
                      selected_skips, 
                      time_embed, 
                      text_conditioning, 
                      text_attention_mask,
                      class_conditioning)

        ### Final Convolutional Output ###
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)

        return x
