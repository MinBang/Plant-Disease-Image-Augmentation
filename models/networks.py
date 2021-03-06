from torch import nn
import torch.nn.functional as F
import numpy as np
import math
import functools

#from cbam import CBAM
from .cbam import *

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

##########
class ResBlk(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False, upsample=False, attention=False, affine=False, reduction_ratio=16):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.upsample = upsample
        self.learned_sc = dim_in != dim_out
        self.attention = attention
        self.reduction_ratio = reduction_ratio
        self.affine = affine
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.Conv2d(dim_in, dim_in, 3, 1, 1)
        self.conv2 = nn.Conv2d(dim_in, dim_out, 3, 1, 1)
        if self.normalize:
            self.norm1 = nn.InstanceNorm2d(dim_in, affine=self.affine)
            self.norm2 = nn.InstanceNorm2d(dim_in, affine=self.affine)
        if self.learned_sc:
            self.conv1x1 = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False)
        if self.attention:
            self.cbam = CBAM(dim_in, reduction_ratio=self.reduction_ratio, no_spatial=False)

    def _shortcut(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)        
        if self.attention:
            x = self.cbam(x)

        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        return x / math.sqrt(2)  # unit variance
        #return x

class ResBlkSN(nn.Module):
    def __init__(self, dim_in, dim_out, actv=nn.LeakyReLU(0.2),
                 normalize=False, downsample=False, attention=False, affine=False, reduction_ratio=16):
        super().__init__()
        self.actv = actv
        self.normalize = normalize
        self.downsample = downsample
        self.learned_sc = dim_in != dim_out
        self.attention = attention
        self.reduction_ratio = reduction_ratio
        self.affine = affine
        self._build_weights(dim_in, dim_out)

    def _build_weights(self, dim_in, dim_out):
        self.conv1 = nn.utils.spectral_norm(nn.Conv2d(dim_in, dim_in, 3, 1, 1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv2d(dim_in, dim_out, 3, 1, 1))
        if self.normalize:
            self.norm1 = nn.BatchNorm2d(dim_in)
            self.norm2 = nn.BatchNorm2d(dim_in)
        if self.learned_sc:
            self.conv1x1 = nn.utils.spectral_norm(nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=False))
        if self.attention:
            self.cbam = CBAM(dim_in, reduction_ratio=self.reduction_ratio, no_spatial=False)

    def _shortcut(self, x):
        if self.learned_sc:
            x = self.conv1x1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        return x

    def _residual(self, x):
        if self.normalize:
            x = self.norm1(x)
        x = self.actv(x)
        x = self.conv1(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        if self.normalize:
            x = self.norm2(x)
        x = self.actv(x)
        x = self.conv2(x)
        if self.attention:
            x = self.cbam(x)
        
        return x

    def forward(self, x):
        x = self._shortcut(x) + self._residual(x)
        #return x / math.sqrt(2)  # unit variance
        return x

class ResnetGenerator_bang(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=6, n_downsampling=3, max_conv_dim=512, attention=True, affine=False, output_tanh=False, reduction_ratio=16):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator_bang, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.affine = affine

        entry = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)]

        to_rgb = [nn.InstanceNorm2d(ngf, affine=self.affine),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(ngf, output_nc, 1, 1, 0),
                ]

        if output_tanh: to_rgb += [nn.Tanh()]
        
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        bottleneck = []
        dim_in = ngf

        for i in range(n_downsampling):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(nn.Sequential(*[norm_layer(dim_in, affine=self.affine),
                                            nn.LeakyReLU(0.2),
                                            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=2, padding=1, bias=use_bias)]))

            self.decode.insert(0, nn.Sequential(*[norm_layer(dim_out, affine=self.affine),
                                            nn.LeakyReLU(0.2),
                                            nn.ConvTranspose2d(dim_out, dim_in, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)]))
            dim_in = dim_out

        for i in range(n_blocks):       # add ResNet blocks
            bottleneck += [ResBlk(dim_out, dim_out, normalize=True, attention=attention, affine=self.affine, reduction_ratio=reduction_ratio)]

        self.entry = nn.Sequential(*entry)
        self.bottleneck = nn.Sequential(*bottleneck)
        self.to_rgb = nn.Sequential(*to_rgb)

    def forward(self, x):
        x = self.entry(x)
        #"""Standard forward"""
        for block in self.encode:
            x = block(x)

        x = self.bottleneck(x)

        for block in self.decode:
            x = block(x)
        
        x = self.to_rgb(x)
        return x

class ResnetGenerator_Att(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, n_blocks=6, n_downsampling=3, max_conv_dim=512, attention=True, affine=False, output_tanh=False):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator_Att, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.affine = affine

        entry = [nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)]

        to_rgb = [nn.InstanceNorm2d(ngf, affine=self.affine),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(ngf, output_nc, 1, 1, 0),
                ]

        to_att = [nn.InstanceNorm2d(ngf, affine=self.affine),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(ngf, 1, 1, 1, 0),
                    nn.Sigmoid()
                ]

        if output_tanh: to_rgb += [nn.Tanh()]
        
        self.encode = nn.ModuleList()
        self.decode = nn.ModuleList()
        bottleneck = []
        dim_in = ngf

        for i in range(n_downsampling):
            dim_out = min(dim_in*2, max_conv_dim)
            self.encode.append(nn.Sequential(*[norm_layer(dim_in, affine=self.affine),
                                            nn.LeakyReLU(0.2),
                                            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=2, padding=1, bias=use_bias)]))

            self.decode.insert(0, nn.Sequential(*[norm_layer(dim_out, affine=self.affine),
                                            nn.LeakyReLU(0.2),
                                            nn.ConvTranspose2d(dim_out, dim_in, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)]))
            dim_in = dim_out

        for i in range(n_blocks):       # add ResNet blocks
            bottleneck += [ResBlk(dim_out, dim_out, normalize=True, attention=attention, affine=self.affine)]

        self.entry = nn.Sequential(*entry)
        self.bottleneck = nn.Sequential(*bottleneck)
        self.to_rgb = nn.Sequential(*to_rgb)
        self.to_att = nn.Sequential(*to_att)

    def forward(self, input):
        x = self.entry(input)
        #"""Standard forward"""
        for block in self.encode:
            x = block(x)

        x = self.bottleneck(x)

        for block in self.decode:
            x = block(x)
        
        #output = self.to_rgb(x)

        #attention_mask = F.sigmoid(output[:, :1])
        #content_mask = output[:, 1:]
        attention_mask = self.to_att(x)
        content_mask = self.to_rgb(x)

        attention_mask = attention_mask.repeat(1, 3, 1, 1)
        result = (content_mask * attention_mask) + (input * (1 - attention_mask))

        return result, attention_mask, content_mask

class NLayerDiscriminatorSpec(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
        """
        super(NLayerDiscriminatorSpec, self).__init__()

        kw = 4
        padw = 1
        sequence = [nn.utils.spectral_norm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)), nn.LeakyReLU(0.2)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.utils.spectral_norm(
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw)),
                nn.LeakyReLU(0.2)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.utils.spectral_norm(
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw)),
            nn.LeakyReLU(0.2)
        ]

        sequence += [nn.utils.spectral_norm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

if __name__ == '__main__':
    g = ResnetGenerator_bang(3, 3).cuda()
    #d = NLayerDiscriminatorSpec(3)
    #d = Discriminator()
    
    from torchinfo import summary
    summary(g, (1, 3, 256, 256))