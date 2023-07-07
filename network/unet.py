# Implementation based on https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
import math 

import torch
from torch import nn, Tensor


class SinusoidalPositionalEncoding(nn.Module):
    """Apply the positional encoding s.t. the UNet knows the at which time 
    step we are (arXiv:2006.11239). This is done by giving as input the noise
    levels, since different noise levels correspond to different time steps.
    
    > Parameters are shared across time, which is specified to the network 
    using the Transformer sinusoidal position embedding [...]
    """
    def __init__(self, dim: int) -> None:
        """Initialize class.
        
        Args:  
            dim: dimension of the embedding 
        
        Returns: 
            None 
        """
        super().__init__()
        self.dim = dim 

    def forward(self, noise_levels: Tensor) -> Tensor:
        """Forward pass.

        Args: 
            noise_levels: Input tensor that is being embedded in shape 
                `(batch_size)` containing all noise levels for the batch.
        """
        assert noise_levels.ndim == 1, "Input tensor must be 2D"
        
        device = noise_levels.device
        half_dim = self.dim // 2 # not so important
        emb = torch.exp(
            torch.arange(half_dim, device) * - (math.log(10000) / (half_dim - 1))
        ) # `1/10000^(2i/d)` where `i` is the dimension and `d` the embedding dimension

        # shape `(batch_size, half_dim)`, first row is `noise levels[0] * embeddings`, second 
        # row is `noise levels[1] * embeddings`, etc.
        emb = noise_levels.unsqueeze(dim=1) * emb.unsqueeze(dim=0)

        # To get the sine and cosine embeddings interleaved:  
        emb = torch.statck((torch.sin(emb), torch.cos(emb)), dim=2).view(-1, 2 * half_dim) # shape `(batch_size, dim)`
        
        return emb


class DoubleConv(nn.Module):
    """Apply (Conv2D, BatchNorm2D, Relu) * 2."""

    def __init__(
        self,
        in_channels: int,
        mid_channels: int, 
        num_groups: int,
        out_channels: int,
        kernel_size: tuple = (3, 3),
        padding: tuple = (1, 1), 
        padding_mode: str = "zeros",
        stride: tuple = (1, 1), 
    ) -> None:
        """Initialize DoubleConv. DDPMs replace batch normalization by group
        # normalization (arXiv:1803.08494):

        > GN divides the channels into groups and computes within each
        group the mean and variance for normalization. GN's computation
        is independent of batch sizes, and its accuracy is stable in a
        wide range of batch sizes.

        Args:
            in_channels: number of input channels
            mid_channels: number of channels in the middle Conv2D layers            
            num_groups: number of groups for GroupNorm
            out_channels: number of output channels
            kernel_size: kernel size for Conv2D
            padding: padding for Conv2D
            padding_mode: padding mode for Conv2D
            stride: stride for Conv2D

        Returns:
            None
        """
        super().__init__()

        self.double_conv = nn.Sequential()

        for _ in range(2):
            self.double_conv = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=mid_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                    stride=stride,
                ),
                nn.SiLU(inplace=True),
                nn.GroupNorm(
                    num_groups=num_groups,
                    num_channels=mid_channels,
                    eps=1e-5,
                    affine=True,
                ),
                nn.Conv2d(
                    in_channels=mid_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    padding_mode=padding_mode,
                    stride=stride,
                ),
                nn.SiLU(inplace=True),
                nn.GroupNorm(
                    num_groups=num_groups,
                    num_channels=mid_channels,
                    eps=1e-5,
                    affine=True,
                ),
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input tensor

        Returns:
            Output of forward pass
        """
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize class.

        Args:
            in_channels: number of input channels
            out_channels: number of output channels

        Returns:
            None
        """
        super().__init__()
        double_conv_block = (
            DoubleConv(
                in_channels=in_channels,
                mid_channels=out_channels,
                num_groups=1,
                out_channels=out_channels,
            ).double_conv,
        )
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
        )
        for idx in range(len(double_conv_block)):
            self.maxpool_conv.append(double_conv_block[idx])

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: input tensor

        Returns:
            Output of forward pass
        """
        return self.maxpool_conv(x)


class UNet(nn.Module):
    def __init__():
        pass
