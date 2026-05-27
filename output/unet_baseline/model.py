#!/usr/bin/env python3
import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(),
    )


class BaselineUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_filters=32):
        super().__init__()

        f = base_filters

        # Input: (B, 1, 64, 64)
        # enc1:
        # Conv2d(1 -> 32, 3x3):
        # Conv2d(32 -> 32, 3x3):
        # Output: (B, 32, 64, 64)
        self.enc1 = conv_block(in_channels, f)

        # pool:
        # (B, 32, 64, 64) -> (B, 32, 32, 32)
        # enc2:
        # Conv2d(32 -> 64, 3x3):
        # Conv2d(64 -> 64, 3x3):
        # Output: (B, 64, 32, 32)
        self.enc2 = conv_block(f, 2 * f)

        # pool:
        # (B, 64, 32, 32) -> (B, 64, 16, 16)
        # bottleneck:
        # Conv2d(64 -> 128, 3x3):
        # Conv2d(128 -> 128, 3x3):
        # Output: (B, 128, 16, 16)
        self.bottleneck = conv_block(2 * f, 4 * f)

        self.pool = nn.MaxPool2d(2)

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )

        # bottleneck output after up:
        # (B, 128, 16, 16) -> (B, 128, 32, 32)
        # cat with enc2 skip:
        # (B, 128, 32, 32) + (B, 64, 32, 32) -> (B, 192, 32, 32)
        # dec2:
        # Conv2d(192 -> 64, 3x3):
        # Conv2d(64 -> 64, 3x3):
        # Output: (B, 64, 32, 32)
        self.dec2 = conv_block(4 * f + 2 * f, 2 * f)

        # dec2 output after up:
        # (B, 64, 32, 32) -> (B, 64, 64, 64)
        # cat with enc1 skip:
        # (B, 64, 64, 64) + (B, 32, 64, 64) -> (B, 96, 64, 64)
        # dec1:
        # Conv2d(96 -> 32, 3x3):
        # Conv2d(32 -> 32, 3x3):
        # Output: (B, 32, 64, 64)
        self.dec1 = conv_block(2 * f + f, f)

        # final 1x1 conv
        # Conv2d(32 -> 1, 1x1):
        # Output: (B, 1, 64, 64)
        self.out = nn.Conv2d(f, out_channels, kernel_size=1)


    def forward(self, x):
        # x: (B, 1, 64, 64)

        c1 = self.enc1(x)
        # c1: (B, 32, 64, 64)

        c2 = self.enc2(self.pool(c1))
        # pool(c1): (B, 32, 32, 32)
        # c2:       (B, 64, 32, 32)

        x = self.bottleneck(self.pool(c2))
        # pool(c2): (B, 64, 16, 16)
        # x:        (B, 128, 16, 16)

        x = self.up(x)
        # x: (B, 128, 32, 32)

        x = torch.cat([x, c2], dim=1) # first skip
        # x: (B, 192, 32, 32)

        x = self.dec2(x)
        # x: (B, 64, 32, 32)

        x = self.up(x)
        # x: (B, 64, 64, 64)

        x = torch.cat([x, c1], dim=1) # second skip
        # x: (B, 96, 64, 64)

        x = self.dec1(x)
        # x: (B, 32, 64, 64)

        return self.out(x)
        # output: (B, 1, 64, 64)


def build_model(input_shape=(64, 64, 1), base_filters=32):
    return BaselineUNet(
        in_channels=input_shape[-1],
        out_channels=1,
        base_filters=base_filters,
    )