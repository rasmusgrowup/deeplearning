import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)

        # Final Convolution
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoder
        up4 = self.upconv4(bottleneck)
        cat4 = torch.cat([enc4, up4], dim=1)
        dec4 = self.dec4(cat4)

        up3 = self.upconv3(dec4)
        cat3 = torch.cat([enc3, up3], dim=1)
        dec3 = self.dec3(cat3)

        up2 = self.upconv2(dec3)
        cat2 = torch.cat([enc2, up2], dim=1)
        dec2 = self.dec2(cat2)

        up1 = self.upconv1(dec2)
        cat1 = torch.cat([enc1, up1], dim=1)
        dec1 = self.dec1(cat1)

        # Final Output
        return torch.sigmoid(self.final_conv(dec1))

    @staticmethod
    def conv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

# Test the model
if __name__ == "__main__":
    model = UNet(in_channels=3, out_channels=1)
    print(model)

    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 128, 128)  # Batch size 1, RGB image, 128x128
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Expected: [1, 1, 128, 128]