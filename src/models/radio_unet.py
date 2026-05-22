import torch
import torch.nn as nn

class _ConvRelu(nn.Module):
    """
    Use to increase output dimension with halfing resolution
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(_ConvRelu, self).__init__()
        padding = (kernel_size-1) // 2  # Ensure same convolution; kernel_size need to be odd number
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Half resolution
        )
    
    def forward(self, x):
        return self.block(x)

class _ConvReluT(nn.Module):
    """
    Use to decrease output dimension with double resolution
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(_ConvReluT, self).__init__()
        padding = kernel_size // 2 - 1  # Ensure convolution to double resolution; kernel_size need to be even number
        self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=2, padding=padding),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)

class _ConvReluNoPool(nn.Module):
    """
    Use to increase output dimension with unchanging resolution
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(_ConvReluNoPool, self).__init__()
        padding = (kernel_size-1) // 2  # Ensure same convolution; kernel_size need to be odd number
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.block(x)
    
class RadioUnet(nn.Module):
    """
    The Radio/First U-net
    """
    def __init__(self, in_channels, first_out_channels):
        super().__init__()

        self.in_channels = in_channels

        # ================= Radio/First U-Net (Encoder) =================
        self.layer00 = _ConvReluNoPool(in_channels, first_out_channels, 3)
        self.layer0 = _ConvRelu(first_out_channels, 40, 5)
        self.layer1 = _ConvRelu(40, 50, 5)
        self.layer10 = _ConvReluNoPool(50, 60, 5)
        self.layer2 = _ConvRelu(60, 100, 5)
        self.layer20 = _ConvReluNoPool(100, 100, 3)
        self.layer3 = _ConvRelu(100, 150, 5)
        self.layer4 = _ConvRelu(150, 300, 5)
        self.layer5 = _ConvRelu(300, 500, 5)

        # ================= Radio/First U-Net (Decoder) =================
        self.conv_up5 = _ConvReluT(500, 300, 4)
        self.conv_up4 = _ConvReluT(600, 150, 4)   # skip connect: 300+300=600
        self.conv_up3 = _ConvReluT(300, 100, 4)   # skip connect: 150+150=300
        self.conv_up20 = _ConvReluNoPool(200, 100, 3) # skip connect: 100+100=200
        self.conv_up2 = _ConvReluT(200, 60, 6)    # skip connect: 100+100=200
        self.conv_up10 = _ConvReluNoPool(120, 50, 5) # skip connect: 60+60=120
        self.conv_up1 = _ConvReluT(100, 40, 6)    # skip connect: 50+50=100
        self.conv_up0 = _ConvReluT(80, 20, 6)     # skip connect: 40+40=80
        self.conv_up00 = _ConvReluNoPool(20 + first_out_channels + in_channels, 20, 5)  # skip connect
        self.conv_up000 = _ConvReluNoPool(20 + in_channels, 1, 5)  # skip connect
    
    def forward(self, x):
        # --- Radio/First U-Net Forward ---
        e00 = self.layer00(x)
        e0 = self.layer0(e00)
        e1 = self.layer1(e0)
        e10 = self.layer10(e1)
        e2 = self.layer2(e10)
        e20 = self.layer20(e2)
        e3 = self.layer3(e20)
        e4 = self.layer4(e3)
        e5 = self.layer5(e4)

        d5 = self.conv_up5(e5)
        d4 = self.conv_up4(torch.cat([d5, e4], dim=1))
        d3 = self.conv_up3(torch.cat([d4, e3], dim=1))
        d20 = self.conv_up20(torch.cat([d3, e20], dim=1))
        d2 = self.conv_up2(torch.cat([d20, e2], dim=1))
        d10 = self.conv_up10(torch.cat([d2, e10], dim=1))
        d1 = self.conv_up1(torch.cat([d10, e1], dim=1))
        d0 = self.conv_up0(torch.cat([d1, e0], dim=1))
        d00 = self.conv_up00(torch.cat([d0, e00, x], dim=1))
        d000 = self.conv_up000(torch.cat([d00, x], dim=1))

        return d000

class _SecondUNet(nn.Module):
    """
    The Append/Second U-net
    """
    def __init__(self, in_channels):
        super().__init__()

        # ================= Append/Second U-Net (Encoder) =================
        self.Wlayer00 = _ConvReluNoPool(1 + in_channels, 20, 3)  # skip connect
        self.Wlayer0 = _ConvRelu(20, 30, 5)
        self.Wlayer1 = _ConvRelu(30, 40, 5)
        self.Wlayer10 = _ConvReluNoPool(40, 50, 5)
        self.Wlayer2 = _ConvRelu(50, 60, 5)
        self.Wlayer20 = _ConvReluNoPool(60, 70, 3)
        self.Wlayer3 = _ConvRelu(70, 90, 5)
        self.Wlayer4 = _ConvRelu(90, 110, 5)
        self.Wlayer5 = _ConvRelu(110, 150, 5)

        # ================= Append/Second U-Net (Decoder) =================
        self.Wconv_up5 = _ConvReluT(150, 110, 4)
        self.Wconv_up4 = _ConvReluT(220, 90, 4)   # skip connect: 110+110=220
        self.Wconv_up3 = _ConvReluT(180, 70, 4)   # skip connect: 90+90=180
        self.Wconv_up20 = _ConvReluNoPool(140, 60, 3) # skip connect: 70+70=140
        self.Wconv_up2 = _ConvReluT(120, 50, 6)   # skip connect: 60+60=120
        self.Wconv_up10 = _ConvReluNoPool(100, 40, 5) # skip connect: 50+50=100
        self.Wconv_up1 = _ConvReluT(80, 30, 6)    # skip connect: 40+40=80
        self.Wconv_up0 = _ConvReluT(60, 20, 6)    # skip connect: 30+30=60
        self.Wconv_up00 = _ConvReluNoPool(20 + 20 + 1 + in_channels, 20, 5) # skip connect
        self.Wconv_up000 = _ConvReluNoPool(20 + 1 + in_channels, 1, 5) # skip connect
    
    def forward(self, d000, x):
        # --- Append/Second U-Net Forward ---
        w_x = torch.cat([d000, x], dim=1)
        
        we00 = self.Wlayer00(w_x)
        we0 = self.Wlayer0(we00)
        we1 = self.Wlayer1(we0)
        we10 = self.Wlayer10(we1)
        we2 = self.Wlayer2(we10)
        we20 = self.Wlayer20(we2)
        we3 = self.Wlayer3(we20)
        we4 = self.Wlayer4(we3)
        we5 = self.Wlayer5(we4)

        wd5 = self.Wconv_up5(we5)
        wd4 = self.Wconv_up4(torch.cat([wd5, we4], dim=1))
        wd3 = self.Wconv_up3(torch.cat([wd4, we3], dim=1))
        wd20 = self.Wconv_up20(torch.cat([wd3, we20], dim=1))
        wd2 = self.Wconv_up2(torch.cat([wd20, we2], dim=1))
        wd10 = self.Wconv_up10(torch.cat([wd2, we10], dim=1))
        wd1 = self.Wconv_up1(torch.cat([wd10, we1], dim=1))
        wd0 = self.Wconv_up0(torch.cat([wd1, we0], dim=1))
        wd00 = self.Wconv_up00(torch.cat([wd0, we00, w_x], dim=1))
        wd000 = self.Wconv_up000(torch.cat([wd00, w_x], dim=1))

        return wd000


class RadioWnet(nn.Module):
    """
    Two-phase cascaded U-Net for radio map prediction; The Radio W-net.
    """
    def __init__(self, in_channels, first_out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.first_unet = RadioUnet(in_channels, first_out_channels)
        self.second_unet = _SecondUNet(in_channels)

        

    def forward(self, x):
        d000 = self.first_unet(x)
        wd000 = self.second_unet(d000, x)

        return wd000