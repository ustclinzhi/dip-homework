import torch.nn as nn

class FullyConvNetwork(nn.Module):

    def __init__(self):
        super().__init__()
         # Encoder (Convolutional Layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 3, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        
        ### FILL: add more CONV Layers
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),  # Input channels: 8, Output channels: 16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),  # Input channels: 16, Output channels: 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # Input channels: 32, Output channels: 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        



        # Decoder (Deconvolutional Layers)
        ### FILL: add ConvTranspose Layers
        self.deconv0 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # Input channels: 64, Output channels: 32
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # Input channels: 32, Output channels: 16
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),  # Input channels: 16, Output channels: 8
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True)
        )

        self.deconv3 = nn.ConvTranspose2d(8, 3, kernel_size=4, stride=2, padding=1)  # Input channels: 8, Output channels: 3 (RGB)

        ### None: since last layer outputs RGB channels, may need specific activation function

    def forward(self, x):
        # Encoder forward pass
        x1 = self.conv1(x)    # Output shape: [batch_size, 8, H/2, W/2]
        x2 = self.conv2(x1)   # Output shape: [batch_size, 16, H/4, W/4]
        x3 = self.conv3(x2)   # Output shape: [batch_size, 32, H/8, W/8]
        x4 = self.conv4(x3)   # Output shape: [batch_size, 64, H/16, W/16]
        # Decoder forward pass
        x5 = self.deconv0(x4) # Output shape: [batch_size, 32, H/8, W/8]
        x6 = self.deconv1(x5) # Output shape: [batch_size, 16, H/4, W/4]
        x7 = self.deconv2(x6) # Output shape: [batch_size, 8, H/2, W/2]
     
        output = self.deconv3(x7)  # Output shape: [batch_size, 3, H, W]
    
        return output

    