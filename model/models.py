import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv_layer = nn.Conv2d(input_channels, output_channels, kernel_size)
        self.batch_norm = nn.BatchNorm2d(output_channels)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.feature_extractor = nn.Sequential(
            ConvBlock(1, 8, 9),
            nn.MaxPool2d(2),
            ConvBlock(8, 16, 5),
            nn.MaxPool2d(2),
            ConvBlock(16, 32, 3),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 9),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.mean(dim=(-1,-2))
        out = self.fc(x)
        return out