import torch
import torch.nn as nn

class ImageEncoder(nn.Module):
    def __init__(self,input_channels = 3,output_dim = 128):
        super().__init__()
        self.cnn_encoder =nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)) # reduce spatial dimensions to 1x1
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.cnn_encoder(x)
        x = torch.flatten(x, start_dim=1) # this flattens ( B, 128, 1, 1) to (B, 128)
        x = self.fc(x)
        return x
