import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Critic(nn.Module):
    def __init__(self, seq_size, num_filters=64):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv1d(1, num_filters, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),

            spectral_norm(nn.Conv1d(num_filters, num_filters * 2, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            spectral_norm(nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),

            spectral_norm(nn.Conv1d(num_filters * 4, num_filters * 8, kernel_size=5, stride=2, padding=2)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Global average pooling instead of flattening
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.output_layer = nn.Linear(num_filters * 8, 1)

    def forward(self, x):
        x = self.net(x)                       # Shape: [batch_size, num_filters*8, T]
        x = self.global_pool(x).squeeze(-1)   # Shape: [batch_size, num_filters*8]
        out = self.output_layer(x)            # Shape: [batch_size, 1]
        return out


class Discriminator(nn.Module):
    def __init__(self, seq_size, num_filters=64):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=num_filters, kernel_size=4, stride=2, padding=1)
        self.leakyRelu = nn.LeakyReLU(0.2)

        self.conv2 = nn.Conv1d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm1d(num_filters*2)

        self.conv3 = nn.Conv1d(in_channels=num_filters*2, out_channels=num_filters*4, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(num_filters*4)

        self.conv4 = nn.Conv1d(in_channels=num_filters*4, out_channels=num_filters*8, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(num_filters*8)

        self.conv5 = nn.Conv1d(in_channels=num_filters*8, out_channels=num_filters*8, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm1d(num_filters*8)

        self.conv6 = nn.Conv1d(in_channels=num_filters*8, out_channels=1, kernel_size=4, stride=2, padding=1)
        self.op = nn.Linear(19,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyRelu(x)

        x = self.conv2(x)
        x = self.bn1(x)
        x = self.leakyRelu(x)

        x = self.conv3(x)
        x = self.bn2(x)
        x = self.leakyRelu(x)

        x = self.conv4(x)
        x = self.bn3(x)
        x = self.leakyRelu(x)

        x = self.conv5(x)
        x = self.bn4(x)
        x = self.leakyRelu(x)

        x = self.conv6(x)
        x = self.leakyRelu(x)
        x = self.op(x)
        x = self.sigmoid(x)
        print(x)
        return x.squeeze(2)
    

class SimpleCritic(nn.Module):
    def __init__(self, seq_size):
        super(SimpleCritic, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear((seq_size // 4) * 64, 1)
        )

    def forward(self, x):
        return self.model(x)
    

class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_filters=32):
        super(Generator, self).__init__()
        self.enc1 = nn.Sequential(
            nn.Conv1d(in_channels, num_filters, 4, 2, 1),  # 500 → 250
            nn.LeakyReLU(0.2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv1d(num_filters, num_filters * 2, 4, 2, 1),  # 250 → 125
            nn.BatchNorm1d(num_filters * 2),
            nn.LeakyReLU(0.2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv1d(num_filters * 2, num_filters * 4, 4, 2, 1),  # 125 → 63
            nn.BatchNorm1d(num_filters * 4),
            nn.LeakyReLU(0.2)
        )
        self.enc4 = nn.Sequential(
            nn.Conv1d(num_filters * 4, num_filters * 8, 4, 2, 1),  # 63 → 32
            nn.BatchNorm1d(num_filters * 8),
            nn.LeakyReLU(0.2)
        )
        self.enc5 = nn.Sequential(
            nn.Conv1d(num_filters * 8, num_filters * 8, 4, 2, 1),  # 32 → 16
            nn.BatchNorm1d(num_filters * 8),
            nn.LeakyReLU(0.2)
        )
        self.enc6 = nn.Sequential(
            nn.Conv1d(num_filters * 8, num_filters * 8, 4, 2, 1),  # 16 → 8
            nn.BatchNorm1d(num_filters * 8),
            nn.LeakyReLU(0.2)
        )

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose1d(num_filters * 8, num_filters * 8, 4, 2, 1),  # 8 → 16
            nn.BatchNorm1d(num_filters * 8),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose1d(num_filters * 8, num_filters * 8, 4, 2, 1),  # 16 → 32
            nn.BatchNorm1d(num_filters * 8),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose1d(num_filters * 8, num_filters * 4, 4, 2, 1),  # 32 → 63
            nn.BatchNorm1d(num_filters * 4),
            nn.ReLU()
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose1d(num_filters * 4, num_filters * 2, 4, 2, 1, output_padding=1),  # 63 → 125
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU()
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose1d(num_filters * 2, num_filters, 4, 2, 1),  # 125 → 250
            nn.BatchNorm1d(num_filters),
            nn.ReLU()
        )
        self.dec6 = nn.ConvTranspose1d(num_filters, out_channels, 4, 2, 1)  # 250 → 500
        self.final_activation = nn.Tanh()

    def match_size(self,x, target):
        diff = x.size(2) - target.size(2)
        if diff > 0:
            return x[:, :, :-diff]
        elif diff < 0:
            return nn.functional.pad(x, (0, -diff))
        else:
            return x
    def forward(self, x):
        x = x.unsqueeze(1)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)

        d1 = self.match_size(self.dec1(e6), e5) + e5
        d2 = self.match_size(self.dec2(d1), e4) + e4
        d3 = self.match_size(self.dec3(d2), e3) + e3
        d4 = self.match_size(self.dec4(d3), e2) + e2
        d5 = self.match_size(self.dec5(d4), e1) + e1
        d6 = self.dec6(d5)

        return self.final_activation(d6)
    

class SimpleGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_filters=32):
        super(SimpleGenerator, self).__init__()

       # Encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, num_filters, kernel_size=4, stride=2, padding=1),  # 1 → 64
            nn.LeakyReLU(0.2),
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1),  # 64 → 128
            nn.BatchNorm1d(num_filters * 2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1),  # 128 → 256
            nn.BatchNorm1d(num_filters * 4),
            nn.LeakyReLU(0.2)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(num_filters * 4, num_filters * 2, kernel_size=4, stride=2, padding=1),  # 256 → 128
            nn.BatchNorm1d(num_filters * 2),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters * 2, num_filters, kernel_size=4, stride=2, padding=1),  # 128 → 64
            nn.BatchNorm1d(num_filters),
            nn.ReLU(),
            nn.ConvTranspose1d(num_filters, out_channels, kernel_size=4, stride=2, padding=1),  # 64 → 1
            nn.Tanh()
        )

    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x  # Ensure input is [B, C, L]
        x = self.encoder(x)
        x = self.decoder(x)
        return x

