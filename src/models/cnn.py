import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """ Convolutional block with skip connection
    """
    def __init__(self, hidden_features, kernel_size):
        super().__init__()

        self.kernel_size = kernel_size
        self.conv_block = nn.Sequential(
            nn.Conv1d(hidden_features, hidden_features, kernel_size, padding='same'),
            nn.BatchNorm1d(num_features=hidden_features),
            nn.LeakyReLU(negative_slope=0.01),

            nn.Conv1d(hidden_features, hidden_features, kernel_size, padding='same'),
            nn.BatchNorm1d(num_features=hidden_features),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.max_pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x):
        x = x + self.conv_block(x)
        x = self.max_pool(x)
        return x


class Conv1dModel(nn.Module):
    def __init__(self, input_channels, num_conv_blocks, hidden_features, kernel_size, num_classes):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, hidden_features, kernel_size, padding='same'),
            nn.BatchNorm1d(num_features=hidden_features),
            nn.LeakyReLU(negative_slope=0.01)
        )

        self.conv_block_list = nn.ModuleList([
            ConvBlock(hidden_features, kernel_size) for _ in range(num_conv_blocks)
        ])

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.fc = nn.Linear(in_features=hidden_features, out_features=num_classes)

    def forward(self, x):
        x = self.conv1(x)

        for conv_block in self.conv_block_list:
            x = conv_block(x)

        x = self.avgpool(x)
        B = x.shape[0]
        x = self.fc(x.view(B, -1))

        x = F.softmax(x, dim=1)
        return x
