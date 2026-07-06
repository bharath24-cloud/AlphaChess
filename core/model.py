import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        x += residual
        x = F.relu(x)

        return x


class AlphaChessNet(nn.Module):
    """
    AlphaZero-style Policy + Value Network
    Input: 19 x 8 x 8 board tensor
    Outputs:
        policy -> move probabilities
        value -> position evaluation (-1 to 1)
    """

    def __init__(
        self,
        input_channels=19,
        channels=256,
        residual_blocks=10,
        policy_size=4672
    ):
        super().__init__()

        self.input_conv = nn.Conv2d(
            input_channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.input_bn = nn.BatchNorm2d(channels)

        self.res_blocks = nn.Sequential(
            *[
                ResidualBlock(channels)
                for _ in range(residual_blocks)
            ]
        )

        # Policy Head
        self.policy_conv = nn.Conv2d(
            channels,
            2,
            kernel_size=1,
            bias=False
        )
        self.policy_bn = nn.BatchNorm2d(2)

        self.policy_fc = nn.Linear(
            2 * 8 * 8,
            policy_size
        )

        # Value Head
        self.value_conv = nn.Conv2d(
            channels,
            1,
            kernel_size=1,
            bias=False
        )
        self.value_bn = nn.BatchNorm2d(1)

        self.value_fc1 = nn.Linear(
            8 * 8,
            256
        )
        self.value_fc2 = nn.Linear(
            256,
            1
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x)

        x = self.res_blocks(x)

        # Policy
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)

        policy = policy.view(
            policy.size(0),
            -1
        )

        policy = self.policy_fc(policy)

        # Value
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)

        value = value.view(
            value.size(0),
            -1
        )

        value = F.relu(
            self.value_fc1(value)
        )

        value = torch.tanh(
            self.value_fc2(value)
        )

        return policy, value


if __name__ == "__main__":
    model = AlphaChessNet()

    sample = torch.randn(
        1,
        19,
        8,
        8
    )

    policy, value = model(sample)

    print("Policy shape:", policy.shape)
    print("Value shape:", value.shape)