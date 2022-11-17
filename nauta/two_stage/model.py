from torch import nn

class CNNVesselNetwork(nn.Module):
    """The standard CNN approach to the Underwater
    Classification problem.
    """

    def __init__(self, model_depth=3, out_classes=4, input_channels=3):
        super().__init__()
        self.depth = model_depth

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2
            ),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        linear_dim = 0
        if self.depth == 2:
            self.conv_layers = [self.conv2]
            linear_dim = 32 * 17 * 17
        elif self.depth == 3:
            self.conv_layers = [self.conv2, self.conv3]
            linear_dim = 64 * 9 * 9
        elif self.depth == 4:
            self.conv_layers = [self.conv2, self.conv3, self.conv4]
            linear_dim = 128 * 5 * 5

        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(linear_dim, out_classes, bias=False),
            nn.Dropout(p=0.1),
            nn.LeakyReLU()
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_data):
        
        x = self.conv1(input_data)
        for layer in self.conv_layers:
            x = layer(x)

        logits = self.linear(x)
        predictions = self.softmax(logits)
        return predictions