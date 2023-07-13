import torch


class DepixelationCNN(torch.nn.Module):

    def __init__(self, n_in_channels: int = 1, n_hidden_layers: int = 3, n_kernels: int = 32, kernel_size: int = 3):

        super().__init__()

        cnn = []
        for i in range(n_hidden_layers):
            cnn.append(torch.nn.Conv2d(
                in_channels=n_in_channels,
                out_channels=n_kernels,
                kernel_size=kernel_size,
                padding=kernel_size // 2
            ))
            cnn.append(torch.nn.ReLU())
            n_in_channels = n_kernels
        self.hidden_layers = torch.nn.Sequential(*cnn)

        self.output_layer = torch.nn.Conv2d(
            in_channels=n_in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

    def forward(self, input_images: torch.Tensor):
        return self.output_layer(self.hidden_layers(input_images))
