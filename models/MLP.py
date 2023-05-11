import torch.nn as nn


# Basic MLP network for classification
class MLPbase(nn.Module):
    def __init__(self, config=None, output_dim = 10):
        super(MLPbase, self).__init__()
        if config is None:
            config = [32 * 32 * 3, 256, 128, output_dim]

        # Ensure the number of layers corresponds to the given configuration
        assert len(config) >= 2

        # self.norm = normalize

        # Add linear layers
        layers = []
        for i in range(len(config) - 1):
            layers.append(nn.Linear(config[i], config[i + 1]))
            if i == len(config) - 2:
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.ReLU())

        self.sequential = nn.Sequential(*layers)

    def forward(self, x):
        # if self.norm is not None:
        #     x = self.norm(x)
        x = x.reshape(x.shape[0], -1)
        return self.sequential(x)


def MLP(num_classes=10):
    return MLPbase(config=[32 * 32 * 3, 256, 128, num_classes])
