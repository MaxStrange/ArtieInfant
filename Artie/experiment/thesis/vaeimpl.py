"""
PyTorch implementation of a (potentially variational) autoencoder.
"""
import torch

class AutoEncoder(torch.nn.Module):
    """
    Basic autoencoder.
    """
    def __init__(self, input_shape: (int)):
        super(AutoEncoder, self).__init__()

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 8, (8, 2), stride=(2, 1), padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(8)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(8, 16, (8, 2), stride=(2, 1), padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm1d(16)
        )
