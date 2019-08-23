"""
This is the PyTorch backend API.
"""
import torch

def seed(s: int):
    """
    Set the random seed.
    """
    torch.manual_seed(s)

def build_autoencoder1(is_variational, input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_prop, recon_loss_prop, std_loss_prop):
    """
    Builds and returns an autoencoder model in PyTorch (241x20x1).
    """
    pass

def build_autoencoder2(is_variational, input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_prop, recon_loss_prop, std_loss_prop):
    """
    Builds and returns an autoencoder model in PyTorch (81x18x1).
    """
    pass

def build_variational_autoencoder1(is_variational, input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_prop, recon_loss_prop, std_loss_prop):
    """
    Builds and returns a variational autoencoder model in PyTorch (241x20x1).
    """
    pass

def build_variational_autoencoder2(is_variational, input_shape, latent_dim, optimizer, loss, tbdir, kl_loss_prop, recon_loss_prop, std_loss_prop):
    """
    Builds and returns a variational autoencoder model in PyTorch (81x18x1).
    """
    pass

def train_autoencoder(autoencoder, visualize: bool, root: str, steps_per_epoch: int, imshapes: [int], batchsize: int, nworkers: int, test: str, nepochs: int):
    """
    Trains the given autoencoder model.
    """
    pass
