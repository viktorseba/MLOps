import torch
import torch.nn as nn


class Encoder(nn.Module):
    """Gaussian MLP Encoder."""

    def __init__(self, input_dim, hidden_dim, latent_dim) -> None:
        super().__init__()

        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.training = True

    def forward(self, x):
        """Forward pass."""
        h_ = torch.relu(self.FC_input(x))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)

        var = torch.exp(0.5 * log_var)
        z = self.reparameterization(mean, var)

        return z, mean, log_var

    def reparameterization(
        self,
        mean,
        var,
    ):
        """Reparameterization trick."""
        epsilon = torch.rand_like(var)
        return mean + var * epsilon


class Decoder(nn.Module):
    """Bernoulli MLP Decoder."""

    def __init__(self, latent_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """Forward pass."""
        h = torch.relu(self.FC_hidden(x))
        return torch.sigmoid(self.FC_output(h))


class Model(nn.Module):
    """VAE Model."""

    def __init__(self, encoder, decoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        """Forward pass."""
        z, mean, log_var = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat, mean, log_var
