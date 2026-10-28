"""VAE em PyTorch — modelo, perda e laço de treino.

O autograd é permitido neste exercício, mas **as partes específicas do VAE são
escritas à mão**:

* a reparametrização :math:`z = \\mu + \\sigma \\odot \\varepsilon`, em
  :meth:`VAE.reparameterize`;
* a entropia cruzada binária **somada sobre os 784 pixels** e dividida pelo
  tamanho do lote, em :func:`vae_loss`;
* a KL em **forma fechada**, escrita termo a termo em :func:`kl_divergence` —
  não se usa ``torch.distributions.kl_divergence``;
* a amostragem do prior, em :meth:`VAE.sample`.

Nenhuma classe VAE de biblioteca é utilizada: o encoder e o decoder são
``nn.Sequential`` montados aqui.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

IMAGE_DIM = 784  # 28 x 28 pixels, achatados


class VAE(nn.Module):
    """Autoencoder variacional com a arquitetura fixada pelo enunciado.

    * Encoder: ``784 → 512 → 256``, ReLU após cada camada;
    * Cabeças: duas camadas lineares ``256 → d``, uma para :math:`\\mu` e outra
      para :math:`\\log \\sigma^2`;
    * Decoder: ``d → 256 → 512 → 784``, ReLU nas escondidas e sigmoide na saída.
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(IMAGE_DIM, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        # Cabeças separadas: a rede prevê log(sigma^2), não sigma. Prever o log
        # deixa a saída livre em toda a reta real (não há restrição de
        # positividade a impor) e a exponencial devolve uma variância positiva.
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, IMAGE_DIM),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Devolve ``(mu, logvar)`` do posterior aproximado :math:`q(z \\mid x)`."""
        hidden = self.encoder(x)
        return self.fc_mu(hidden), self.fc_logvar(hidden)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Truque da reparametrização: :math:`z = \\mu + \\sigma \\odot \\varepsilon`.

        Amostrar :math:`z \\sim \\mathcal{N}(\\mu, \\sigma^2)` diretamente cortaria
        o grafo — a amostragem não é diferenciável em relação a ``mu`` e
        ``logvar``. Escrevendo :math:`z` como uma função determinística dos
        parâmetros mais um ruído :math:`\\varepsilon \\sim \\mathcal{N}(0, I)`
        **independente deles**, o gradiente volta a fluir para o encoder.

        De ``logvar`` para o desvio: :math:`\\sigma = \\exp(\\tfrac{1}{2}\\log\\sigma^2)`.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decodifica ``z`` para probabilidades de pixel em :math:`[0, 1]`."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    @torch.no_grad()
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """Amostra ``n_samples`` do prior :math:`\\mathcal{N}(0, I)` e decodifica."""
        z = torch.randn(n_samples, self.latent_dim, device=device)
        return self.decode(z)


def reconstruction_bce(x_hat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """BCE **somada sobre os 784 pixels**, somada sobre o lote.

    Quem divide pelo tamanho do lote é o chamador, para obter nats por imagem.
    Usar ``reduction="mean"`` aqui estaria errado: ela também faria a média sobre
    os pixels, encolhendo o termo de reconstrução 784 vezes em relação à KL — o
    mesmo que definir :math:`\\beta \\approx 784` sem perceber.
    """
    return F.binary_cross_entropy(x_hat, x, reduction="sum")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL fechada entre :math:`q(z \\mid x)` e o prior :math:`\\mathcal{N}(0, I)`.

    .. math::
        D_{KL} = \\tfrac{1}{2}\\sum_{i=1}^{d}
                 \\left(\\mu_i^2 + \\sigma_i^2 - 1 - \\log \\sigma_i^2\\right)

    Escrita termo a termo, com :math:`\\sigma^2 = \\exp(\\log\\sigma^2)`. Somada
    sobre as dimensões latentes e sobre o lote; o chamador divide pelo tamanho do
    lote.
    """
    return 0.5 * torch.sum(mu.pow(2) + logvar.exp() - 1.0 - logvar)


def vae_loss(
    x_hat: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, beta: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Perda do VAE. Devolve ``(total, bce, kl)``, **em nats por imagem**."""
    batch_size = x.shape[0]
    bce = reconstruction_bce(x_hat, x) / batch_size
    kl = kl_divergence(mu, logvar) / batch_size
    return bce + beta * kl, bce, kl


@torch.no_grad()
def evaluate(model: VAE, loader: Any, device: torch.device, beta: float = 1.0) -> dict[str, float]:
    """Média de BCE e KL sobre um conjunto inteiro, em nats por imagem."""
    model.eval()
    totals = {"bce": 0.0, "kl": 0.0}
    n_images = 0
    for images, _ in loader:
        x = images.view(images.shape[0], -1).to(device)
        x_hat, mu, logvar = model(x)
        # Reacumula somas e divide no fim: assim o último lote, possivelmente
        # menor, não pesa igual aos demais.
        totals["bce"] += reconstruction_bce(x_hat, x).item()
        totals["kl"] += kl_divergence(mu, logvar).item()
        n_images += x.shape[0]
    bce, kl = totals["bce"] / n_images, totals["kl"] / n_images
    return {"bce": bce, "kl": kl, "neg_elbo": bce + beta * kl}


def train_epoch(
    model: VAE, loader: Any, optimizer: torch.optim.Optimizer, device: torch.device, beta: float = 1.0
) -> dict[str, float]:
    """Uma época de treino. Devolve as médias de BCE e KL da própria época."""
    model.train()
    totals = {"bce": 0.0, "kl": 0.0}
    n_images = 0
    for images, _ in loader:
        x = images.view(images.shape[0], -1).to(device)
        optimizer.zero_grad(set_to_none=True)
        x_hat, mu, logvar = model(x)
        loss, bce, kl = vae_loss(x_hat, x, mu, logvar, beta)
        loss.backward()
        optimizer.step()
        totals["bce"] += bce.item() * x.shape[0]
        totals["kl"] += kl.item() * x.shape[0]
        n_images += x.shape[0]
    bce, kl = totals["bce"] / n_images, totals["kl"] / n_images
    return {"bce": bce, "kl": kl, "neg_elbo": bce + beta * kl}
