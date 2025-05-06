"""
Module Name: autoencoder.py
Description: A configurable MLP Autoencoder whose encoder/decoder depths
             and hidden sizes are specified at run‑time.
Author: Henry Huang  • Last updated: 2025‑05‑05
"""

from typing import Sequence, Iterable, Type
import torch
import torch.nn as nn


def _build_mlp(
    dims: Iterable[int],
    act_cls: Type[nn.Module]
) -> nn.Sequential:
    """
    dims : e.g. [in, 256, 128, 64]  → Linear(in→256)+Act → Linear(256→128)+Act …
           The **last** Linear has **no** activation.
    """
    layers: list[nn.Module] = []
    it = iter(dims)
    prev = next(it)
    for d in it:
        layers.append(nn.Linear(prev, d))
        prev = d
        # no activation after the very last Linear
        if prev != dims[-1]:
            layers.append(act_cls())
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    """
    Args
    ----
    input_dim  : dimensionality of each KV vector.
    latent_dim : size of compressed representation.
    encoder_layer_sizes : Sequence[int] – hidden sizes *between*
                          input_dim and latent_dim **not including either**.
                          [] or None  ⇒ single Linear(input_dim→latent_dim)
    decoder_layer_sizes : Sequence[int] – hidden sizes *between*
                          latent_dim and input_dim **not including either**.
    activation          : str – any torch.nn activation name, default "ReLU".
    dtype               : torch dtype for parameters & forward tensors.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_layer_sizes: Sequence[int] | None = None,
        decoder_layer_sizes: Sequence[int] | None = None,
        activation: str = "ReLU",
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        act_cls: Type[nn.Module] = getattr(nn, activation)

        enc_dims = [input_dim] + list(encoder_layer_sizes or []) + [latent_dim]
        dec_dims = [latent_dim] + list(decoder_layer_sizes or []) + [input_dim]

        self.encoder = _build_mlp(enc_dims, act_cls)
        self.decoder = _build_mlp(dec_dims, act_cls)

        if dtype is not None:
            self.to(dtype)

    # -----------------------------------------------------

    def forward(self, x: torch.Tensor):
        device = next(self.parameters()).device
        dtype  = next(self.parameters()).dtype
        x      = x.to(device=device, dtype=dtype)

        z        = self.encoder(x)
        x_recon  = self.decoder(z)
        return x_recon, z


# ───── quick smoke‑test ─────────────────────────────────
if __name__ == "__main__":
    ae = Autoencoder(
        input_dim=64,
        latent_dim=8,
        encoder_layer_sizes=[128, 32],
        decoder_layer_sizes=[32, 128],
        activation="GELU"
    )
    dummy = torch.randn(4, 64)
    rec, z = ae(dummy)
    print("latent:", z.shape, "recon:", rec.shape)