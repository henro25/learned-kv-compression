"""
Module Name: autoencoder.py
Description: A configurable MLP Autoencoder with optional normalization layers
             (LayerNorm or BatchNorm) to improve generalization.
Author: Henry Huang
Last updated: 2025-5-07
"""

from typing import Sequence, Iterable, Type, Optional
import torch
import torch.nn as nn


def _build_mlp(
    dims: Iterable[int],
    act_cls: Type[nn.Module],
    norm_cls: Optional[Type[nn.Module]] = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    it = iter(dims)
    prev = next(it)
    for d in it:
        layers.append(nn.Linear(prev, d))
        is_last = (d == dims[-1])
        if not is_last:
            if norm_cls is not None:
                # LayerNorm expects normalized_shape, BatchNorm1d expects num_features
                layers.append(norm_cls(d))
            layers.append(act_cls())
        prev = d
    # Add normalization after the last linear layer
    if norm_cls is not None and len(dims) > 1:
        layers.append(norm_cls(dims[-1]))
    return nn.Sequential(*layers)


class Autoencoder(nn.Module):
    """Configurable MLP Autoencoder.

    Args
    ----
    input_dim           : dimensionality of each KV vector.
    latent_dim          : size of compressed representation.
    encoder_layer_sizes : hidden sizes *between* input_dim and latent_dim.
    decoder_layer_sizes : hidden sizes *between* latent_dim and input_dim.
    activation          : name of activation in ``torch.nn`` (default "ReLU").
    norm_type           : "LayerNorm", "BatchNorm1d", or ``None``. Default "LayerNorm".
    dtype               : torch dtype for parameters & forward tensors.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_layer_sizes: Sequence[int] | None = None,
        decoder_layer_sizes: Sequence[int] | None = None,
        activation: str = "ReLU",
        norm_type: str | None = "LayerNorm",
        dtype: torch.dtype | None = None,
    ):
        super().__init__()

        act_cls: Type[nn.Module] = getattr(nn, activation)
        norm_cls: Optional[Type[nn.Module]] = getattr(nn, norm_type) if norm_type else None

        enc_dims = [input_dim] + list(encoder_layer_sizes or []) + [latent_dim]
        dec_dims = [latent_dim] + list(decoder_layer_sizes or []) + [input_dim]

        self.encoder = _build_mlp(enc_dims, act_cls, norm_cls)
        self.decoder = _build_mlp(dec_dims, act_cls, norm_cls)

        if dtype is not None:
            self.to(dtype)

    # -----------------------------------------------------

    def forward(self, x: torch.Tensor):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        x = x.to(device=device, dtype=dtype)

        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z


# ───── quick smoke‑test ─────────────────────────────────
if __name__ == "__main__":
    ae = Autoencoder(
        input_dim=64,
        latent_dim=8,
        encoder_layer_sizes=[128, 32],
        decoder_layer_sizes=[32, 128],
        activation="GELU",
        norm_type="LayerNorm",
    )
    dummy = torch.randn(4, 64)
    rec, z = ae(dummy)
    print("latent:", z.shape, "recon:", rec.shape)
