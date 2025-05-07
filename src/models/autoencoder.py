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

# def _print_stats(tensor_name: str, tensor: torch.Tensor):
#     if tensor is None:
#         print(f"Tensor {tensor_name} is None")
#         return
    
#     # These produce Python booleans, let's rename for clarity inside this function
#     py_has_nan = torch.isnan(tensor).any().item()
#     py_has_inf = torch.isinf(tensor).any().item()
    
#     num_elements = tensor.numel()
    
#     # Use the Python booleans (py_has_nan, py_has_inf) in the conditions
#     min_val = 'N/A'
#     max_val = 'N/A'
#     mean_val = 'N/A'
#     std_val = 'N/A'

#     if num_elements > 0 and not py_has_nan and not py_has_inf:
#         min_val = tensor.min().item()
#         max_val = tensor.max().item()
#         mean_val = tensor.mean().item()
#         if num_elements == 1:
#             std_val = 0.0 # Standard deviation of a single point is 0
#         elif num_elements > 1:
#             std_val = tensor.std().item()
#         # if num_elements is 0, std_val remains 'N/A' which is fine
            
#     print(f"  {tensor_name} - isnan: {py_has_nan}, isinf: {py_has_inf}, min: {min_val}, max: {max_val}, mean: {mean_val}, std: {std_val}, shape: {tensor.shape}")
#     if py_has_nan or py_has_inf:
#         print(f"  !!! Problem detected in {tensor_name} !!!")

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

    # def forward(self, x: torch.Tensor):
    #     device = next(self.parameters()).device
    #     dtype = next(self.parameters()).dtype
    #     x = x.to(device=device, dtype=dtype)

    #     z = self.encoder(x)
    #     x_recon = self.decoder(z)
    #     return x_recon, z
    
    def forward(self, x: torch.Tensor):
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype # Should be torch.float32
        # x_orig_shape = x.shape
        x = x.to(device=device, dtype=dtype)

        # print(f"\n--- Autoencoder Forward Pass (Input shape: {x_orig_shape}) ---")
        # _print_stats("AE Input (x)", x)

        # Debug Encoder
        temp_z = x
        # print("-- Encoder Layers --")
        for i, layer in enumerate(self.encoder):
            layer_name = f"Encoder Layer {i} ({layer.__class__.__name__})"
            # if hasattr(layer, 'weight') and layer.weight is not None:
                #  _print_stats(f"{layer_name} weights", layer.weight.data)
            # if hasattr(layer, 'bias') and layer.bias is not None:
                #  _print_stats(f"{layer_name} bias", layer.bias.data)

            # temp_z_before_layer = temp_z
            try:
                temp_z = layer(temp_z)
            except Exception as e:
                print(f"  !!! ERROR during {layer_name}: {e} !!!")
                # _print_stats("Input to failing layer", temp_z_before_layer)
                raise e
            # _print_stats(layer_name, temp_z)
            if torch.isnan(temp_z).any() or torch.isinf(temp_z).any():
                print(f"  !!! Problem detected after {layer_name}. Stopping further AE processing. !!!")
                # To preserve z for return if needed, even if problematic:
                # For example, you could return x, x here if z becomes NaN to see if it helps isolate
                # but for now, let it flow to see decoder issues too.
                # return temp_z, temp_z # Or (x,x) if z is bad, to debug if x was issue

        z = temp_z
        # _print_stats("AE Latent (z)", z)

        if torch.isnan(z).any() or torch.isinf(z).any():
            print("  !!! Latent representation z contains NaN/Inf. Decoder will likely fail. !!!")

        # Debug Decoder
        temp_recon = z
        # print("-- Decoder Layers --")
        for i, layer in enumerate(self.decoder):
            layer_name = f"Decoder Layer {i} ({layer.__class__.__name__})"
            # if hasattr(layer, 'weight') and layer.weight is not None:
            #      _print_stats(f"{layer_name} weights", layer.weight.data)
            # if hasattr(layer, 'bias') and layer.bias is not None:
            #      _print_stats(f"{layer_name} bias", layer.bias.data)

            # temp_recon_before_layer = temp_recon
            try:
                temp_recon = layer(temp_recon)
            except Exception as e:
                print(f"  !!! ERROR during {layer_name}: {e} !!!")
                # _print_stats("Input to failing layer", temp_recon_before_layer)
                raise e
            # _print_stats(layer_name, temp_recon)
            # if torch.isnan(temp_recon).any() or torch.isinf(temp_recon).any():
            #     print(f"  !!! Problem detected after {layer_name}. Stopping further AE processing. !!!")
            #     # return temp_recon, z # Or some other placeholder if recon is bad

        x_recon = temp_recon
        # _print_stats("AE Recon (x_recon)", x_recon)
        # print("--- End Autoencoder Forward Pass ---")
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
