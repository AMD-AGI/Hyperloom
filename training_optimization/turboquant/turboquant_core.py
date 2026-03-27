"""
TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate

Paper: Zandieh, Daliri, Hadian, Mirrokni (ICLR 2026) — arXiv:2504.19874

This module implements:
  - Lloyd-Max codebook solver for the Beta coordinate distribution
  - TurboQuantEngine: rotation + per-coordinate quantization + dequantization
  - OutlierAwareTurboQuant: mixed-precision quantization with outlier channel handling

The paper's key-cache strategy (Section 4.3):
  - Split head_dim channels into outlier (high-variance) and non-outlier groups
  - Allocate more bits to outlier channels
  - Use TurboQuant_prod for keys (unbiased inner products via QJL)
  - Use TurboQuant_mse for values (optimal reconstruction)
"""

import torch
import math
from typing import Tuple, Optional, Dict
from scipy import integrate


def _solve_lloyd_max(dim: int, bits: int, n_iter: int = 200,
                     tol: float = 1e-10) -> Tuple[torch.Tensor, torch.Tensor]:
    """Solve for optimal Lloyd-Max codebook for Beta coordinate distribution.

    On the d-dimensional unit sphere, each rotated coordinate follows
    Beta((d-1)/2, (d-1)/2) scaled to [-1,1], which for large d converges
    to N(0, 1/d). We use the Gaussian approximation.

    Returns: (centroids, boundaries) as sorted 1D float32 tensors.
    """
    n_levels = 2 ** bits
    sigma = 1.0 / math.sqrt(dim)

    def pdf(x):
        return (1.0 / math.sqrt(2 * math.pi * sigma ** 2)) * math.exp(
            -x * x / (2 * sigma ** 2)
        )

    lo, hi = -4.0 * sigma, 4.0 * sigma
    centroids = [lo + (hi - lo) * (i + 0.5) / n_levels for i in range(n_levels)]

    for _ in range(n_iter):
        boundaries = [
            (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
        ]
        edges = [-10 * sigma] + boundaries + [10 * sigma]
        new_centroids = []
        for i in range(n_levels):
            a, b = edges[i], edges[i + 1]
            num, _ = integrate.quad(lambda x: x * pdf(x), a, b)
            den, _ = integrate.quad(pdf, a, b)
            new_centroids.append(num / den if den > 1e-15 else centroids[i])
        max_change = max(abs(new_centroids[i] - centroids[i]) for i in range(n_levels))
        centroids = new_centroids
        if max_change < tol:
            break

    boundaries = [
        (centroids[i] + centroids[i + 1]) / 2.0 for i in range(n_levels - 1)
    ]
    return (
        torch.tensor(centroids, dtype=torch.float32),
        torch.tensor(boundaries, dtype=torch.float32),
    )


_codebook_cache: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def get_codebook(dim: int, bits: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get or compute Lloyd-Max codebook (cached)."""
    key = (dim, bits)
    if key not in _codebook_cache:
        _codebook_cache[key] = _solve_lloyd_max(dim, bits)
    return _codebook_cache[key]


class TurboQuantEngine:
    """Core quantization engine for a single channel group.

    Implements Algorithm 1 (MSE) and Algorithm 2 (Prod) from the paper.
    """

    def __init__(
        self,
        dim: int,
        bits: int,
        mode: str = "mse",
        device: str = "cuda",
        seed: int = 42,
    ):
        self.dim = dim
        self.bits = bits
        self.mode = mode
        self.device = device

        self.mse_bits = max(bits - 1, 1) if mode == "prod" else bits

        gen = torch.Generator(device="cpu").manual_seed(seed)
        G = torch.randn(dim, dim, generator=gen)
        Q, R = torch.linalg.qr(G)
        diag_sign = torch.sign(torch.diag(R))
        diag_sign[diag_sign == 0] = 1.0
        self.Pi = (Q * diag_sign.unsqueeze(0)).to(device)

        centroids_cpu, boundaries_cpu = get_codebook(dim, self.mse_bits)
        self.centroids = centroids_cpu.to(device)
        self.boundaries = boundaries_cpu.to(device)

        if mode == "prod":
            gen_qjl = torch.Generator(device="cpu").manual_seed(seed + 10000)
            self.S = torch.randn(dim, dim, generator=gen_qjl).to(device)
            self.qjl_scale = math.sqrt(math.pi / 2) / dim
        else:
            self.S = None
            self.qjl_scale = 0.0

    def _ensure_device(self, device):
        if self.Pi.device != device:
            self.Pi = self.Pi.to(device)
            self.centroids = self.centroids.to(device)
            self.boundaries = self.boundaries.to(device)
            if self.S is not None:
                self.S = self.S.to(device)

    @torch.no_grad()
    def compress(self, x: torch.Tensor) -> dict:
        """Compress vectors. x shape: (..., dim). Returns compressed dict."""
        self._ensure_device(x.device)
        orig_shape = x.shape
        orig_dtype = x.dtype
        flat = x.reshape(-1, self.dim).float()

        norms = torch.norm(flat, dim=-1, keepdim=True).clamp(min=1e-8)
        unit = flat / norms

        rotated = unit @ self.Pi.T
        indices = torch.searchsorted(self.boundaries, rotated.contiguous())
        deq_rotated = self.centroids[indices]
        k_mse_unit = deq_rotated @ self.Pi

        result = {
            "k_mse_unit": k_mse_unit,
            "norms": norms.squeeze(-1),
            "orig_shape": orig_shape,
            "orig_dtype": orig_dtype,
        }

        if self.mode == "prod" and self.S is not None:
            residual = unit - k_mse_unit
            r_norm = torch.norm(residual, dim=-1)
            projected = residual @ self.S.T
            signs = (projected >= 0).to(torch.int8) * 2 - 1
            result["qjl_signs"] = signs
            result["residual_norm"] = r_norm

        return result

    @torch.no_grad()
    def decompress(self, compressed: dict) -> torch.Tensor:
        """Reconstruct vectors from compressed representation."""
        k_mse_unit = compressed["k_mse_unit"]
        norms = compressed["norms"]
        orig_shape = compressed["orig_shape"]
        orig_dtype = compressed.get("orig_dtype", torch.float32)
        self._ensure_device(k_mse_unit.device)

        if self.mode == "prod" and "qjl_signs" in compressed:
            signs = compressed["qjl_signs"].float()
            r_norm = compressed["residual_norm"]
            correction = self.qjl_scale * r_norm.unsqueeze(-1) * (signs @ self.S)
            recon_unit = k_mse_unit + correction
        else:
            recon_unit = k_mse_unit

        return (recon_unit * norms.unsqueeze(-1)).reshape(orig_shape).to(orig_dtype)


class OutlierAwareTurboQuant:
    """Outlier-aware mixed-precision TurboQuant (paper Section 4.3).

    Splits head_dim into outlier and regular channel groups, applying
    higher bit-width to outlier channels for better fidelity.

    Example configurations matching the paper:
      - 2.5-bit: 32 outlier at 4b, 96 regular at 2b → (32*4+96*2)/128 = 2.5
      - 3.5-bit: 32 outlier at 5b, 96 regular at 3b → (32*5+96*3)/128 = 3.5
    """

    def __init__(
        self,
        head_dim: int,
        outlier_bits: int,
        regular_bits: int,
        n_outlier: int,
        key_mode: str = "prod",
        value_mode: str = "mse",
        device: str = "cuda",
        seed: int = 42,
    ):
        self.head_dim = head_dim
        self.outlier_bits = outlier_bits
        self.regular_bits = regular_bits
        self.n_outlier = min(n_outlier, head_dim)
        self.n_regular = head_dim - self.n_outlier
        self.device = device

        effective = (self.n_outlier * outlier_bits + self.n_regular * regular_bits) / head_dim
        self.effective_bits = effective

        self.outlier_key_q = TurboQuantEngine(
            self.n_outlier, outlier_bits, key_mode, device, seed
        )
        self.outlier_val_q = TurboQuantEngine(
            self.n_outlier, outlier_bits, value_mode, device, seed + 100
        )
        self.regular_key_q = TurboQuantEngine(
            self.n_regular, regular_bits, key_mode, device, seed + 200
        )
        self.regular_val_q = TurboQuantEngine(
            self.n_regular, regular_bits, value_mode, device, seed + 300
        )

        self.outlier_indices: Optional[torch.Tensor] = None
        self.regular_indices: Optional[torch.Tensor] = None

    def calibrate(self, sample_states: torch.Tensor):
        """Identify outlier channels from sample KV states.

        sample_states: any shape with last dim = head_dim.
        Selects top-k channels by variance as outliers.
        """
        flat = sample_states.reshape(-1, self.head_dim).float()
        channel_var = flat.var(dim=0)
        _, top_idx = channel_var.topk(self.n_outlier)
        outlier_set = set(top_idx.tolist())
        regular_idx = [i for i in range(self.head_dim) if i not in outlier_set]

        self.outlier_indices = top_idx.sort().values.to(self.device)
        self.regular_indices = torch.tensor(
            sorted(regular_idx), dtype=torch.long, device=self.device
        )

    def set_outlier_indices(self, outlier_indices: torch.Tensor):
        """Set outlier channel indices directly."""
        self.outlier_indices = outlier_indices.to(self.device)
        all_idx = set(range(self.head_dim))
        outlier_set = set(outlier_indices.tolist())
        self.regular_indices = torch.tensor(
            sorted(all_idx - outlier_set), dtype=torch.long, device=self.device
        )

    def _split(self, x: torch.Tensor):
        oi = self.outlier_indices.to(x.device)
        ri = self.regular_indices.to(x.device)
        return x[..., oi], x[..., ri]

    def _merge(self, outlier: torch.Tensor, regular: torch.Tensor, shape):
        result = torch.empty(
            *shape, dtype=outlier.dtype, device=outlier.device
        )
        oi = self.outlier_indices.to(outlier.device)
        ri = self.regular_indices.to(outlier.device)
        result[..., oi] = outlier
        result[..., ri] = regular
        return result

    @torch.no_grad()
    def compress_keys(self, keys: torch.Tensor) -> dict:
        assert self.outlier_indices is not None, "Call calibrate() first"
        k_out, k_reg = self._split(keys)
        return {
            "outlier": self.outlier_key_q.compress(k_out),
            "regular": self.regular_key_q.compress(k_reg),
            "orig_shape": keys.shape,
        }

    @torch.no_grad()
    def decompress_keys(self, compressed: dict) -> torch.Tensor:
        k_out = self.outlier_key_q.decompress(compressed["outlier"])
        k_reg = self.regular_key_q.decompress(compressed["regular"])
        return self._merge(k_out, k_reg, compressed["orig_shape"])

    @torch.no_grad()
    def compress_values(self, values: torch.Tensor) -> dict:
        assert self.outlier_indices is not None, "Call calibrate() first"
        v_out, v_reg = self._split(values)
        return {
            "outlier": self.outlier_val_q.compress(v_out),
            "regular": self.regular_val_q.compress(v_reg),
            "orig_shape": values.shape,
        }

    @torch.no_grad()
    def decompress_values(self, compressed: dict) -> torch.Tensor:
        v_out = self.outlier_val_q.decompress(compressed["outlier"])
        v_reg = self.regular_val_q.decompress(compressed["regular"])
        return self._merge(v_out, v_reg, compressed["orig_shape"])


class UniformTurboQuant:
    """Non-outlier-aware TurboQuant (same bits for all channels). Baseline."""

    def __init__(
        self,
        head_dim: int,
        bits: int,
        key_mode: str = "prod",
        value_mode: str = "mse",
        device: str = "cuda",
        seed: int = 42,
    ):
        self.head_dim = head_dim
        self.bits = bits
        self.effective_bits = float(bits)
        self.device = device

        self.key_q = TurboQuantEngine(head_dim, bits, key_mode, device, seed)
        self.val_q = TurboQuantEngine(head_dim, bits, value_mode, device, seed + 500)

    def calibrate(self, sample_states: torch.Tensor):
        pass  # no calibration needed

    @torch.no_grad()
    def compress_keys(self, keys: torch.Tensor) -> dict:
        return self.key_q.compress(keys)

    @torch.no_grad()
    def decompress_keys(self, compressed: dict) -> torch.Tensor:
        return self.key_q.decompress(compressed)

    @torch.no_grad()
    def compress_values(self, values: torch.Tensor) -> dict:
        return self.val_q.compress(values)

    @torch.no_grad()
    def decompress_values(self, compressed: dict) -> torch.Tensor:
        return self.val_q.decompress(compressed)
