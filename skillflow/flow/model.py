"""Classic improved MeanFlow: conditional DiT, boundary JVP loss, one-step sampling."""

from __future__ import annotations

import math

from typing import Optional, Union

import torch

import torch.nn.functional as F

from torch import nn

MEANFLOW_TRAINING_OBJECTIVE_IMPROVED = "improved_boundary_velocity_vloss_v1"

MEANFLOW_SAMPLER_INTERVAL_AVERAGE = "meanflow_interval_average_velocity"

FLOW_ARCHITECTURE_VERSION = "tokenized_dit_combined_condition_v3"

FLOW_OUTPUT_PARAMETERIZATION = "zero_init_tokenwise_conditioned_final_v1"

FLOW_CONDITION_INJECTION = (
    "t_h_combined_task_goal_adaln_zero_every_block_and_final_v1"
)

def project_unit(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Project latent vectors to the unit hypersphere used by the frozen codec."""
    return F.normalize(x, p=2, dim=-1, eps=eps)

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype)
            * (-math.log(10000) / max(half - 1, 1))
        )
        x = t[:, None] * freq[None]
        out = torch.cat([x.sin(), x.cos()], dim=-1)
        return F.pad(out, (0, self.dim - out.shape[-1]))

def _modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Apply DiT's tokenwise adaLN shift and scale."""

    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def _fixed_1d_sincos_embedding(num_tokens: int, hidden_dim: int) -> torch.Tensor:
    """Return a fixed ``[1, num_tokens, hidden_dim]`` 1-D sin/cos table."""

    half = hidden_dim // 2
    frequencies = torch.exp(
        torch.arange(half, dtype=torch.float64)
        * (-math.log(10000.0) / max(half - 1, 1))
    )
    positions = torch.arange(num_tokens, dtype=torch.float64).unsqueeze(1)
    angles = positions * frequencies.unsqueeze(0)
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if embedding.shape[-1] < hidden_dim:
        embedding = F.pad(embedding, (0, hidden_dim - embedding.shape[-1]))
    return embedding.to(dtype=torch.float32).unsqueeze(0)

class MultiheadSelfAttention(nn.Module):
    """Forward-AD-safe multi-head self-attention used by MeanFlow JVPs."""

    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by attention_heads: "
                f"{hidden_dim} % {num_heads} != 0"
            )
        self.num_heads = int(num_heads)
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=True)
        self.proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, hidden = x.shape
        qkv = self.qkv(x).reshape(
            batch,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = weights.softmax(dim=-1)
        attended = torch.matmul(weights, v)
        attended = attended.transpose(1, 2).reshape(batch, tokens, hidden)
        return self.proj(attended)

class MeanFlowDiTBlock(nn.Module):
    """A DiT block with official six-way adaLN-Zero modulation."""

    def __init__(
        self,
        hidden_dim: int,
        attention_heads: int,
        mlp_ratio: float,
    ):
        super().__init__()
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        if mlp_hidden_dim <= 0:
            raise ValueError(
                "hidden_dim * mlp_ratio must produce a positive MLP width"
            )
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = MultiheadSelfAttention(hidden_dim, attention_heads)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden_dim, hidden_dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6, bias=True),
        )
        modulation_output = self.adaLN_modulation[-1]
        assert isinstance(modulation_output, nn.Linear)
        nn.init.zeros_(modulation_output.weight)
        nn.init.zeros_(modulation_output.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        (
            shift_attn,
            scale_attn,
            gate_attn,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(condition).chunk(6, dim=-1)
        x = x + gate_attn.unsqueeze(1) * self.attn(
            _modulate(self.norm1(x), shift_attn, scale_attn)
        )
        return x + gate_mlp.unsqueeze(1) * self.mlp(
            _modulate(self.norm2(x), shift_mlp, scale_mlp)
        )

class MeanFlowFinalLayer(nn.Module):
    """DiT's conditioned final normalization and zero-initialized projection."""

    def __init__(self, hidden_dim: int, latent_token_size: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(
            hidden_dim,
            elementwise_affine=False,
            eps=1e-6,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2, bias=True),
        )
        self.linear = nn.Linear(hidden_dim, latent_token_size, bias=True)
        modulation_output = self.adaLN_modulation[-1]
        assert isinstance(modulation_output, nn.Linear)
        nn.init.zeros_(modulation_output.weight)
        nn.init.zeros_(modulation_output.bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(condition).chunk(2, dim=-1)
        return self.linear(_modulate(self.norm_final(x), shift, scale))

def _batch_time(
    value: torch.Tensor,
    *,
    batch_size: int,
    reference: torch.Tensor,
    label: str,
) -> torch.Tensor:
    """Canonicalize a scalar time to ``[batch]`` without severing JVPs."""

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if value.ndim == 0:
        value = value.expand(batch_size)
    elif value.ndim == 2 and value.shape[1] == 1:
        value = value[:, 0]
    elif value.ndim != 1:
        raise ValueError(f"{label} must be scalar, [batch], or [batch, 1]")
    if value.shape[0] != batch_size:
        raise ValueError(
            f"{label} batch size {value.shape[0]} does not match {batch_size}"
        )
    return value.to(device=reference.device, dtype=reference.dtype)

class VelocityField(nn.Module):
    """Tokenized conditional DiT for the Euclidean MeanFlow average velocity.

    The flat codec latent is split into fixed-width tokens and processed with
    self-attention. Every DiT block and the final projection receives the same
    ``t + (t-r) + y`` condition through adaLN-Zero. ``y`` is the one combined
    frozen query embedding, which already represents task type and goal. The
    final linear layer is zero initialized, so both the untrained output and its
    MeanFlow JVP are exactly zero without an identity skip or gate.
    """

    def __init__(
        self,
        z_dim: int,
        q_dim: int,
        hidden_dim: int,
        depth: int,
        time_dim: int,
        context_dim: int,
        attention_heads: int = 8,
        latent_token_size: int = 16,
        mlp_ratio: float = 4.0,
        architecture: str = FLOW_ARCHITECTURE_VERSION,
        output_parameterization: str = FLOW_OUTPUT_PARAMETERIZATION,
        condition_injection: str = FLOW_CONDITION_INJECTION,
    ):
        super().__init__()
        for label, value in (
            ("z_dim", z_dim),
            ("q_dim", q_dim),
            ("hidden_dim", hidden_dim),
            ("time_dim", time_dim),
            ("context_dim", context_dim),
            ("attention_heads", attention_heads),
            ("latent_token_size", latent_token_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{label} must be an integer")
            if value <= 0:
                raise ValueError(f"{label} must be positive, got {value}")
        if isinstance(depth, bool) or not isinstance(depth, int):
            raise TypeError("depth must be an integer")
        if depth < 0:
            raise ValueError(f"depth must be non-negative, got {depth}")
        if z_dim % latent_token_size != 0:
            raise ValueError(
                "z_dim must be divisible by latent_token_size: "
                f"{z_dim} % {latent_token_size} != 0"
            )
        if hidden_dim % attention_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by attention_heads: "
                f"{hidden_dim} % {attention_heads} != 0"
            )
        if isinstance(mlp_ratio, bool) or not isinstance(mlp_ratio, (int, float)):
            raise TypeError("mlp_ratio must be a real number")
        mlp_ratio = float(mlp_ratio)
        if not math.isfinite(mlp_ratio) or mlp_ratio <= 0.0:
            raise ValueError(f"mlp_ratio must be finite and positive, got {mlp_ratio}")
        self.z_dim = z_dim
        self.q_dim = q_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.time_dim = time_dim
        self.context_dim = int(context_dim)
        self.attention_heads = int(attention_heads)
        self.latent_token_size = int(latent_token_size)
        self.num_latent_tokens = z_dim // latent_token_size
        self.mlp_ratio = mlp_ratio
        expected_metadata = {
            "architecture": FLOW_ARCHITECTURE_VERSION,
            "output_parameterization": FLOW_OUTPUT_PARAMETERIZATION,
            "condition_injection": FLOW_CONDITION_INJECTION,
        }
        actual_metadata = {
            "architecture": architecture,
            "output_parameterization": output_parameterization,
            "condition_injection": condition_injection,
        }
        for label, expected in expected_metadata.items():
            actual = actual_metadata[label]
            if actual != expected:
                raise ValueError(
                    f"unsupported {label}: {actual!r}; expected {expected!r}"
                )
        self.architecture = architecture
        self.output_parameterization = output_parameterization
        self.condition_injection = condition_injection
        self.t_embedder = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.h_embedder = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.y_embedder = nn.Sequential(
            nn.Linear(q_dim, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, hidden_dim),
        )
        self.x_embedder = nn.Linear(latent_token_size, hidden_dim)
        self.register_buffer(
            "pos_embed",
            _fixed_1d_sincos_embedding(self.num_latent_tokens, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                MeanFlowDiTBlock(hidden_dim, attention_heads, mlp_ratio)
                for _ in range(depth)
            ]
        )
        self.final_layer = MeanFlowFinalLayer(hidden_dim, latent_token_size)

    def forward(
        self,
        z_t: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        q: torch.Tensor,
    ) -> torch.Tensor:
        if z_t.ndim != 2 or z_t.shape[1] != self.z_dim:
            raise ValueError(
                f"z_t must have shape [batch, {self.z_dim}], got {tuple(z_t.shape)}"
            )
        if q.ndim != 2 or q.shape != (z_t.shape[0], self.q_dim):
            raise ValueError(
                f"q must have shape [{z_t.shape[0]}, {self.q_dim}], "
                f"got {tuple(q.shape)}"
            )
        r = _batch_time(r, batch_size=z_t.shape[0], reference=z_t, label="r")
        t = _batch_time(t, batch_size=z_t.shape[0], reference=z_t, label="t")
        h = t - r
        q_rms = q * torch.rsqrt(
            q.square().mean(dim=-1, keepdim=True).clamp_min(1e-8)
        )
        condition = (
            self.t_embedder(t)
            + self.h_embedder(h)
            + self.y_embedder(q_rms)
        )
        tokens = z_t.reshape(
            z_t.shape[0],
            self.num_latent_tokens,
            self.latent_token_size,
        )
        hidden = self.x_embedder(tokens) + self.pos_embed.to(
            device=z_t.device,
            dtype=z_t.dtype,
        )
        for block in self.blocks:
            hidden = block(hidden, condition)
        output_tokens = self.final_layer(hidden, condition)
        return output_tokens.reshape(z_t.shape[0], self.z_dim)

    def config(self) -> dict:
        configuration = {
            "z_dim": self.z_dim,
            "q_dim": self.q_dim,
            "hidden_dim": self.hidden_dim,
            "depth": self.depth,
            "time_dim": self.time_dim,
            "context_dim": self.context_dim,
            "attention_heads": self.attention_heads,
            "latent_token_size": self.latent_token_size,
            "mlp_ratio": self.mlp_ratio,
            "architecture": self.architecture,
            "output_parameterization": self.output_parameterization,
            "condition_injection": self.condition_injection,
        }
        return configuration

def sample_meanflow_times(
    batch_size: int,
    *,
    device: Union[str, torch.device],
    dtype: torch.dtype = torch.float32,
    logit_normal_mean: float = -0.4,
    logit_normal_std: float = 1.0,
    offdiag_probability: float = 0.25,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the ordered ``(r, t)`` pair used by MeanFlow.

    Two independent logit-normal times are sorted so ``0 < r <= t < 1``.
    Only ``offdiag_probability`` of examples retain the two distinct values;
    the remainder use the diagonal objective ``r = t``.  Defaults match the
    ImageNet setting from the original paper.
    """

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(
            "batch_size must be an integer, "
            f"got {type(batch_size).__name__}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    mean = float(logit_normal_mean)
    std = float(logit_normal_std)
    probability = float(offdiag_probability)
    if not math.isfinite(mean):
        raise ValueError(f"logit_normal_mean must be finite, got {mean!r}")
    if not math.isfinite(std) or std <= 0.0:
        raise ValueError(f"logit_normal_std must be finite and positive, got {std!r}")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(
            "offdiag_probability must be finite and in [0, 1], "
            f"got {probability!r}"
        )
    if not dtype.is_floating_point:
        raise TypeError(f"dtype must be floating point, got {dtype}")

    logits = torch.randn(
        (batch_size, 2),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    times = torch.sigmoid(logits * std + mean).sort(dim=-1).values
    r, t = times.unbind(dim=-1)
    if probability < 1.0:
        keep_offdiag = torch.rand(
            batch_size,
            device=device,
            generator=generator,
        ) < probability
        r = torch.where(keep_offdiag, r, t)
    return r, t

def sample_meanflow_prior(
    batch_size: int,
    latent_dim: int,
    device: Union[str, torch.device],
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample the iMF prior ``epsilon ~ N(0, I)``.

    The prior is deliberately *not* unit-normalized. MeanFlow interpolation,
    targets, JVPs, and predicted average velocities remain Euclidean. Only the
    final 1-NFE endpoint is projected for the frozen unit-latent codec.
    """

    for label, value in (("batch_size", batch_size), ("latent_dim", latent_dim)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(
                f"{label} must be an integer, got {type(value).__name__}"
            )
        if value <= 0:
            raise ValueError(f"{label} must be positive, got {value}")
    if not dtype.is_floating_point:
        raise TypeError(f"dtype must be floating point, got {dtype}")

    epsilon = torch.randn(
        (batch_size, latent_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    if not bool(torch.isfinite(epsilon).all().item()):
        raise FloatingPointError("sampled MeanFlow prior contains NaN/Inf")
    return epsilon

def meanflow_jvp(
    model: nn.Module,
    z_t: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    q: torch.Tensor,
    *,
    instantaneous_velocity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the iMF field value and its total derivative along the path.

    The boundary prediction supplies the JVP direction. The interval start and
    combined task-goal embedding remain closed-over constants.
    """
    if instantaneous_velocity.shape != z_t.shape:
        raise ValueError(
            "instantaneous_velocity must have the same shape as z_t: "
            f"{tuple(instantaneous_velocity.shape)} != {tuple(z_t.shape)}"
        )

    def field_along_path(
        path_latent: torch.Tensor,
        path_time: torch.Tensor,
    ) -> torch.Tensor:
        return model(path_latent, r, path_time, q)

    return torch.func.jvp(
        field_along_path,
        (z_t, t),
        (instantaneous_velocity, torch.ones_like(t)),
    )

def improved_meanflow_boundary_target(
    model: nn.Module,
    z_t: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    q: torch.Tensor,
    *,
    conditional_velocity: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build the parameter-free boundary-iMF target.

    Improved MeanFlow replaces the sample-specific JVP tangent ``epsilon-x``
    with the marginal-velocity estimate ``u_theta(z_t, t, t, q)``. The
    returned regression target is algebraically equivalent to optimizing the
    iMF compound prediction

    ``V_theta = u_theta + (t-r) * stopgrad(d_t u_theta)``

    against ``conditional_velocity``. All three returned tensors are detached;
    the caller computes the gradient-carrying ``u_theta`` prediction separately.
    This is the boundary formulation from iMF and introduces no auxiliary head.
    """

    if conditional_velocity.shape != z_t.shape:
        raise ValueError(
            "conditional_velocity must have the same shape as z_t: "
            f"{tuple(conditional_velocity.shape)} != {tuple(z_t.shape)}"
        )
    with torch.no_grad():
        boundary_velocity = model(z_t, t, t, q)
        _validate_velocity(
            boundary_velocity,
            "improved MeanFlow boundary velocity",
            z_t,
        )
        _average_velocity, average_velocity_jvp = meanflow_jvp(
            model,
            z_t,
            r,
            t,
            q,
            instantaneous_velocity=boundary_velocity,
        )
        target = meanflow_target(
            conditional_velocity,
            average_velocity_jvp,
            r,
            t,
        )
    return target.detach(), boundary_velocity.detach(), average_velocity_jvp.detach()

def meanflow_target(
    instantaneous_velocity: torch.Tensor,
    average_velocity_jvp: torch.Tensor,
    r: torch.Tensor,
    t: torch.Tensor,
    *,
    detach: bool = True,
) -> torch.Tensor:
    """Build the MeanFlow target ``v_t - (t-r) d_t u_theta``.

    The paper stop-gradients this self-referential target.  ``detach=False`` is
    available for numerical identity tests, while training should use the
    default detached target.
    """

    if average_velocity_jvp.shape != instantaneous_velocity.shape:
        raise ValueError(
            "average_velocity_jvp must have the same shape as "
            "instantaneous_velocity"
        )
    r = _batch_time(
        r,
        batch_size=instantaneous_velocity.shape[0],
        reference=instantaneous_velocity,
        label="r",
    )
    t = _batch_time(
        t,
        batch_size=instantaneous_velocity.shape[0],
        reference=instantaneous_velocity,
        label="t",
    )
    interval = t - r
    while interval.ndim < instantaneous_velocity.ndim:
        interval = interval.unsqueeze(-1)
    target = instantaneous_velocity - interval * average_velocity_jvp
    return target.detach() if detach else target

def _validate_batched_vectors(
    value: torch.Tensor,
    label: str,
    *,
    require_nonzero: bool,
    expected_shape: Optional[torch.Size] = None,
) -> None:
    """Fail before normalization can silently hide corrupt latent vectors."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{label} must be a torch.Tensor, got {type(value).__name__}")
    if value.ndim != 2:
        raise ValueError(
            f"{label} must have shape [batch, dim], got {tuple(value.shape)}"
        )
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"{label} must have a non-empty batch and dimension")
    if expected_shape is not None and value.shape != expected_shape:
        raise ValueError(
            f"{label} shape mismatch: got {tuple(value.shape)}, "
            f"expected {tuple(expected_shape)}"
        )
    if not torch.is_floating_point(value):
        raise TypeError(f"{label} must use a floating-point dtype, got {value.dtype}")
    if not bool(torch.isfinite(value).all().item()):
        invalid = int((~torch.isfinite(value)).sum().item())
        raise FloatingPointError(
            f"{label} contains {invalid} NaN/Inf value(s); refusing flow integration"
        )
    if require_nonzero:
        norms = value.norm(dim=-1)
        invalid_norm = (~torch.isfinite(norms)) | (norms <= 1e-8)
        if bool(invalid_norm.any().item()):
            rows = invalid_norm.nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"{label} contains zero/invalid-norm vector(s) at batch rows {rows[:8]}"
            )

def _validate_integration_inputs(
    epsilon: torch.Tensor,
    q: torch.Tensor,
) -> None:
    _validate_batched_vectors(epsilon, "epsilon", require_nonzero=True)
    _validate_batched_vectors(q, "q", require_nonzero=True)
    if q.shape[0] != epsilon.shape[0]:
        raise ValueError(
            f"q batch size {q.shape[0]} does not match epsilon batch "
            f"size {epsilon.shape[0]}"
        )
    if q.device != epsilon.device:
        raise ValueError(
            f"q device {q.device} does not match epsilon device {epsilon.device}"
        )
    if q.dtype != epsilon.dtype:
        raise ValueError(
            f"q dtype {q.dtype} does not match epsilon dtype {epsilon.dtype}"
        )

def _validate_velocity(
    value: torch.Tensor,
    label: str,
    reference: torch.Tensor,
) -> None:
    _validate_batched_vectors(
        value,
        label,
        require_nonzero=False,
        expected_shape=reference.shape,
    )

@torch.no_grad()
def sample_meanflow_endpoint(model, epsilon, q, *, sampling_steps=1,
                             sampler=MEANFLOW_SAMPLER_INTERVAL_AVERAGE,
                             guidance_scale=1.0, max_velocity_norm=None):
    """Sample with u(epsilon, r=0, t=1, q), then unit-project for Codec."""
    if type(sampling_steps) is not int or sampling_steps != 1:
        raise ValueError("This release requires sampling_steps=1 (NFE=1)")
    if sampler != MEANFLOW_SAMPLER_INTERVAL_AVERAGE:
        raise ValueError("Only improved MeanFlow interval-average sampling is supported")
    if float(guidance_scale) != 1.0 or max_velocity_norm not in (None, 0, 0.0):
        raise ValueError("Guidance and velocity clipping are disabled")
    _validate_integration_inputs(epsilon, q)
    model.eval()
    t = torch.ones(epsilon.shape[0], device=epsilon.device, dtype=epsilon.dtype)
    r = torch.zeros_like(t)
    velocity = model(epsilon, r, t, project_unit(q))
    _validate_velocity(velocity, "one-step interval-average velocity", epsilon)
    result = project_unit(epsilon - velocity)
    _validate_batched_vectors(result, "one-step endpoint", require_nonzero=True,
                              expected_shape=epsilon.shape)
    return result
