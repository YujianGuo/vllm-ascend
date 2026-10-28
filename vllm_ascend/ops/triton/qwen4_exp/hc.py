"""Triton-Ascend HyperConnection kernels for Qwen4Exp."""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _grouped_gemma_rmsnorm_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    stride_x,
    stride_y,
    DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    group_dim: tl.constexpr = DIM // NUM_GROUPS

    program_id = tl.program_id(0)
    group_id = program_id % NUM_GROUPS
    row = program_id // NUM_GROUPS
    group_offsets = tl.arange(0, BLOCK_SIZE)
    offsets = group_id * group_dim + group_offsets
    mask = group_offsets < group_dim
    weight_offsets = group_offsets if W_SHARED else offsets

    values = tl.load(
        x_ptr + row * stride_x + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(w_ptr + weight_offsets, mask=mask, other=0.0)
    reciprocal_rms = tl.rsqrt(tl.sum(values * values) / group_dim + EPS)
    output = values * reciprocal_rms
    output += output * weight.to(tl.float32)
    tl.store(y_ptr + row * stride_y + offsets, output, mask=mask)


def _grouped_gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_groups: int,
) -> torch.Tensor:
    rows, dim = x.shape
    assert x.stride(1) == 1
    assert weight.is_contiguous()
    assert dim % num_groups == 0
    group_dim = dim // num_groups
    assert weight.numel() in (group_dim, dim)

    output = torch.empty_like(x)
    _grouped_gemma_rmsnorm_kernel[(rows * num_groups,)](
        x,
        weight,
        output,
        x.stride(0),
        output.stride(0),
        DIM=dim,
        NUM_GROUPS=num_groups,
        W_SHARED=weight.numel() == group_dim,
        EPS=eps,
        BLOCK_SIZE=triton.next_power_of_2(group_dim),
    )
    return output


@triton.jit
def _hc_silu_kernel(
    x_ptr,
    y_ptr,
    stride_x,
    stride_y,
    DIM: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < DIM
    values = tl.load(
        x_ptr + row * stride_x + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    values /= HC_COUNT
    output = values * tl.sigmoid(values)
    tl.store(y_ptr + row * stride_y + offsets, output, mask=mask)


def _hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    rows, dim = x.shape
    assert x.stride(1) == 1
    output = torch.empty_like(x)
    _hc_silu_kernel[(rows,)](
        x,
        output,
        x.stride(0),
        output.stride(0),
        DIM=dim,
        HC_COUNT=hc_count,
        BLOCK_SIZE=triton.next_power_of_2(dim),
    )
    return output


@triton.jit
def _hc_gate_mix_kernel(
    x_ptr,
    gate_ptr,
    output_ptr,
    stride_x,
    stride_gate,
    stride_output,
    DIM: tl.constexpr,
    HC_COUNT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    hidden_size: tl.constexpr = DIM // HC_COUNT
    row = tl.program_id(0)
    tile_id = tl.program_id(1)
    hidden_offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = hidden_offsets < hidden_size

    accumulator = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for stream in tl.static_range(HC_COUNT):
        offsets = stream * hidden_size + hidden_offsets
        gate = tl.load(
            gate_ptr + row * stride_gate + offsets,
            mask=mask,
            other=0.0,
        )
        values = tl.load(
            x_ptr + row * stride_x + offsets,
            mask=mask,
            other=0.0,
        )
        accumulator += tl.sigmoid(gate.to(tl.float32)) * values.to(tl.float32)
    accumulator /= HC_COUNT
    tl.store(
        output_ptr + row * stride_output + hidden_offsets,
        accumulator,
        mask=mask,
    )


def _hc_gate_mix(
    x: torch.Tensor,
    gate: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    rows, dim = gate.shape
    assert x.shape == gate.shape
    assert dim % hc_count == 0
    assert x.stride(1) == 1
    assert gate.stride(1) == 1

    hidden_size = dim // hc_count
    output = x.new_empty(rows, hidden_size)
    block_size = 512
    _hc_gate_mix_kernel[(rows, triton.cdiv(hidden_size, block_size))](
        x,
        gate,
        output,
        x.stride(0),
        gate.stride(0),
        output.stride(0),
        DIM=dim,
        HC_COUNT=hc_count,
        BLOCK_SIZE=block_size,
    )
    return output


@triton.jit
def _hc_combine_kernel(
    block_ptr,
    residual_ptr,
    injection_ptr,
    output_ptr,
    stride_block,
    stride_residual,
    stride_injection,
    stride_output,
    HIDDEN_SIZE: tl.constexpr,
    HC_COUNT: tl.constexpr,
    HC_PAD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    tile_id = tl.program_id(1)
    hidden_offsets = tile_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    stream_offsets = tl.arange(0, HC_PAD)
    stream_mask = stream_offsets < HC_COUNT
    offsets = stream_offsets[:, None] * HIDDEN_SIZE + hidden_offsets[None, :]
    mask = stream_mask[:, None] & hidden_mask[None, :]

    injection = tl.load(
        injection_ptr + row * stride_injection + stream_offsets,
        mask=stream_mask,
        other=0.0,
    )
    block = tl.load(
        block_ptr + row * stride_block + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    )
    residual = tl.load(
        residual_ptr + row * stride_residual + offsets,
        mask=mask,
        other=0.0,
    )
    injection = 2.0 * tl.sigmoid(injection.to(tl.float32) / HC_COUNT)
    output = (
        residual.to(tl.float32)
        + block.to(tl.float32)[None, :] * injection[:, None]
    )
    tl.store(
        output_ptr + row * stride_output + offsets,
        output,
        mask=mask,
    )


def _hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    rows, dim = residual.shape
    assert dim % hc_count == 0
    hidden_size = dim // hc_count
    assert block_output.shape == (rows, hidden_size)
    assert injection_logits.shape == (rows, hc_count)
    assert residual.stride(1) == 1
    assert block_output.stride(1) == 1
    assert injection_logits.stride(1) == 1

    output = torch.empty_like(residual)
    block_size = 512
    _hc_combine_kernel[(rows, triton.cdiv(hidden_size, block_size))](
        block_output,
        residual,
        injection_logits,
        output,
        block_output.stride(0),
        residual.stride(0),
        injection_logits.stride(0),
        output.stride(0),
        HIDDEN_SIZE=hidden_size,
        HC_COUNT=hc_count,
        HC_PAD=triton.next_power_of_2(hc_count),
        BLOCK_SIZE=block_size,
    )
    return output


def _hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    output = _hc_combine(
        residual,
        block_output,
        injection_logits,
        hc_count,
    )
    norm_output = _grouped_gemma_rmsnorm(
        output,
        norm_weight,
        eps,
        hc_count,
    )
    return output, norm_output


def _same_shape_fake(x: torch.Tensor, *args) -> torch.Tensor:
    return torch.empty_like(x)


def _hc_gate_mix_fake(
    x: torch.Tensor,
    gate: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    del gate
    return x.new_empty((x.shape[0], x.shape[1] // hc_count))


def _hc_combine_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    del block_output, injection_logits, hc_count
    return torch.empty_like(residual)


def _hc_combine_norm_fake(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    del block_output, injection_logits, norm_weight, eps, hc_count
    return torch.empty_like(residual), torch.empty_like(residual)


direct_register_custom_op(
    op_name="qwen4_exp_grouped_gemma_rmsnorm_ascend",
    op_func=_grouped_gemma_rmsnorm,
    fake_impl=_same_shape_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_silu_ascend",
    op_func=_hc_silu,
    fake_impl=_same_shape_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_gate_mix_ascend",
    op_func=_hc_gate_mix,
    fake_impl=_hc_gate_mix_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_combine_ascend",
    op_func=_hc_combine,
    fake_impl=_hc_combine_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_hc_combine_norm_ascend",
    op_func=_hc_combine_norm,
    fake_impl=_hc_combine_norm_fake,
)


def grouped_gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    num_groups: int,
) -> torch.Tensor:
    if x.device.type == "cpu":
        group_dim = x.shape[-1] // num_groups
        grouped = x.float().view(x.shape[0], num_groups, group_dim)
        normalized = grouped * torch.rsqrt(
            grouped.square().mean(-1, keepdim=True) + eps
        )
        weight_shape = (
            (1, 1, group_dim)
            if weight.numel() == group_dim
            else (1, num_groups, group_dim)
        )
        return (
            normalized * (1.0 + weight.float().view(weight_shape))
        ).to(x.dtype).view_as(x)
    return torch.ops.vllm.qwen4_exp_grouped_gemma_rmsnorm_ascend(
        x,
        weight,
        eps,
        num_groups,
    )


def hc_silu(x: torch.Tensor, hc_count: int) -> torch.Tensor:
    if x.device.type == "cpu":
        return torch.nn.functional.silu(x.float() / hc_count).to(x.dtype)
    return torch.ops.vllm.qwen4_exp_hc_silu_ascend(x, hc_count)


def hc_gate_mix(
    x: torch.Tensor,
    gate: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if x.device.type == "cpu":
        group_dim = x.shape[-1] // hc_count
        mixed = torch.sigmoid(gate.float()) * x.float()
        return mixed.view(-1, hc_count, group_dim).mean(1).to(x.dtype)
    return torch.ops.vllm.qwen4_exp_hc_gate_mix_ascend(x, gate, hc_count)


def hc_combine(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    hc_count: int,
) -> torch.Tensor:
    if residual.device.type == "cpu":
        group_dim = residual.shape[-1] // hc_count
        injection = 2.0 * torch.sigmoid(injection_logits.float() / hc_count)
        output = residual.float().view(-1, hc_count, group_dim)
        output += block_output.float().unsqueeze(1) * injection.unsqueeze(-1)
        return output.to(residual.dtype).view_as(residual)
    return torch.ops.vllm.qwen4_exp_hc_combine_ascend(
        residual,
        block_output,
        injection_logits,
        hc_count,
    )


def hc_combine_norm(
    residual: torch.Tensor,
    block_output: torch.Tensor,
    injection_logits: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    hc_count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if residual.device.type == "cpu":
        output = hc_combine(
            residual,
            block_output,
            injection_logits,
            hc_count,
        )
        return output, grouped_gemma_rmsnorm(
            output,
            norm_weight,
            eps,
            hc_count,
        )
    return torch.ops.vllm.qwen4_exp_hc_combine_norm_ascend(
        residual,
        block_output,
        injection_logits,
        norm_weight,
        eps,
        hc_count,
    )


__all__ = [
    "grouped_gemma_rmsnorm",
    "hc_combine",
    "hc_combine_norm",
    "hc_gate_mix",
    "hc_silu",
]
