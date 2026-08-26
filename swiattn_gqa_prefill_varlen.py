# ruff: noqa
"""BF16 forward prefill kernel for Switch Attention GQA.

The router is deliberately outside this module.  Its hard decisions are compacted
into two monotonically increasing query-position lists:

* ``full_query_indices`` attends to the complete causal prefix;
* ``local_query_indices`` attends to a causal sliding window.

Each TileLang launch gathers Q directly from the original sequence and scatters O
and LSE back to the same positions.  K/V stay contiguous and shared by both
branches, so routing does not create another KV cache or an S-by-S index tensor.

The kernel targets BF16, D=128, causal GQA prefill.  This file implements forward
only.  LSE uses the repository's log2 convention.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import tilelang
from tilelang import language as T

from .metadata import SwitchAttentionMetadata, build_switch_attention_metadata

_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}

@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _switch_gqa_prefill_branch(
    Q,
    K,
    V,
    QueryIndices,
    QueryCount,
    FullStarts,
    LocalStarts,
    CausalEnds,
    Output,
    Lse,
    heads,
    dim,
    num_queries,
    kv_group,
    window_size,
    sm_scale=None,
    is_local=False,
    block_M=64,
    block_K=64,
    num_stages=1,
    threads=128,
):
    """Execute one compacted Switch Attention branch.

    Grid: ``ceildiv(num_queries, block_M) x batch x query heads``.  QueryIndices
    contains absolute positions and is padded with -1.  QueryCount supplies the
    runtime valid prefix length, allowing a small set of padded capacities to reuse
    compiled kernels.
    """

    assert dim == 128, "the first Switch Attention kernel is specialized for D128"
    assert heads % kv_group == 0, "query heads must be divisible by KV heads"
    assert block_M == 64, "the qualified schedule is BM64"
    assert block_K == 64, "the qualified schedule is BK64"
    assert num_queries % block_M == 0, "query capacity must be block aligned"
    assert num_stages == 1, "the qualified schedule uses one stage"
    assert threads == 128, "the qualified schedule uses 128 threads"
    assert window_size > 0
    if sm_scale is None:
        sm_scale = (1.0 / dim) ** 0.5 * 1.4426950408889634
    else:
        sm_scale = sm_scale * 1.4426950408889634

    batch = T.dynamic("batch")
    seq_len = T.dynamic("seq_len")
    seq_len_kv = T.dynamic("seq_len_kv")

    dtype = T.bfloat16
    accum_dtype = T.float32
    index_dtype = T.int32
    BM = block_M
    BK = block_K
    D = dim
    NM = num_queries // BM
    heads_per_kv = heads // kv_group
    M = BM

    Q: T.Tensor([batch, seq_len, heads, D], dtype)  # type: ignore
    K: T.Tensor([batch, seq_len_kv, kv_group, D], dtype)  # type: ignore
    V: T.Tensor([batch, seq_len_kv, kv_group, D], dtype)  # type: ignore
    QueryIndices: T.Tensor([batch, num_queries], index_dtype)  # type: ignore
    QueryCount: T.Tensor([batch], index_dtype)  # type: ignore
    FullStarts: T.Tensor([seq_len], index_dtype)  # type: ignore
    LocalStarts: T.Tensor([seq_len], index_dtype)  # type: ignore
    CausalEnds: T.Tensor([seq_len], index_dtype)  # type: ignore
    Output: T.Tensor([batch, seq_len, heads, D], dtype)  # type: ignore
    Lse: T.Tensor([batch, seq_len, heads], accum_dtype)  # type: ignore

    with T.Kernel(NM, batch, heads, threads=threads) as (bx, by, bz):
        Q_shared = T.alloc_shared([M, D], dtype)
        K_shared = T.alloc_shared([BK, D], dtype)
        V_shared = T.alloc_shared([BK, D], dtype)
        O_shared = T.alloc_shared([M, D], dtype)

        qpos = T.alloc_fragment([M], index_dtype)
        row_start = T.alloc_fragment([M], index_dtype)
        row_end = T.alloc_fragment([M], index_dtype)
        valid_q = T.alloc_fragment([M], "bool")
        acc_s = T.alloc_fragment([M, BK], accum_dtype)
        acc_s_cast = T.alloc_fragment([M, BK], dtype)
        acc_o = T.alloc_fragment([M, D], accum_dtype)
        row_max = T.alloc_fragment([M], accum_dtype)
        row_max_prev = T.alloc_fragment([M], accum_dtype)
        row_sum = T.alloc_fragment([M], accum_dtype)
        row_sum_block = T.alloc_fragment([M], accum_dtype)
        alpha = T.alloc_fragment([M], accum_dtype)

        b_i = by
        h0 = bz
        kv_i = h0 // heads_per_kv
        q0 = bx * BM
        count = QueryCount[b_i]

        for row in T.Parallel(M):
            slot = q0 + row
            valid_q[row] = slot < count
            qpos[row] = T.if_then_else(
                valid_q[row], QueryIndices[b_i, slot], 0
            )
            row_start[row] = T.if_then_else(
                is_local, LocalStarts[qpos[row]], FullStarts[qpos[row]]
            )
            row_end[row] = CausalEnds[qpos[row]]

        # Metadata requires Q to be ordered by (segment, logical position), while
        # QueryIndices are ordered local Q rows.  The first/last routed rows therefore
        # bound the union even though qrow and gathered-kv row are different under CP.
        first_slot = T.min(q0, T.max(count - 1, 0))
        last_slot = T.min(q0 + BM - 1, T.max(count - 1, 0))
        first_qrow = T.max(QueryIndices[b_i, first_slot], 0)
        last_qrow = T.max(QueryIndices[b_i, last_slot], 0)
        first_key = T.if_then_else(
            q0 < count,
            T.if_then_else(is_local, LocalStarts[first_qrow], FullStarts[first_qrow]),
            seq_len_kv,
        )
        last_key = T.if_then_else(q0 < count, CausalEnds[last_qrow], 0)
        first_block = first_key // BK
        last_block = T.ceildiv(T.min(last_key, seq_len_kv), BK)
        num_key_blocks = T.max(last_block - first_block, 0)

        # Direct indexed Q gather avoids a separate compact-Q allocation/copy.
        # Keep every row copy on the explicit non-descriptor path: QueryIndices
        # makes this a gather, and a 128K tensor's large outer strides are not valid
        # for the compiler's otherwise tempting BMxD descriptor recognition.
        for row in T.serial(M):
            T.copy(
                Q[
                    b_i,
                    T.max(0, T.min(qpos[row], seq_len - 1)),
                    h0,
                    :,
                ],
                Q_shared[row, :],
                disable_tma=True,
            )

        T.fill(acc_o, 0)
        T.fill(row_sum, 0)
        T.fill(row_max, -(2**30))

        for jb in T.Pipelined(num_key_blocks, num_stages=num_stages):
            key_block = first_block + jb
            key0 = key_block * BK
            # The dataloader/all-gather keeps the true KV length.  Clamp only
            # the physical load for the final partial tile; ``legal`` below
            # masks the replicated lane before softmax.  This is the same
            # no-materialized-padding pattern used by the DSA indexer kernels.
            for ni, d_i in T.Parallel(BK, D):
                safe_key = T.max(0, T.min(key0 + ni, seq_len_kv - 1))
                K_shared[ni, d_i] = K[b_i, safe_key, kv_i, d_i]
            for row, ni in T.Parallel(M, BK):
                key_idx = key0 + ni
                legal = (
                    valid_q[row]
                    and (key_idx < seq_len_kv)
                    and (key_idx >= row_start[row])
                    and (key_idx < row_end[row])
                )
                acc_s[row, ni] = T.if_then_else(
                    legal, 0, -T.infinity(accum_dtype)
                )

            T.wgmma_gemm(
                Q_shared,
                K_shared,
                acc_s,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.wait_wgmma(0)

            T.copy(row_max, row_max_prev)
            T.reduce_max(acc_s, row_max, dim=1, clear=False)
            for row in T.Parallel(M):
                row_max[row] = T.max(row_max[row], row_max_prev[row])
                alpha[row] = T.exp2(
                    (row_max_prev[row] - row_max[row]) * sm_scale
                )
            for row, ni in T.Parallel(M, BK):
                acc_s[row, ni] = T.exp2(
                    acc_s[row, ni] * sm_scale
                    - row_max[row] * sm_scale
                )
            T.reduce_sum(acc_s, row_sum_block, dim=1)
            for row in T.Parallel(M):
                row_sum[row] = row_sum[row] * alpha[row] + row_sum_block[row]
            for row, d_i in T.Parallel(M, D):
                acc_o[row, d_i] *= alpha[row]

            T.copy(acc_s, acc_s_cast)
            for ni, d_i in T.Parallel(BK, D):
                safe_key = T.max(0, T.min(key0 + ni, seq_len_kv - 1))
                V_shared[ni, d_i] = V[b_i, safe_key, kv_i, d_i]
            T.gemm(
                acc_s_cast,
                V_shared,
                acc_o,
                policy=T.GemmWarpPolicy.FullRow,
            )

        for row, d_i in T.Parallel(M, D):
            acc_o[row, d_i] /= row_sum[row]
        for row in T.Parallel(M):
            row_sum[row] = (
                T.log2(row_sum[row]) + row_max[row] * sm_scale
            )

        T.copy(acc_o, O_shared)
        for row, d_i in T.Parallel(M, D):
            if valid_q[row]:
                Output[b_i, qpos[row], h0, d_i] = O_shared[row, d_i]
        for row in T.Parallel(M):
            if valid_q[row]:
                Lse[b_i, qpos[row], h0] = row_sum[row]

def _validate_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        raise TypeError("Switch Attention prefill requires BF16 Q/K/V")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("Q/K/V must be tensors on a supported accelerator device")
    if not (q.is_contiguous() and k.is_contiguous() and v.is_contiguous()):
        raise ValueError("Q/K/V must be contiguous")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Q/K/V must use [B,S,H,D] layout")
    if k.shape != v.shape:
        raise ValueError("K and V must have identical shapes")
    if q.shape[0] != k.shape[0]:
        raise ValueError("Q and KV batch sizes must match")
    if q.shape[3] != 128 or k.shape[3] != 128:
        raise ValueError("the first kernel is specialized for head_dim=128")
    if q.shape[2] != 32 or k.shape[2] != 8:
        raise ValueError("the qualified kernel requires Hq=32 and Hkv=8")
    if q.shape[2] % k.shape[2]:
        raise ValueError("query heads must be divisible by KV heads")

def _prepare_indices(
    indices: torch.Tensor,
    *,
    seq_len: int,
    block_m: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if indices.ndim == 1:
        indices = indices.unsqueeze(0)
    if indices.ndim != 2 or indices.shape[0] != 1:
        raise ValueError("the first kernel accepts one batch of query indices")
    if indices.device.type != "cuda":
        raise ValueError("query indices must be on a supported accelerator device")
    indices = indices.to(dtype=torch.int32).contiguous()
    count = int(indices.shape[1])
    if count == 0:
        return indices, torch.zeros((1,), dtype=torch.int32, device=indices.device)
    # Power-of-two capacity buckets bound the number of JIT specializations when
    # router counts fluctuate from step to step.  Inactive CTAs detect q0>=count
    # and execute zero key blocks.
    padded_count = block_m << max(0, math.ceil(count / block_m).bit_length() - 1)
    if padded_count < count:
        padded_count <<= 1
    if padded_count != count:
        padded = torch.full(
            (1, padded_count), -1, dtype=torch.int32, device=indices.device
        )
        padded[:, :count].copy_(indices)
        indices = padded
    if bool(((indices[:, :count] < 0) | (indices[:, :count] >= seq_len)).any().item()):
        raise ValueError("query index is outside [0, sequence_length)")
    if count > 1 and bool((indices[:, 1:count] <= indices[:, : count - 1]).any().item()):
        raise ValueError("query indices must be strictly increasing")
    count_tensor = torch.tensor([count], dtype=torch.int32, device=indices.device)
    return indices, count_tensor

def build_switch_query_indices(full_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compact a B=1 hard route mask into sorted full and local query positions.

    ``True`` routes to full causal attention and ``False`` to sliding-window
    attention.  Router evaluation itself is not part of the TileLang kernel.
    """

    if full_mask.ndim == 2:
        if full_mask.shape[0] != 1:
            raise ValueError("the first kernel supports B=1 routing")
        full_mask = full_mask[0]
    if full_mask.ndim != 1 or full_mask.dtype != torch.bool or not full_mask.is_cuda:
        raise TypeError(
            "full_mask must be a device-resident bool tensor with shape [S] or [1,S]"
        )
    positions = torch.arange(full_mask.numel(), device=full_mask.device, dtype=torch.int32)
    return positions[full_mask], positions[~full_mask]

def switch_gqa_prefill_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    full_query_indices: torch.Tensor,
    local_query_indices: torch.Tensor,
    *,
    metadata: SwitchAttentionMetadata | None = None,
    window_size: int = 1024,
    sm_scale: float | None = None,
    block_m: int = 64,
    block_k: int = 64,
    num_stages: int = 1,
    threads: int = 128,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run Switch Attention prefill forward for a single sequence.

    Returns BF16 output and FP32 log2-LSE.  The two index lists must be sorted,
    disjoint, and together cover every query position exactly once.
    """

    _validate_qkv(q, k, v)
    if q.shape[0] != 1:
        raise ValueError("the first kernel supports batch size 1")
    if window_size <= 0:
        raise ValueError("window_size must be positive")

    q_len = int(q.shape[1])
    kv_len = int(k.shape[1])
    if metadata is None:
        metadata = build_switch_attention_metadata(
            q_len,
            kv_len,
            device=q.device,
            window_size=window_size,
        )
    metadata_tensors = (
        metadata.q_positions,
        metadata.kv_positions,
        metadata.q_segment_ids,
        metadata.kv_segment_ids,
        metadata.full_starts,
        metadata.local_starts,
        metadata.causal_ends,
    )
    bound_tensors = (
        metadata.full_starts,
        metadata.local_starts,
        metadata.causal_ends,
    )
    expected_lengths = (q_len, kv_len, q_len, kv_len, q_len, q_len, q_len)
    for tensor, expected in zip(metadata_tensors, expected_lengths):
        if (
            tensor.device != q.device
            or tensor.dtype != torch.int32
            or tensor.ndim != 1
            or tensor.numel() != expected
            or not tensor.is_contiguous()
        ):
            raise ValueError("invalid Switch Attention position metadata")
    full_indices, full_count = _prepare_indices(
        full_query_indices, seq_len=q_len, block_m=block_m
    )
    local_indices, local_count = _prepare_indices(
        local_query_indices, seq_len=q_len, block_m=block_m
    )
    full_n = int(full_count.item())
    local_n = int(local_count.item())
    if full_n + local_n != q_len:
        raise ValueError("full and local query lists must cover the sequence")

    output = torch.empty_like(q)
    lse = torch.empty(
        q.shape[:3], dtype=torch.float32, device=q.device
    )
    heads = int(q.shape[2])
    kv_group = int(k.shape[2])
    dim = int(q.shape[3])
    if full_n:
        _switch_gqa_prefill_branch(
            q,
            k,
            v,
            full_indices,
            full_count,
            *bound_tensors,
            output,
            lse,
            heads,
            dim,
            int(full_indices.shape[1]),
            kv_group,
            int(window_size),
            sm_scale,
            is_local=False,
            block_M=block_m,
            block_K=block_k,
            num_stages=num_stages,
            threads=threads,
        )
    if local_n:
        _switch_gqa_prefill_branch(
            q,
            k,
            v,
            local_indices,
            local_count,
            *bound_tensors,
            output,
            lse,
            heads,
            dim,
            int(local_indices.shape[1]),
            kv_group,
            int(window_size),
            sm_scale,
            is_local=True,
            block_M=block_m,
            block_K=block_k,
            num_stages=num_stages,
            threads=threads,
        )
    return output, lse
