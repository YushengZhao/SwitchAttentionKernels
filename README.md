# Switch Attention Kernels

High-performance prefill kernels for **Switch Attention (SwiAttn)**, implemented in [TileLang](https://github.com/tile-ai/tilelang).

This repository accompanies our paper:

> **Switch Attention: Towards Dynamic and Fine-grained Hybrid Transformers**  
> Yusheng Zhao, Hourun Li, Bohan Wu, Yichun Yin, Lifeng Shang, Jingyang Yuan, Meng Zhang, and Ming Zhang  
> [Paper](https://arxiv.org/abs/2603.26380)

## Overview

Switch Attention is a dynamic hybrid attention mechanism. At every Transformer layer, each query token is routed to one of two branches:

- **Full causal attention** for global information aggregation.
- **Causal sliding-window attention** for efficient local pattern matching.

Unlike static hybrid architectures that assign an attention type to an entire layer, SwiAttn makes a fine-grained routing decision for each token at each layer. See the [paper](https://arxiv.org/abs/2603.26380) for the model, training objective, and experimental results.

## Kernel design

The kernels target the prefill phase of causal grouped-query attention (GQA). A hard routing mask is compacted into two monotonically increasing query-index lists, one for each attention branch. Each branch is executed by a separate TileLang launch.

Key implementation features include:

- **Compacted query execution.** Only queries assigned to a branch are processed by that branch.
- **Direct Q gather and O/LSE scatter.** Queries are loaded from their original sequence positions, and results are written back to the same positions without materializing an additional compact-Q tensor.
- **Shared contiguous K/V.** Both branches reuse the same K/V tensors; routing does not duplicate the KV cache.
- **No dense routing matrix.** The implementation uses compact query indices rather than an \(S \times S\) index or mask tensor.
- **Online softmax.** Softmax statistics and output accumulation use FP32 across key tiles.
- **GQA head mapping.** Multiple query heads share each KV head.
- **Bounded JIT specialization.** Routed query counts are padded into capacity buckets so changing routing decisions do not require an unbounded number of compiled kernels.

## Installation

### Requirements

- Python 3.10 or newer
- PyTorch with an accelerator backend supported by TileLang
- TileLang 0.1.11

Create an environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch tilelang==0.1.11
```

Then clone this repository:

```bash
git clone https://github.com/YushengZhao/SwitchAttentionKernels.git
cd SwitchAttentionKernels
```

For source builds, nightly wheels, and backend-specific toolchain options, refer to the official [TileLang installation guide](https://tilelang.com/get_started/Installation.html).

## Usage

The forward API accepts Q/K/V tensors and two sorted, disjoint query-index lists:

```python
output, lse = switch_gqa_prefill_forward(
    q,
    k,
    v,
    full_query_indices,
    local_query_indices,
    window_size=1024,
)
```

Tensor and index contracts:

- `q`: contiguous device-resident BF16 tensor with shape `[B, Sq, Hq, 128]`.
- `k`, `v`: contiguous device-resident BF16 tensors with shape `[B, Skv, Hkv, 128]`.
- `full_query_indices`: sorted positions routed to full causal attention.
- `local_query_indices`: sorted positions routed to causal sliding-window attention.
- The two index lists must be disjoint and must cover every query position exactly once.
- `output`: BF16 tensor with the same shape as `q`.
- `lse`: FP32 tensor with shape `[B, Sq, Hq]`, stored in the kernel's log2 convention.

The router itself is outside the kernel. Given a device-resident Boolean mask in which `True` selects full attention, `build_switch_query_indices` compacts it into the two query lists.

## Citation

If you find Switch Attention or these kernels useful, please cite:

```bibtex
@article{zhao2026switch,
  title={Switch Attention: Towards Dynamic and Fine-grained Hybrid Transformers},
  author={Zhao, Yusheng and Li, Hourun and Wu, Bohan and Yin, Yichun and Shang, Lifeng and Yuan, Jingyang and Zhang, Meng and Zhang, Ming},
  journal={arXiv preprint arXiv:2603.26380},
  year={2026}
}
```

## Acknowledgements

The kernels are implemented with [TileLang](https://github.com/tile-ai/tilelang). We thank the TileLang and PyTorch communities for their open-source work.
