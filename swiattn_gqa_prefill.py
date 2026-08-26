
_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
}

@tilelang.jit(pass_configs=_PASS_CONFIGS)
def _switch_gqa_prefill_branch(
    Q,
    K,
    V,
    QueryIndices,
    QueryCount,
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
    num_stages=2,
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
    assert block_M in (64, 128), "qualified query tiles are BM64/BM128"
    assert block_K in (64, 128), "qualified key tiles are BK64/BK128"
    assert num_queries % block_M == 0, "query capacity must be block aligned"
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

    Q: T.Tensor([batch, seq_len, heads, D], dtype)  # type: ignore
    K: T.Tensor([batch, seq_len_kv, kv_group, D], dtype)  # type: ignore
    V: T.Tensor([batch, seq_len_kv, kv_group, D], dtype)  # type: ignore
    QueryIndices: T.Tensor([batch, num_queries], index_dtype)  # type: ignore
    QueryCount: T.Tensor([batch], index_dtype)  # type: ignore
    Output: T.Tensor([batch, seq_len, heads, D], dtype)  # type: ignore
    Lse: T.Tensor([batch, seq_len, heads], accum_dtype)  # type: ignore

    with T.Kernel(NM, batch, heads, threads=threads) as (bx, by, bz):
        Q_shared = T.alloc_shared([BM, D], dtype)
        K_shared = T.alloc_shared([BK, D], dtype)
        V_shared = T.alloc_shared([BK, D], dtype)
        P_shared = T.alloc_shared([BM, BK], dtype)
        O_shared = T.alloc_shared([BM, D], dtype)

        qpos = T.alloc_fragment([BM], index_dtype)
        valid_q = T.alloc_fragment([BM], "bool")
        acc_s = T.alloc_fragment([BM, BK], accum_dtype)
        acc_o = T.alloc_fragment([BM, D], accum_dtype)
        row_max = T.alloc_fragment([BM], accum_dtype)
        row_max_prev = T.alloc_fragment([BM], accum_dtype)
        row_sum = T.alloc_fragment([BM], accum_dtype)
        row_sum_block = T.alloc_fragment([BM], accum_dtype)
        alpha = T.alloc_fragment([BM], accum_dtype)

        b_i = by
        h_i = bz
        kv_i = h_i // heads_per_kv
        q0 = bx * BM
        count = QueryCount[b_i]

        # Every real list is sorted.  The first and last valid positions determine
        # the union of key tiles needed by this CTA; per-row masking below preserves
        # exact causal/window semantics inside that union.
        first_slot = T.min(q0, T.max(count - 1, 0))
        last_slot = T.min(q0 + BM - 1, T.max(count - 1, 0))
        first_qpos = T.max(QueryIndices[b_i, first_slot], 0)
        last_qpos = T.max(QueryIndices[b_i, last_slot], 0)

        first_key = 0
        if is_local:
            first_key = T.max(first_qpos - window_size + 1, 0)
        first_block = first_key // BK
        last_block = T.ceildiv(T.min(last_qpos + 1, seq_len_kv), BK)
        num_key_blocks = T.max(last_block - first_block, 0)

        for mi in T.Parallel(BM):
            slot = q0 + mi
            valid_q[mi] = slot < count
            qpos[mi] = T.if_then_else(
                valid_q[mi], QueryIndices[b_i, slot], 0
            )

        # Direct indexed Q gather avoids a separate compact-Q allocation/copy.
        # Keep every row copy on the explicit non-descriptor path: QueryIndices
        # makes this a gather, and a 128K tensor's large outer strides are not valid
        # for the compiler's otherwise tempting BMxD descriptor recognition.
        for mi in T.serial(BM):
            T.copy(
                Q[
                    b_i,
                    T.max(0, T.min(qpos[mi], seq_len - 1)),
                    h_i,
                    :,
                ],
                Q_shared[mi, :],
                disable_tma=True,
            )

        T.fill(acc_o, 0)
        T.fill(row_sum, 0)
        T.fill(row_max, -(2**30))

        for jb in T.Pipelined(num_key_blocks, num_stages=num_stages):
            key_block = first_block + jb
            key0 = key_block * BK
            # The public wrapper requires S % BK == 0, so every visited key tile is
            # physically complete.  Explicit non-descriptor copies avoid the same
            # large-stride descriptor issue as the indexed Q gather.
            T.copy(
                K[b_i, key0 : key0 + BK, kv_i, :],
                K_shared,
                disable_tma=True,
            )
            for mi, ni in T.Parallel(BM, BK):
                key_pos = key0 + ni
                lower = 0
                if is_local:
                    lower = T.max(qpos[mi] - window_size + 1, 0)
                legal = (
                    valid_q[mi]
                    and (key_pos < seq_len_kv)
                    and (key_pos >= lower)
                    and (key_pos <= qpos[mi])
                )
                acc_s[mi, ni] = T.if_then_else(
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
            for mi in T.Parallel(BM):
                row_max[mi] = T.max(row_max[mi], row_max_prev[mi])
                alpha[mi] = T.exp2(
                    (row_max_prev[mi] - row_max[mi]) * sm_scale
                )
            for mi, ni in T.Parallel(BM, BK):
                acc_s[mi, ni] = T.exp2(
                    acc_s[mi, ni] * sm_scale
                    - row_max[mi] * sm_scale
                )
            T.reduce_sum(acc_s, row_sum_block, dim=1)
            for mi in T.Parallel(BM):
                row_sum[mi] = row_sum[mi] * alpha[mi] + row_sum_block[mi]
            for mi, d_i in T.Parallel(BM, D):
                acc_o[mi, d_i] *= alpha[mi]

            T.copy(acc_s, P_shared)
            T.copy(
                V[b_i, key0 : key0 + BK, kv_i, :],
                V_shared,
                disable_tma=True,
            )
            T.wgmma_gemm(
                P_shared,
                V_shared,
                acc_o,
                policy=T.GemmWarpPolicy.FullRow,
            )
            T.wait_wgmma(0)

        for mi, d_i in T.Parallel(BM, D):
            acc_o[mi, d_i] /= row_sum[mi]
        for mi in T.Parallel(BM):
            row_sum[mi] = (
                T.log2(row_sum[mi]) + row_max[mi] * sm_scale
            )

        T.copy(acc_o, O_shared)
        for mi, d_i in T.Parallel(BM, D):
            if valid_q[mi]:
                Output[b_i, qpos[mi], h_i, d_i] = O_shared[mi, d_i]
        for mi in T.Parallel(BM):
            if valid_q[mi]:
                Lse[b_i, qpos[mi], h_i] = row_sum[mi]
