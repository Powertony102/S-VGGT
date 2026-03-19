import torch
from typing import Tuple, Callable, Optional, Union


@torch.jit.script
def fast_similarity_chunks(
    a: torch.Tensor, b_transposed: torch.Tensor, chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:

    B, num_src, C = a.shape
    original_dtype = a.dtype

    # Convert to bf16 for computation to improve performance and reduce memory usage
    a_bf16 = a.to(torch.bfloat16)
    b_transposed_bf16 = b_transposed.to(torch.bfloat16)
    node_max = torch.empty(B, num_src, device=a.device, dtype=original_dtype)
    node_idx = torch.empty(B, num_src, device=a.device, dtype=torch.long)

    # Process in chunks
    num_dst = b_transposed_bf16.shape[-1]
    chunk_size_src = min(chunk_size, num_src)
    chunk_size_dst = max(1024, min(4096, num_dst))

    for i in range(0, num_src, chunk_size_src):
        end_i = min(i + chunk_size_src, num_src)
        a_chunk = a_bf16[:, i:end_i, :]
        chunk_max = torch.full((B, end_i - i), -1e9, device=a.device, dtype=a_bf16.dtype)
        chunk_arg = torch.zeros(B, end_i - i, device=a.device, dtype=torch.long)
        for j in range(0, num_dst, chunk_size_dst):
            end_j = min(j + chunk_size_dst, num_dst)
            b_chunk = b_transposed_bf16[:, :, j:end_j]
            scores = torch.bmm(a_chunk, b_chunk)
            max_j, idx_j = torch.max(scores, dim=2)
            better = max_j > chunk_max
            chunk_max = torch.where(better, max_j, chunk_max)
            chunk_arg = torch.where(better, idx_j + j, chunk_arg)
        node_max[:, i:end_i] = chunk_max.to(original_dtype)
        node_idx[:, i:end_i] = chunk_arg
    return node_max, node_idx


def do_nothing(
    x: torch.Tensor,
    extra_tensors=None,
    extra_tensors_2=None,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    if extra_tensors is not None and extra_tensors_2 is not None:
        return x, extra_tensors, extra_tensors_2
    elif extra_tensors is not None:
        return x, extra_tensors
    else:
        return x


def token_merge_bipartite2d(
    metric: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    no_rand: bool = False,
    generator: Optional[torch.Generator] = None,
    enable_protection: bool = False,
    ) -> Tuple[Callable, Callable]:
    """
    Divide tokens into source (src) and destination (dst) groups, and merge r tokens from src to dst.
    dst tokens are selected by randomly choosing one token from each (sx, sy) region.
    Optionally protect the top 10% of tokens from merging based on importance scores.

    Args:
     - metric [B, N, C]: Tensor for similarity computation, B=batch size, N=token count, C=feature dimension
     - w: Image width in tokens
     - h: Image height in tokens
     - sx: dst stride in x dimension, must divide w evenly
     - sy: dst stride in y dimension, must divide h evenly
     - r: Number of tokens to remove through merging
     - no_rand: If True, disable randomness (use only top-left token)
     - generator: Random number generator if no_rand is False and not None
     - enable_protection: If True, enable importance protection feature

    Returns:
     - (merge, unmerge): Two functions for merging tokens and restoring pre-merge state
    """
    # 计算批量大小与序列长度；metric 的最后一维是特征维 C（未使用）
    B, N, _ = metric.shape  # Batch size B, total tokens N
    # 若需要合并的数量 r <= 0，直接返回空操作的闭包，避免无谓计算
    if r <= 0:
        return do_nothing, do_nothing

    # 简写 gather，后续大量使用按索引收集的操作
    gather = torch.gather

    # 每张图像的 token 数：patch 网格 w*h 加上 5 个特殊 token（camera/register）
    tokens_per_img = w * h + 5
    # 图像总数（多帧）：要求整除
    num_imgs = N // tokens_per_img
    assert tokens_per_img * num_imgs == N, "Token count doesn't match (w*h+5)*num_imgs"

    # 整个匹配与分组过程不需要梯度，且可减少显存占用
    with torch.no_grad():
        # 是否启用保护：当前实现为均匀抽样 N 的 10% 作为 protected 索引，防止被合并
        if enable_protection:
            num_protected = int(N * 0.1)
            step = max(1, N // num_protected)
            # 均匀步进采样出受保护的索引列表
            protected_indices = torch.arange(0, N, step, device=metric.device)[
                :num_protected
            ]
        else:
            protected_indices = None
            num_protected = 0

        # idx_buffer_seq：长度为 N 的标记向量；-1 表示 dst，0 表示 src
        idx_buffer_seq = torch.zeros(N, device=metric.device, dtype=torch.int64)
        # hsy、wsx：分别是以 sy、sx 网格步幅划分后，每张图上的网格块数量（高、宽方向）
        hsy, wsx = h // sy, w // sx  # Number of blocks within each image

        # 第一张图像（含 5 个特殊 token + 所有 patch）全部作为 dst 聚合目标
        if num_imgs > 0:
            idx_buffer_seq[:tokens_per_img] = -1

        # 后续图像的处理（批量化）：
        # 1) 每张图的 5 个特殊 token 标记为 dst
        # 2) 在每个 (sy, sx) 网格中选择 1 个 token 为 dst（其余默认 src）
        if num_imgs > 1:
            # 计算所有其它图像的 5 个特殊 token 的全局索引，并标记为 dst
            cls_indices = (
                torch.arange(1, num_imgs, device=metric.device) * tokens_per_img
            )
            cls_indices = cls_indices[:, None] + torch.arange(5, device=metric.device)
            idx_buffer_seq[cls_indices.flatten()] = -1
            del cls_indices
            # 有效网格尺寸（避免越界）：当 h 或 w 不能被 sy/sx 完整整除时，按可覆盖区域截断
            effective_h = min(hsy * sy, h)
            effective_w = min(wsx * sx, w)
            effective_grid_size = effective_h * effective_w

            if no_rand:
                # 非随机：每个网格只选择固定（左上）位置为 dst
                base_pattern = torch.zeros(
                    effective_grid_size, device=metric.device, dtype=torch.int64
                )
                # 每张图像的网格起始偏移（跳过 5 个特殊 token）
                grid_starts = (
                    torch.arange(1, num_imgs, device=metric.device) * tokens_per_img + 5
                )
                grid_indices = grid_starts[:, None] + torch.arange(
                    effective_grid_size, device=metric.device
                )
                # 将所有其它图像对应位置标记为 dst（-1），但 base_pattern 为 0，此处仍保持 src 标记；
                # 因为非随机时，默认只保留网格中的第一个元素为 dst（见下方随机分支对应的 scatter 逻辑）
                idx_buffer_seq[grid_indices.flatten()] = base_pattern.repeat(
                    num_imgs - 1
                )
                del base_pattern, grid_starts, grid_indices
            else:
                # 随机：在每个 (sy, sx) 子网格里随机选一个索引作为 dst
                total_other_imgs = num_imgs - 1
                all_rand_idx = torch.randint(
                    sy * sx,
                    size=(total_other_imgs, hsy, wsx),
                    device=metric.device,
                    generator=generator,
                )

                # scatter_src 为 -1，用于把被选中的网格位置标记为 dst
                scatter_src = -torch.ones(
                    total_other_imgs, hsy, wsx, device=metric.device, dtype=torch.int64
                )

                # idx_buffer_batch 的最后一维大小为 sy*sx，代表每个网格内的所有位置；
                # 先在该维度上做 scatter，把随机选中的位置置为 -1（dst），其余保持 0（src）
                idx_buffer_batch = torch.zeros(
                    total_other_imgs,
                    hsy,
                    wsx,
                    sy * sx,
                    device=metric.device,
                    dtype=torch.int64,
                )
                idx_buffer_batch.scatter_(
                    dim=3,
                    index=all_rand_idx.unsqueeze(-1),
                    src=scatter_src.unsqueeze(-1),
                )

                # 将每个网格的 sy*sx 展开成二维平面，并按 (H, W) 的顺序重排（transpose + reshape）
                idx_buffer_batch = (
                    idx_buffer_batch.view(total_other_imgs, hsy, wsx, sy, sx)
                    .transpose(2, 3)
                    .reshape(total_other_imgs, hsy * sy, wsx * sx)
                )

                # 将每张图对应的二维标记平面（-1 或 0）批量写入到全局 idx_buffer_seq 中对应的 patch 区域
                # 注意 grid_start 跳过了前 5 个特殊 token
                for i in range(total_other_imgs):
                    img_idx = i + 1
                    grid_start = img_idx * tokens_per_img + 5
                    flat_view = idx_buffer_batch[
                        i, :effective_h, :effective_w
                    ].flatten()
                    idx_buffer_seq[grid_start : grid_start + effective_grid_size] = (
                        flat_view
                    )
                del idx_buffer_batch, all_rand_idx, scatter_src

        # 通过 argsort 将 -1（dst）排在前面，0（src）排在后面，得到两侧的索引集合
        rand_idx = idx_buffer_seq.reshape(1, -1, 1).argsort(dim=1)
        num_dst_orig = int((idx_buffer_seq == -1).sum())

        # 原始 src 与 dst 的索引切分（形状 [1, K, 1]）
        a_idx_orig = rand_idx[:, num_dst_orig:, :]
        b_idx_orig = rand_idx[:, :num_dst_orig, :]
        a_idx = a_idx_orig
        b_idx = b_idx_orig
        del rand_idx, idx_buffer_seq

        # 若启用保护，构造受保护索引的形状以适配 gather（[1, P, 1]）
        if enable_protection:
            protected_idx = protected_indices.unsqueeze(0).unsqueeze(-1)
            num_protected_actual = protected_idx.shape[1]
        else:
            protected_idx = None
            num_protected_actual = 0

        # 记录两侧的数量，便于后续分割与 scatter/gather
        num_src = a_idx.shape[1]
        num_dst = b_idx.shape[1]

        # 内部辅助函数：根据 a_idx/b_idx/protected_idx 将输入 x 分割为 src/dst/(protected)
        def split(x):
            C = x.shape[-1]

            if enable_protection:
                src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                protected = gather(
                    x, dim=1, index=protected_idx.expand(B, num_protected_actual, C)
                )
                return src, dst, protected
            else:
                src = gather(x, dim=1, index=a_idx.expand(B, num_src, C))
                dst = gather(x, dim=1, index=b_idx.expand(B, num_dst, C))
                return src, dst

        # 计算余弦相似度：先做 L2 归一化，再用点积近似余弦
        metric = metric / metric.norm(dim=-1, keepdim=True)
        if enable_protection:
            a, b, protected = split(metric)
        else:
            a, b = split(metric)

        # 合并数量 r 不得超过当前 src 的实际数量
        r = min(a.shape[1], r)
        num_src_actual = a.shape[1]
        # 分块大小上限 5000，避免一次性计算大规模 bmm 造成显存峰值
        chunk_size = min(5000, num_src_actual)

        # 创建未初始化的张量用于存储每个 src 的最大相似度与对应的 dst 索引
        node_max = torch.empty(B, num_src_actual, device=a.device, dtype=a.dtype)
        node_idx = torch.empty(B, num_src_actual, device=a.device, dtype=torch.long)

        # 将 dst 转置为 [B, C, num_dst]，用于 a 与 bmm 计算
        b_transposed = b.transpose(-1, -2)
        # 分块计算每个 src 的最大相似度与 argmax dst
        node_max, node_idx = fast_similarity_chunks(a, b_transposed, chunk_size)
        # 按最大相似度由大到小排序，得到 src 的优先级队列（edge_idx）
        edge_idx = node_max.argsort(dim=-1, descending=True)[..., None]
        del node_max, b_transposed

        # 若启用保护：过滤掉属于保护集合的 src，保证这些 token 不被合并
        if enable_protection:
            src_indices = a_idx[0, :, 0]
            protected_mask_src = torch.isin(src_indices, protected_indices)
            edge_flat = edge_idx[0, :, 0]
            # 仅保留未被保护的边
            valid_mask = ~protected_mask_src[edge_flat]
            valid_edges = edge_flat[valid_mask]

            valid_count = valid_edges.shape[0]
            r_actual = min(r, valid_count)

            # 根据 r_actual 切分：前 r_actual 为待合并 src，后面为未合并 src
            unm_idx = valid_edges[r_actual:].unsqueeze(0).unsqueeze(-1)
            src_idx = valid_edges[:r_actual].unsqueeze(0).unsqueeze(-1)
        else:
            # 无保护时，直接用 edge_idx 切分；前 r 为待合并 src，其余为未合并 src
            unm_idx = edge_idx[..., r:, :]
            src_idx = edge_idx[..., :r, :]
            r_actual = r
        del edge_idx
        if enable_protection:
            # 保护集合索引不再需要
            del protected_indices

        # 为每个待合并的 src 查出它指向的 argmax dst 索引
        dst_idx = gather(node_idx[..., None], dim=-2, index=src_idx)
        # 用 r_actual 回写 r，便于后续一致性（extra_tensors 同步）
        r = r_actual
        del node_idx
        # 大型中间张量及时释放
        if enable_protection:
            del protected
        del a, b, metric
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # 定义合并函数：把选中的 src 汇聚到对应的 dst（默认 reduce=mean），并返回拼接结果
    def merge(
        x: torch.Tensor,
        mode: str = "mean",
        extra_tensors=None,
        extra_tensors_2=None,
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        # 按与 metric 相同的分割策略拆分 x
        if enable_protection:
            src, dst, protected = split(x)
        else:
            src, dst = split(x)

        n, t1, c = src.shape

        # 未合并的 src：根据 unm_idx 取出保留的 src 片段
        unm_len = unm_idx.shape[1]
        unm = gather(src, dim=-2, index=unm_idx.expand(n, unm_len, c))
        # 待合并的 src：根据 src_idx 选出前 r 个
        src_len = src_idx.shape[1]
        src = gather(src, dim=-2, index=src_idx.expand(n, src_len, c))
        # 将待合并的 src 聚合到对应的 dst；mode 默认为 mean，可改为 sum 等
        dst = dst.scatter_reduce(-2, dst_idx.expand(n, src_len, c), src, reduce=mode)

        # ---------------- 额外张量（如 k、v）与主张量保持一致的聚合 ----------------
        merged_extra_1 = None
        merged_extra_2 = None
        if extra_tensors is not None:
            E_dim = extra_tensors.shape[-1]
            if enable_protection:
                src_e, dst_e, protected_e = split(extra_tensors)
            else:
                src_e, dst_e = split(extra_tensors)

            # 与主张量一致：仅对前 r 个 src 进行合并
            src_e_r = gather(src_e, dim=-2, index=src_idx.expand(n, src_len, E_dim))
            unm_e = gather(src_e, dim=-2, index=unm_idx.expand(n, unm_len, E_dim))

            dst_e = dst_e.scatter_reduce(
                -2, dst_idx.expand(n, src_len, E_dim), src_e_r, reduce=mode
            )
            if enable_protection:
                merged_extra_1 = torch.cat([unm_e, dst_e, protected_e], dim=1)
            else:
                merged_extra_1 = torch.cat([unm_e, dst_e], dim=1)

        if extra_tensors_2 is not None:
            E_dim_2 = extra_tensors_2.shape[-1]
            if enable_protection:
                src_e2, dst_e2, protected_e2 = split(extra_tensors_2)
            else:
                src_e2, dst_e2 = split(extra_tensors_2)

            src_e2_r = gather(src_e2, dim=-2, index=src_idx.expand(n, src_len, E_dim_2))
            unm_e2 = gather(src_e2, dim=-2, index=unm_idx.expand(n, unm_len, E_dim_2))

            dst_e2 = dst_e2.scatter_reduce(
                -2, dst_idx.expand(n, src_len, E_dim_2), src_e2_r, reduce=mode
            )
            if enable_protection:
                merged_extra_2 = torch.cat([unm_e2, dst_e2, protected_e2], dim=1)
            else:
                merged_extra_2 = torch.cat([unm_e2, dst_e2], dim=1)

        # 主结果拼接：若启用保护，顺序为 [unm, dst, protected]；否则为 [unm, dst]
        if enable_protection:
            main_result = torch.cat([unm, dst, protected], dim=1)
        else:
            main_result = torch.cat([unm, dst], dim=1)

        # 调试输出：每次 merge 调用后剩余的 token 数量
        try:
            protected_len = protected.shape[1] if enable_protection else 0
        except NameError:
            protected_len = 0
        print(
            f"[merge] tokens: before={N}, after={main_result.shape[1]}, "
            f"unm={unm.shape[1]}, dst={dst.shape[1]}, protected={protected_len}, merged_src={src_len}"
        )

        # 返回主结果与（可选）额外张量的合并结果，保持 q/k/v 对齐
        if merged_extra_1 is not None and merged_extra_2 is not None:
            return main_result, merged_extra_1, merged_extra_2
        elif merged_extra_1 is not None:
            return main_result, merged_extra_1
        else:
            return main_result

    # 定义反合并函数：将合并后的短序列恢复为原始顺序与长度（供注意力输出之后使用）
    def unmerge(x: torch.Tensor) -> torch.Tensor:
        # 先按顺序切分短序列：未合并 src（unm）、聚合后 dst、（可选）protected
        unm_len = unm_idx.shape[1]
        dst_len = num_dst
        src_len = src_idx.shape[1]
        unm = x[..., :unm_len, :]
        dst = x[..., unm_len : unm_len + dst_len, :]

        if enable_protection:
            protected = x[
                ..., unm_len + dst_len : unm_len + dst_len + num_protected_actual, :
            ]

        # 用 dst 的聚合结果反向还原被合并的 src：根据 dst_idx 把 dst 拷回合适位置
        _, _, c = unm.shape
        src = gather(dst, dim=-2, index=dst_idx.expand(B, src_len, c))
        # 准备完整输出，并逐块 scatter 到原始索引位置
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        # 1) 把 dst 放回原来的 b_idx（目标侧）位置
        out.scatter_(dim=-2, index=b_idx.expand(B, num_dst, c), src=dst)
        # 2) 把未合并的 src 放回 a_idx 对应的 unm_idx 位置
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=unm_idx
            ).expand(B, unm_len, c),
            src=unm,
        )

        # 3) 把被合并的 src（从 dst 反向恢复出来的片段）放回 a_idx 对应的 src_idx 位置
        out.scatter_(
            dim=-2,
            index=gather(
                a_idx.expand(B, a_idx.shape[1], 1), dim=1, index=src_idx
            ).expand(B, src_len, c),
            src=src,
        )

        # 4) 若启用保护，将 protected 片段 scatter 回 protected_idx 对应位置
        if enable_protection:
            out.scatter_(
                dim=-2,
                index=protected_idx.expand(B, num_protected_actual, c),
                src=protected,
            )

        return out

    # 额外工具：根据当前分割索引，为位置编码 pos 构建与合并后序列对应的 pos_m（不做聚合，仅选择）
    def build_pos(pos: torch.Tensor) -> torch.Tensor:
        # pos: (B, N, Dpos) 通常 Dpos=2，dtype=long
        Dpos = pos.shape[-1]
        if enable_protection:
            src_p, dst_p, prot_p = split(pos)
        else:
            src_p, dst_p = split(pos)
            prot_p = None

        # 未合并的 src 的位置：按 unm_idx 选取
        n = src_p.shape[0]
        unm_len = unm_idx.shape[1]
        pos_unm = gather(src_p, dim=-2, index=unm_idx.expand(n, unm_len, Dpos))

        if enable_protection:
            pos_m = torch.cat([pos_unm, dst_p, prot_p], dim=1)
        else:
            pos_m = torch.cat([pos_unm, dst_p], dim=1)
        return pos_m

    # 将工具函数附加到 merge 闭包，供调用方构造合并后位置编码
    merge.build_pos = build_pos  # type: ignore[attr-defined]

    # 返回两个闭包，供 Attention 在合并前后调用
    return merge, unmerge
