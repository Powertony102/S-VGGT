# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
from numpy import block
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Optional, Tuple, Union, List, Dict, Any

from vggt.layers import PatchEmbed
from vggt.layers.block import Block
from vggt.layers.rope import RotaryPositionEmbedding2D, PositionGetter
from vggt.layers.vision_transformer import vit_small, vit_base, vit_large, vit_giant2
import time

FLAG_FP16 = False
_FP16_DTYPE = torch.float16


def _maybe_fp16(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(_FP16_DTYPE) if FLAG_FP16 else tensor


logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """
    The Aggregator applies alternating-attention over input frames,
    as described in VGGT: Visual Geometry Grounded Transformer.

    Remember to set model.train() to enable gradient checkpointing to reduce memory usage.

    Args:
        img_size (int): Image size in pixels.
        patch_size (int): Size of each patch for PatchEmbed.
        embed_dim (int): Dimension of the token embeddings.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of MLP hidden dim to embedding dim.
        num_register_tokens (int): Number of register tokens.
        block_fn (nn.Module): The block type used for attention (Block by default).
        qkv_bias (bool): Whether to include bias in QKV projections.
        proj_bias (bool): Whether to include bias in the output projection.
        ffn_bias (bool): Whether to include bias in MLP layers.
        patch_embed (str): Type of patch embed. e.g., "conv" or "dinov2_vitl14_reg".
        aa_order (list[str]): The order of alternating attention, e.g. ["frame", "global"].
        aa_block_size (int): How many blocks to group under each attention type before switching. If not necessary, set to 1.
        qk_norm (bool): Whether to apply QK normalization.
        rope_freq (int): Base frequency for rotary embedding. -1 to disable.
        init_values (float): Init scale for layer scale.
    """

    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        num_register_tokens=4,
        block_fn=Block,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        patch_embed="dinov2_vitl14_reg",
        aa_order=["frame", "global"],
        aa_block_size=1,
        qk_norm=True,
        rope_freq=100,
        init_values=0.01,
        global_merging=False,
        merging=0,
        vis_attn_map=False,
    ):
        super().__init__()

        self.__build_patch_embed__(
            patch_embed, img_size, patch_size, num_register_tokens, embed_dim=embed_dim
        )

        # Initialize rotary position embedding if frequency > 0
        self.rope = (
            RotaryPositionEmbedding2D(frequency=rope_freq) if rope_freq > 0 else None
        )
        self.position_getter = PositionGetter() if self.rope is not None else None
        self.global_merging = global_merging
        self.merging = merging
        self.vis_attn_map = vis_attn_map
        self.frame_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.global_blocks = nn.ModuleList(
            [
                block_fn(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    proj_bias=proj_bias,
                    ffn_bias=ffn_bias,
                    init_values=init_values,
                    qk_norm=qk_norm,
                    rope=self.rope,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.aa_order = aa_order
        self.patch_size = patch_size
        self.aa_block_size = aa_block_size

        # Validate that depth is divisible by aa_block_size
        if self.depth % self.aa_block_size != 0:
            raise ValueError(
                f"depth ({depth}) must be divisible by aa_block_size ({aa_block_size})"
            )

        self.aa_block_num = self.depth // self.aa_block_size

        # Note: We have two camera tokens, one for the first frame and one for the rest
        # The same applies for register tokens
        self.camera_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(
            torch.randn(1, 2, num_register_tokens, embed_dim)
        )

        # The patch tokens start after the camera and register tokens
        self.patch_start_idx = 1 + num_register_tokens

        # Initialize parameters with small values
        nn.init.normal_(self.camera_token, std=1e-6)
        nn.init.normal_(self.register_token, std=1e-6)

        # Register normalization constants as buffers (use bf16-compatible tensor)
        for name, value in (
            ("_resnet_mean", _RESNET_MEAN),
            ("_resnet_std", _RESNET_STD),
        ):
            self.register_buffer(
                name,
                torch.tensor(
                    value, dtype=_FP16_DTYPE if FLAG_FP16 else torch.float32
                ).view(1, 1, 3, 1, 1),
                persistent=False,
            )

        self.use_reentrant = False  # hardcoded to False

    def __build_patch_embed__(
        self,
        patch_embed,
        img_size,
        patch_size,
        num_register_tokens,
        interpolate_antialias=True,
        interpolate_offset=0.0,
        block_chunks=0,
        init_values=1.0,
        embed_dim=1024,
    ):
        """
        Build the patch embed layer. If 'conv', we use a
        simple PatchEmbed conv layer. Otherwise, we use a vision transformer.
        """

        if "conv" in patch_embed:
            self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=3,
                embed_dim=embed_dim,
            )
        else:
            vit_models = {
                "dinov2_vitl14_reg": vit_large,
                "dinov2_vitb14_reg": vit_base,
                "dinov2_vits14_reg": vit_small,
                "dinov2_vitg2_reg": vit_giant2,
            }

            self.patch_embed = vit_models[patch_embed](
                img_size=img_size,
                patch_size=patch_size,
                num_register_tokens=num_register_tokens,
                interpolate_antialias=interpolate_antialias,
                interpolate_offset=interpolate_offset,
                block_chunks=block_chunks,
                init_values=init_values,
            )

            # Disable gradient updates for mask token
            if hasattr(self.patch_embed, "mask_token"):
                self.patch_embed.mask_token.requires_grad_(False)

    def forward(
        self,
        images: torch.Tensor = None,
        patch_tokens_bspc: torch.Tensor = None,
        hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[torch.Tensor], int]:
        """
        Args:
            images (torch.Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width

        Returns:
            (list[torch.Tensor], int):
                The list of outputs from the attention blocks,
                and the patch_start_idx indicating where patch tokens begin.
        """
        if patch_tokens_bspc is not None:
            # Use precomputed patch tokens with shape (B, S, P, C)
            if images is None and hw is None:
                raise ValueError("When providing patch_tokens_bspc, please also pass hw=(H, W) if images is None.")
            B, S, Pp_in, C_in = patch_tokens_bspc.shape
            # H, W from provided images or hw
            if images is not None:
                _, _, _, H, W = images.shape
            else:
                H, W = hw
            patch_tokens = patch_tokens_bspc.view(B * S, Pp_in, C_in)
            patch_tokens = _maybe_fp16(patch_tokens)
        else:
            B, S, C_in, H, W = images.shape

            if C_in != 3:
                raise ValueError(f"Expected 3 input channels, got {C_in}")

            # Normalize images and reshape for patch embed - optionally switch to fp16/bf16
            # 这里强行开了Fp16模式
            images = _maybe_fp16(images)
            images = (images - self._resnet_mean) / self._resnet_std

            images = images.view(B * S, C_in, H, W)
            patch_tokens = self.patch_embed(images)
            del images

            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens["x_norm_patchtokens"]

            patch_tokens = _maybe_fp16(patch_tokens)

        _, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(_maybe_fp16(self.camera_token), B, S)
        register_token = slice_expand_and_flatten(_maybe_fp16(self.register_token), B, S)

        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        tokens = _maybe_fp16(tokens)
        del camera_token, register_token, patch_tokens
        # Explicitly clean up image data since patch embedding is complete
        if "images_normalized" in locals():
            del images_normalized

        pos = None
        if self.rope is not None:
            pos = self.position_getter(
                B * S, H // self.patch_size, W // self.patch_size, device="cuda"
            )

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos_original = pos
            pos = pos + 1
            pos_special = torch.zeros(
                B * S, self.patch_start_idx, 2, device="cuda", dtype=torch.long
            )
            pos = torch.cat([pos_special, pos], dim=1)
            # Clean up temporary variables
            del pos_special, pos_original

        # update P because we added special tokens
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []
        block4DPT_idx = [4, 11, 17, 23]
        global_merging = None

        # Set global variables for attention visualization
        if self.vis_attn_map:
            import vggt.layers.attention as attn_module

            # Set the global variables that attention.py needs
            attn_module.vis_attn_map = True
            attn_module.current_images = self._load_image_paths()  # Load from temp file
        else:
            import vggt.layers.attention as attn_module

            attn_module.vis_attn_map = False

        frame_attn_time_ms = 0.0
        global_attn_time_ms = 0.0
        for block_num in range(self.aa_block_num):
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            need_intermediates = True if block_num in block4DPT_idx else False
            if block_num % 1 == 0:
                # Clean up RoPE cache to prevent accumulation
                if hasattr(self, "rope") and self.rope is not None:
                    if hasattr(self.rope, "frequency_cache"):
                        self.rope.frequency_cache.clear()
                # Clean up position cache
                if (
                    hasattr(self, "position_getter")
                    and self.position_getter is not None
                ):
                    if hasattr(self.position_getter, "position_cache"):
                        # Keep only current size cache, clean up others
                        current_cache = self.position_getter.position_cache.copy()
                        if (
                            len(current_cache) > 1
                        ):  # If there are multiple cache entries
                            self.position_getter.position_cache.clear()
                            # Keep only the most recently used one
                            if current_cache:
                                key = list(current_cache.keys())[-1]
                                self.position_getter.position_cache[key] = (
                                    current_cache[key]
                                )
            # Avoid saving block_num to instance variable to reduce references
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    _start_evt = torch.cuda.Event(enable_timing=True)
                    _end_evt = torch.cuda.Event(enable_timing=True)
                    _start_evt.record()
                    tokens, frame_idx, frame_intermediates = (
                        self._process_frame_attention(
                            tokens,
                            B,
                            S,
                            P,
                            C,
                            frame_idx,
                            pos=pos,
                            need_intermediates=need_intermediates,
                        )
                    )
                    _end_evt.record()
                    _end_evt.synchronize()
                    frame_attn_time_ms += _start_evt.elapsed_time(_end_evt)
                elif attn_type == "global":
                    gm = None
                    if (
                        self.global_merging
                        and self.merging is not None
                        and block_num >= self.merging
                    ):
                        gm = block_num
                        if self.vis_attn_map:
                            import vggt.layers.attention as attn_module
                            attn_module.attention_map = block_num
                    _start_evt = torch.cuda.Event(enable_timing=True)
                    _end_evt = torch.cuda.Event(enable_timing=True)
                    _start_evt.record()
                    tokens, global_idx, global_intermediates = (
                        self._process_global_attention(
                            tokens,
                            B,
                            S,
                            P,
                            C,
                            global_idx,
                            pos=pos,
                            global_merging=gm,
                            need_intermediates=need_intermediates,
                        )
                    )
                    _end_evt.record()
                    _end_evt.synchronize()
                    global_attn_time_ms += _start_evt.elapsed_time(_end_evt)
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            if block_num not in block4DPT_idx:
                if "frame_intermediates" in locals():
                    del frame_intermediates
                if "global_intermediates" in locals():
                    del global_intermediates
            else:
                concat_inter = torch.cat(
                    [frame_intermediates[0].detach(), global_intermediates[0].detach()],
                    dim=-1,
                )
                if FLAG_FP16 and concat_inter.dtype != _FP16_DTYPE:
                    concat_inter = concat_inter.to(_FP16_DTYPE)
                output_list.append(concat_inter)
                del concat_inter, frame_intermediates, global_intermediates

        # Do final cleanup before returning
        del tokens, pos
        if "pos_special" in locals():
            del pos_special
        if "pos_original" in locals():
            del pos_original
        torch.cuda.empty_cache()
        print(f"Frame Attention total time: {frame_attn_time_ms/1000.0:.4f}s")
        print(f"Global Attention total time: {global_attn_time_ms/1000.0:.4f}s")

        return output_list, self.patch_start_idx

    def _process_frame_attention(
        self, tokens, B, S, P, C, frame_idx, pos=None, need_intermediates=False
    ):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = [] if need_intermediates else None

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.frame_blocks[frame_idx](tokens, pos=pos)
            frame_idx += 1
            if need_intermediates:
                intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates

    def _process_global_attention(
        self,
        tokens,
        B,
        S,
        P,
        C,
        global_idx,
        pos=None,
        global_merging=None,
        need_intermediates=False,
    ):
        """
        Process global attention blocks. We keep tokens in shape (B, S*P, C).
        """
        if tokens.shape != (B, S * P, C):
            tokens = tokens.view(B, S, P, C).view(B, S * P, C)

        if pos is not None and pos.shape != (B, S * P, 2):
            pos = pos.view(B, S, P, 2).view(B, S * P, 2)

        intermediates = [] if need_intermediates else None

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            tokens = self.global_blocks[global_idx](
                tokens,
                pos=pos,
                global_merging=global_merging,
            )
            global_idx += 1
            if need_intermediates:
                intermediates.append(tokens.view(B, S, P, C))

        return tokens, global_idx, intermediates

    def _load_image_paths(self):
        """Load image paths from temporary file for visualization"""
        try:
            import os
            import tempfile
            import pickle

            temp_dir = tempfile.gettempdir()
            image_paths_file = os.path.join(temp_dir, "vggt_image_paths.pkl")

            if os.path.exists(image_paths_file):
                with open(image_paths_file, "rb") as f:
                    return pickle.load(f)
            else:
                return []
        except Exception as e:
            print(f"Warning: Could not load image paths for visualization: {e}")
            return []


def slice_expand_and_flatten(token_tensor, B, S):
    """
    Processes specialized tokens with shape (1, 2, X, C) for multi-frame processing:
    1) Uses the first position (index=0) for the first frame only
    2) Uses the second position (index=1) for all remaining frames (S-1 frames)
    3) Expands both to match batch size B
    4) Concatenates to form (B, S, X, C) where each sequence has 1 first-position token
       followed by (S-1) second-position tokens
    5) Flattens to (B*S, X, C) for processing

    Returns:
        torch.Tensor: Processed tokens with shape (B*S, X, C)
    """

    query = token_tensor[:, 0:1, ...].expand(B, 1, *token_tensor.shape[2:])
    # Slice out the "other" tokens => shape (1, S-1, ...)
    others = token_tensor[:, 1:, ...].expand(B, S - 1, *token_tensor.shape[2:])
    # Concatenate => shape (B, S, ...)
    combined = torch.cat([query, others], dim=1)

    # Finally flatten => shape (B*S, ...)
    combined = combined.view(B * S, *combined.shape[2:])
    return combined


import time
from typing import Any, Dict, List

import torch

# Debug controls for scene partition optimization (single switch)
# Set to True to enable step logs, early-stop notice, and group index summaries
PARTITION_DEBUG: bool = False
# Output limits when debug is enabled (not flags, just verbosity caps)
PARTITION_DEBUG_MAX_GROUPS: int = 16
PARTITION_DEBUG_MAX_INDICES: int = 64


def optimize_scene_partitions(
    similarity: torch.Tensor,
    num_groups: int = 5,
    steps: int = 2000,
    lr: float = 1e-1,
    lam: float = 0.5,
    alpha: float = 1.0,
    beta: float = 1.0,
    patience: int = 10,
    log_interval: int = 50,
) -> List[Dict[str, Any]]:
    """
    Distribution-preserving version:
    每个子场景的“相似性画像”都去对齐全局画像，而不是把相似帧全挤到一个组里。

    Args 与返回值保持不变。
    lam: distribution-matching 的权重
    alpha: coverage 约束权重（每帧行和≈1）
    beta: group-size balance 权重
    """

    if num_groups <= 0:
        raise ValueError("num_groups must be a positive integer")

    if similarity.dim() == 2:
        similarity = similarity.unsqueeze(0)

    results: List[Dict[str, Any]] = []
    B = similarity.shape[0]

    for batch_idx in range(B):
        S_mat = similarity[batch_idx].detach()
        S_mat = S_mat.clamp(0.0, 1.0)
        N = S_mat.shape[0]
        device = S_mat.device

        # learnable logits
        W = torch.randn(N, num_groups, device=device, requires_grad=True)
        opt = torch.optim.Adam([W], lr=lr)

        # timing
        use_cuda_ev = device.type == "cuda" and torch.cuda.is_available()
        if use_cuda_ev:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device=device)
            start_evt.record()
        else:
            start_time = time.perf_counter()

        # 预先算好全局 profile: g = (1/N) S 1
        ones = torch.ones(N, 1, device=device)
        global_profile = (S_mat @ ones) / float(N)      # (N, 1)

        best_loss = float("inf")
        wait = 0
        last_step = 0

        eps = 1e-6

        for step in range(steps):
            # soft assignment A: (N, K)
            A = torch.softmax(W, dim=1)

            # group-level profiles: P = S @ A  -> (N, K)
            # 每一列是该组与所有帧的相似度和，再除以该组的总权重做归一化
            group_mass = A.sum(dim=0, keepdim=True)                  # (1, K)
            group_profiles = S_mat @ A                               # (N, K)
            group_profiles = group_profiles / (group_mass + eps)     # (N, K)

            # global_profile: (N, 1) -> broadcast to (N, K)
            diff = group_profiles - global_profile                  # (N, K)
            L_dist = (diff ** 2).sum()

            # coverage: 每一帧的分配和应该≈1
            row_sum = A.sum(dim=1)                                   # (N,)
            L_cover = ((row_sum - 1.0) ** 2).sum()

            # balance: 每个group的总量≈N/K
            col_sum = A.sum(dim=0)                                   # (K,)
            target = N / num_groups
            L_balance = ((col_sum - target) ** 2).sum()

            loss = lam * L_dist + alpha * L_cover + beta * L_balance

            opt.zero_grad()
            loss.backward()
            opt.step()

            # Optional debugging logs
            dbg = PARTITION_DEBUG
            if dbg and (step == 0 or (log_interval and step % log_interval == 0)):
                try:
                    print(
                        f"[Partition][b{batch_idx}] step={step} loss={loss.item():.6f} "
                        f"Ldist={L_dist.item():.6f} Lcover={L_cover.item():.6f} Lbal={L_balance.item():.6f}"
                    )
                except Exception:
                    pass

            # early stop
            if loss.item() + 1e-4 < best_loss:
                best_loss = loss.item()
                wait = 0
            else:
                wait += 1
            last_step = step + 1
            if wait >= patience:
                if PARTITION_DEBUG:
                    try:
                        print(f"[Partition][b{batch_idx}] early stop at step={last_step}, best_loss={best_loss:.6f}")
                    except Exception:
                        pass
                break

        # timing end
        if use_cuda_ev:
            torch.cuda.synchronize(device=device)
            end_evt.record()
            torch.cuda.synchronize(device=device)
            elapsed_ms = start_evt.elapsed_time(end_evt)
        else:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # -------- hard assignment 部分保持原样风格 --------
        with torch.no_grad():
            A = torch.softmax(W, dim=1)  # (N, K)

            # 为了让每组都不至于0个，还是按“先信心高的帧先分配”的策略
            base_target = max(1, N // num_groups)
            counts = torch.zeros(num_groups, dtype=torch.long, device=device)

            prefs = torch.argsort(A, dim=1, descending=True)  # 每帧对各组的优先级 (N, K)
            confid, _ = torch.max(A, dim=1)                   # 每帧的最大置信度 (N,)
            sorted_idx = torch.argsort(confid, descending=True)
            hard_assign = torch.full((N,), -1, dtype=torch.long, device=device)

            for idx_frame in sorted_idx.tolist():
                assigned = False
                # 尝试按该帧的偏好排序来塞，尽量保持每组有人
                for choice in prefs[idx_frame].tolist():
                    if counts[choice] < base_target:
                        hard_assign[idx_frame] = choice
                        counts[choice] += 1
                        assigned = True
                        break
                if not assigned:
                    # 全部达到base_target后，塞到当前最少的组
                    choice = torch.argmin(counts).item()
                    hard_assign[idx_frame] = choice
                    counts[choice] += 1

        # 收集每组索引并padding
        group_indices: List[torch.Tensor] = []
        group_lengths: List[int] = []
        for g in range(num_groups):
            idxs = (hard_assign == g).nonzero(as_tuple=True)[0]
            group_indices.append(idxs)
            group_lengths.append(int(idxs.numel()))

        Lg = max(group_lengths) if group_lengths else 0
        padded_group_indices: List[torch.Tensor] = []
        if Lg > 0:
            for idxs in group_indices:
                if idxs.numel() == 0:
                    # 空组就重复最后一帧
                    last_idx = torch.tensor([N - 1], dtype=torch.long, device=device)
                    padded = last_idx.repeat(Lg)
                else:
                    last_idx = idxs[-1]
                    if idxs.numel() < Lg:
                        pad = last_idx.repeat(Lg - idxs.numel())
                        padded = torch.cat([idxs, pad], dim=0)
                    else:
                        padded = idxs
                padded_group_indices.append(padded)
        else:
            for _ in range(num_groups):
                padded_group_indices.append(torch.zeros(0, dtype=torch.long, device=device))

        results.append(
            {
                "soft_assignment": A.detach(),
                "hard_assignment": hard_assign.detach(),
                "counts": counts.detach(),
                "group_indices": [g.detach() for g in group_indices],
                "group_indices_padded": [g.detach() for g in padded_group_indices],
                "Lg": Lg,
                "best_loss": best_loss,
                "steps": last_step,
                "elapsed_ms": elapsed_ms,
            }
        )

        # Optional group index summary printing
        if PARTITION_DEBUG:
            try:
                print(
                    f"[Partition][b{batch_idx}] steps={last_step}, Lg={Lg}, group_sizes={group_lengths}"
                )
                # Print per-group indices with caps to avoid excessive logs
                for g in range(min(num_groups, PARTITION_DEBUG_MAX_GROUPS)):
                    idxs = group_indices[g].detach().cpu().tolist()
                    shown = idxs[:PARTITION_DEBUG_MAX_INDICES]
                    suffix = " ..." if len(idxs) > PARTITION_DEBUG_MAX_INDICES else ""
                    print(f"  - g{g} ({len(idxs)}): {shown}{suffix}")
            except Exception:
                pass

    return results
