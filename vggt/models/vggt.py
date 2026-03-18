# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import time
from huggingface_hub import PyTorchModelHubMixin  # used for model hub

# Toggle all timing prints and synchronizations
TIMING_ENABLED = True

# Use the standard Aggregator; subscene orchestration happens here in VGGT
from vggt.models.aggregator import Aggregator, optimize_scene_partitions
from vggt.heads.camera_head import CameraHead
from vggt.heads.dpt_head import DPTHead
from vggt.heads.track_head import TrackHead


class VGGT(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        enable_camera=True,
        enable_point=False,
        enable_depth=True,
        enable_track=False,
        merging=0,
        vis_attn_map=False,
        similarity_temperature: float = 1.0,
    ):
        super().__init__()

        self.vis_attn_map = vis_attn_map
        # Temperature for sharpening similarity (smaller -> sharper). 1.0 means no change.
        self.similarity_temperature = float(similarity_temperature)

        self.aggregator = Aggregator(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            merging=merging,
            global_merging=(merging is not None),
            vis_attn_map=vis_attn_map,
        )

        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.point_head = (
            DPTHead(
                dim_in=2 * embed_dim,
                output_dim=4,
                activation="inv_log",
                conf_activation="expp1",
            )
            if enable_point
            else None
        )
        self.depth_head = (
            DPTHead(
                dim_in=2 * embed_dim,
                output_dim=2,
                activation="exp",
                conf_activation="expp1",
            )
            if enable_depth
            else None
        )
        self.track_head = (
            TrackHead(dim_in=2 * embed_dim, patch_size=patch_size)
            if enable_track
            else None
        )

    def update_patch_dimensions(self, patch_width: int, patch_height: int):
        """
        Update patch dimensions for all attention layers in the model

        Args:
            patch_width: Patch width (typically 37)
            patch_height: Patch height (typically 28)
        """

        def update_attention_in_module(module):
            for name, child in module.named_children():
                # Recursively update submodules
                update_attention_in_module(child)
                # If it is an attention layer, update its patch dimensions
                if hasattr(child, "patch_width") and hasattr(child, "patch_height"):
                    child.patch_width = patch_width
                    child.patch_height = patch_height

        # Update all attention layers in the aggregator
        update_attention_in_module(self.aggregator)

        # print(
        #     f"🔧 Updated model attention layer patch dimensions: {patch_width}x{patch_height}"
        # )

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor = None,
        image_paths: list = None,
        num_groups: int | None = None,
    ):
        """
        Forward pass of the VGGT model.

        Args:
            images (torch.Tensor): Input images with shape [S, 3, H, W] or [B, S, 3, H, W], in range [0, 1].
                B: batch size, S: sequence length, 3: RGB channels, H: height, W: width
            query_points (torch.Tensor, optional): Query points for tracking, in pixel coordinates.
                Shape: [N, 2] or [B, N, 2], where N is the number of query points.
                Default: None
            image_paths (list, optional): List of image file paths for attention visualization.
                Only used when vis_attn_map=True. Default: None

        Returns:
            dict: A dictionary containing the following predictions:
                - pose_enc (torch.Tensor): Camera pose encoding with shape [B, S, 9] (from the last iteration)
                - depth (torch.Tensor): Predicted depth maps with shape [B, S, H, W, 1]
                - depth_conf (torch.Tensor): Confidence scores for depth predictions with shape [B, S, H, W]
                - world_points (torch.Tensor): 3D world coordinates for each pixel with shape [B, S, H, W, 3]
                - world_points_conf (torch.Tensor): Confidence scores for world points with shape [B, S, H, W]
                - images (torch.Tensor): Original input images, preserved for visualization

                If query_points is provided, also includes:
                - track (torch.Tensor): Point tracks with shape [B, S, N, 2] (from the last iteration), in pixel coordinates
                - vis (torch.Tensor): Visibility scores for tracked points with shape [B, S, N]
                - conf (torch.Tensor): Confidence scores for tracked points with shape [B, S, N]
        """
        if num_groups is not None and num_groups != -1 and num_groups <= 0:
            raise ValueError("num_groups must be None, -1, or a positive integer")

        # If without batch dimension, add it
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if query_points is not None and len(query_points.shape) == 2:
            query_points = query_points.unsqueeze(0)

        # If num_groups is None, or sequence length is 1, run the original path.
        # Grouping is enabled only when num_groups is -1 (auto) or a positive integer.
        if num_groups is None or images.shape[1] <= 1:
            # Lightweight timing helpers (local scope)
            def _timer_start_n():
                if not TIMING_ENABLED:
                    return (None, None, False)
                if torch.cuda.is_available():
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize()
                    s.record()
                    return (s, e, True)
                else:
                    return (time.perf_counter(), None, False)

            def _timer_end_n(t, label: str):
                if not TIMING_ENABLED:
                    return
                if t[2]:
                    torch.cuda.synchronize()
                    t[1].record()
                    torch.cuda.synchronize()
                    ms = t[0].elapsed_time(t[1])
                    print(f"[Timing] {label}: {ms:.2f} ms (CUDA)")
                else:
                    ms = (time.perf_counter() - t[0]) * 1000.0
                    print(f"[Timing] {label}: {ms:.2f} ms (CPU)")

            t_agg = _timer_start_n()
            aggregated_tokens_list, patch_start_idx = self.aggregator(images)
            _timer_end_n(t_agg, "aggregator_forward")

            predictions = {}

            if self.camera_head is not None:
                t_cam = _timer_start_n()
                pose_enc_list = self.camera_head(aggregated_tokens_list)
                predictions["pose_enc"] = pose_enc_list[-1]
                predictions["pose_enc_list"] = pose_enc_list
                _timer_end_n(t_cam, "camera_head.forward")

            if self.depth_head is not None:
                t_depth = _timer_start_n()
                depth, depth_conf = self.depth_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf
                _timer_end_n(t_depth, "depth_head.forward")

            if self.point_head is not None:
                t_point = _timer_start_n()
                pts3d, pts3d_conf = self.point_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                )
                predictions["world_points"] = pts3d
                predictions["world_points_conf"] = pts3d_conf
                _timer_end_n(t_point, "point_head.forward")

            if self.track_head is not None and query_points is not None:
                t_track = _timer_start_n()
                track_list, vis, conf = self.track_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_start_idx=patch_start_idx,
                    query_points=query_points,
                )
                predictions["track"] = track_list[-1]
                predictions["vis"] = vis
                predictions["conf"] = conf
                _timer_end_n(t_track, "track_head.forward")

            if not self.training:
                predictions["images"] = images

            return predictions

        # Subscene grouping path
        B, S, C_in, H, W = images.shape if images.dim() == 5 else (images.unsqueeze(0).shape)

        # Timing helpers
        def _timer_start():
            if not TIMING_ENABLED:
                return (None, None, False)
            if torch.cuda.is_available():
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                s.record()
                return (s, e, True)
            else:
                return (time.perf_counter(), None, False)

        def _timer_end(t, label: str):
            if not TIMING_ENABLED:
                return
            if t[2]:
                torch.cuda.synchronize()
                t[1].record()
                torch.cuda.synchronize()
                ms = t[0].elapsed_time(t[1])
                print(f"[Timing] {label}: {ms:.2f} ms (CUDA)")
            else:
                ms = (time.perf_counter() - t[0]) * 1000.0
                print(f"[Timing] {label}: {ms:.2f} ms (CPU)")

        # Compute similarity using Aggregator's patch_embed on normalized images (exclude frame 0)
        t_sim = _timer_start()
        with torch.no_grad():
            # sub-step: normalization
            t_norm = _timer_start()
            images_norm = (images - self.aggregator._resnet_mean) / self.aggregator._resnet_std
            _timer_end(t_norm, "similarity.norm")

            # sub-step: reshape/view
            t_view = _timer_start()
            imgs_flat = images_norm.view(B * S, C_in, H, W)
            _timer_end(t_view, "similarity.view")

            # sub-step: patch embedding
            t_pe = _timer_start()
            patch_tokens = self.aggregator.patch_embed(imgs_flat)
            _timer_end(t_pe, "similarity.patch_embed")
            if isinstance(patch_tokens, dict):
                patch_tokens = patch_tokens.get("x_norm_patchtokens", patch_tokens)
            Pp, Cc = patch_tokens.shape[1], patch_tokens.shape[2]
            patches_bspc = patch_tokens.view(B, S, Pp, Cc)
            if S > 1:
                patches_sel = patches_bspc[:, 1:, :, :]
            else:
                patches_sel = patches_bspc[:, :1, :, :]
            pooled = patches_sel.mean(dim=2)
            pooled = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-6)
            similarity = pooled @ pooled.transpose(1, 2)
            # Optional sharpening via temperature -> convert to affinity and symmetrize
            tau = getattr(self, "similarity_temperature", 1.0)
            if tau is not None and tau > 0 and abs(tau - 1.0) > 1e-6:
                # Row-wise softmax with temperature, then symmetrize
                sim_row = torch.softmax(similarity / tau, dim=-1)
                similarity = 0.5 * (sim_row + sim_row.transpose(1, 2))
                # Encourage strong self-affinity
                Bx, Nx, _ = similarity.shape
                eye = torch.eye(Nx, device=similarity.device, dtype=similarity.dtype).unsqueeze(0).expand(Bx, -1, -1)
                similarity = similarity * (1 - eye) + eye
            del images_norm, imgs_flat, patch_tokens, patches_sel, pooled
            if "sim_row" in locals():
                del sim_row
            if "eye" in locals():
                del eye
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _timer_end(t_sim, "similarity_compute")

        # Compute average neighbors per frame above a similarity threshold
        # Default threshold set to 0.995 as requested
        with torch.no_grad():
            sim_thres = 0.995
            if similarity is not None and similarity.ndim == 3:
                Bx, Nx, _ = similarity.shape
                # Exclude self-similarity on the diagonal
                eye_mask = torch.eye(Nx, device=similarity.device, dtype=torch.bool).unsqueeze(0)
                above = (similarity > sim_thres) & (~eye_mask)
                # Count per frame, then average across frames and batch
                counts_per_frame = above.sum(dim=-1).float()
                avg_neighbors_per_frame = counts_per_frame.mean()
            else:
                avg_neighbors_per_frame = torch.tensor(0.0)
        if "eye_mask" in locals():
            del eye_mask
        if "above" in locals():
            del above
        if "counts_per_frame" in locals():
            del counts_per_frame
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Print the statistic and keep for output dictionary later
        try:
            print(f"[Similarity] sim_thres={sim_thres:.3f}; avg>thres/frame: {avg_neighbors_per_frame.item():.3f}")
        except Exception:
            pass

        # Decide number of groups: auto only when caller uses -1; otherwise use explicit positive value.
        Nx = similarity.shape[1]
        k_auto = int(torch.floor(avg_neighbors_per_frame).item())
        k_auto = max(1, min(k_auto, Nx))
        if num_groups == -1:
            num_groups_eff = min(5, k_auto)
            print(f"[Partitions] Auto num_groups from similarity: {num_groups_eff}")
        else:
            num_groups_eff = int(num_groups)
            print(f"[Partitions] Override num_groups: {num_groups_eff}")


        # Optimize partitions (ensure grads enabled even if caller is in no_grad)
        t_opt = _timer_start()
        with torch.enable_grad():
            scene_partition_results = optimize_scene_partitions(
                similarity,
                num_groups=num_groups_eff,
                steps=2000,
                lr=1e-1,
                lam=0.5,
                alpha=1.0,
                beta=1.0,
                patience=10,
                log_interval=50,
            )
        _timer_end(t_opt, "partition_optimize")
        del similarity
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        device = images.device
        if S > 1:
            ids_all = torch.stack(
                [torch.stack(res["group_indices_padded"], dim=0) for res in scene_partition_results],
                dim=0,
            ).to(device)  # (B, K, Lg)
            K = int(ids_all.shape[1])
            Lg = int(ids_all.shape[2])
            S_eff = Lg + 1
        else:
            K = int(num_groups_eff)
            Lg = 0
            S_eff = 1
        BK = B * K

        # Precompute vectorized restore indices (B, S) for k and s_eff (GPU-only)
        restore_k = torch.zeros((B, S), dtype=torch.long, device=device)
        restore_s = torch.zeros((B, S), dtype=torch.long, device=device)
        if S > 1:
            # ids_all already stacked above; values in 0..S-2
            ids_img_all = ids_all + 1  # map to 1..S-1, shape (B, K, Lg)
            Bk, Kl, Lgl = ids_img_all.shape
            frames = torch.arange(1, S, dtype=torch.long, device=device)  # (S-1,)
            big = K * max(Lg, 1) + 1
            # Flatten K,Lg then compare: mask (B, K*Lg, S-1)
            ids_flat = ids_img_all.view(B, -1)  # (B, K*Lg)
            pos = torch.arange(ids_flat.shape[1], dtype=torch.long, device=device)  # (K*Lg,)
            mask = ids_flat.unsqueeze(-1) == frames.view(1, 1, -1)  # (B, K*Lg, S-1)
            masked_pos = torch.where(mask, pos.view(1, -1, 1), torch.full((1, ids_flat.shape[1], S-1), big, device=device, dtype=torch.long))
            min_pos = masked_pos.min(dim=1).values  # (B, S-1)
            k_sel = torch.div(min_pos, Lg, rounding_mode='floor')
            j_sel = min_pos % Lg
            restore_k[:, 1:] = k_sel
            restore_s[:, 1:] = j_sel + 1  # shift by 1 because grouped pos 0 is frame0
            del frames, mask, masked_pos, min_pos, k_sel, j_sel, ids_flat, pos
        # frame 0 maps to (0,0) by initialization
        del scene_partition_results

        # Build grouped image tensor (B, K, S_eff, 3, H, W): [frame0] + frames by padded indices (+1)
        t_group = _timer_start()
        images_grouped = []
        if S > 1:
            # GPU index without host roundtrip
            ids_img_all = ids_all + 1  # (B, K, Lg)
            zero_idx = torch.zeros((B, ids_img_all.shape[1], 1), dtype=torch.long, device=device)
            idx_seq_all = torch.cat([zero_idx, ids_img_all], dim=2)  # (B, K, S_eff)
            # Still loop over b,k to avoid complex batched gather along dim=1
            for b in range(B):
                grouped_b = [images[b].index_select(0, idx_seq_all[b, k]) for k in range(K)]
                images_grouped.append(torch.stack(grouped_b, dim=0))
            del zero_idx
        else:
            for b in range(B):
                images_grouped.append(images[b:b+1].expand(K, -1, -1, -1, -1))
        images_grouped = torch.stack(images_grouped, dim=0)  # (B, K, S_eff, 3, H, W)
        images_eff = images_grouped.view(BK, S_eff, C_in, H, W)
        del images_grouped
        _timer_end(t_group, "group_build")

        # Build grouped patch tokens directly from precomputed patches (avoid re-embedding)
        t_group_pe = _timer_start()
        patches_grouped = []
        if S > 1:
            for b in range(B):
                grouped_b = [patches_bspc[b].index_select(0, idx_seq_all[b, k]) for k in range(K)]
                patches_grouped.append(torch.stack(grouped_b, dim=0))
        else:
            for b in range(B):
                patches_grouped.append(patches_bspc[b:b+1].expand(K, -1, -1, -1))
        patches_grouped = torch.stack(patches_grouped, dim=0)  # (B, K, S_eff, Pp, Cc)
        patches_eff = patches_grouped.view(BK, S_eff, Pp, Cc)
        del patches_grouped, patches_bspc
        if "idx_seq_all" in locals():
            del idx_seq_all
        if "ids_img_all" in locals():
            del ids_img_all
        if "ids_all" in locals():
            del ids_all
        _timer_end(t_group_pe, "group_build.patch_tokens")

        # Run aggregator on grouped patch tokens (skip internal patch_embed)
        t_agg = _timer_start()
        aggregated_tokens_list_eff, patch_start_idx = self.aggregator(
            images=None, patch_tokens_bspc=patches_eff, hw=(H, W)
        )
        _timer_end(t_agg, "aggregator_forward")
        del patches_eff
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Helper to restore (BK, S_eff, ...) -> (B, S, ...) using precomputed indices
        def restore_sequence(z_eff: torch.Tensor) -> torch.Tensor:
            z_grouped = z_eff.view(B, K, S_eff, *z_eff.shape[2:])
            batch_idx = torch.arange(B, device=z_eff.device)[:, None]
            out = z_grouped[batch_idx, restore_k, restore_s, ...]
            out0 = z_grouped.mean(dim=1)[:, 0, ...]
            out = torch.cat([out0.unsqueeze(1), out[:, 1:, ...]], dim=1)
            return out

        predictions = {}
        # Attach similarity threshold statistic
        if 'avg_neighbors_per_frame' not in locals():
            avg_neighbors_per_frame = torch.tensor(0.0)
        predictions["sim_above_thres_avg"] = float(avg_neighbors_per_frame.item())
        predictions["num_groups_auto"] = int(K)

        # Build head processing list: (name, forward_fn, restore_writer)
        heads = []
        if self.camera_head is not None:
            heads.append((
                "camera_head",
                lambda: self.camera_head(aggregated_tokens_list_eff),
                lambda eff: (
                    {"pose_enc": eff[-1], "pose_enc_list": eff},
                    True,  # camera returns list, need per-item restore
                ),
            ))
        if self.depth_head is not None:
            heads.append((
                "depth_head",
                lambda: self.depth_head(aggregated_tokens_list_eff, images=images_eff, patch_start_idx=patch_start_idx),
                lambda eff: (
                    {"depth": eff[0], "depth_conf": eff[1]},
                    False,
                ),
            ))
        if self.point_head is not None:
            heads.append((
                "point_head",
                lambda: self.point_head(aggregated_tokens_list_eff, images=images_eff, patch_start_idx=patch_start_idx),
                lambda eff: (
                    {"world_points": eff[0], "world_points_conf": eff[1]},
                    False,
                ),
            ))
        if self.track_head is not None and query_points is not None:
            if query_points is not None and query_points.dim() == 3 and query_points.shape[0] == B:
                query_points_eff = query_points.repeat_interleave(K, dim=0)
            else:
                query_points_eff = query_points
            heads.append((
                "track_head",
                lambda: self.track_head(aggregated_tokens_list_eff, images=images_eff, patch_start_idx=patch_start_idx, query_points=query_points_eff),
                lambda eff: (
                    {"track": eff[0], "vis": eff[1], "conf": eff[2]},
                    True,  # track track_list is list
                ),
            ))

        for name, fwd_fn, writer in heads:
            t_fwd = _timer_start()
            eff_out = fwd_fn()
            _timer_end(t_fwd, f"{name}.forward")

            tensors_map, is_list = writer(eff_out)
            t_rest = _timer_start()
            for key, tensor_eff in tensors_map.items():
                if is_list and isinstance(tensor_eff, list):
                    # list of (BK, S_eff, ...)
                    restored_list = [restore_sequence(te) for te in tensor_eff]
                    if key == "track":
                        predictions[key] = restored_list[-1]
                    else:
                        predictions[key] = restored_list
                else:
                    predictions[key] = restore_sequence(tensor_eff)
            _timer_end(t_rest, f"{name}.restore")
        heads.clear()
        del heads
        if "query_points_eff" in locals():
            del query_points_eff
        del aggregated_tokens_list_eff
        del restore_k, restore_s, restore_sequence
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not self.training:
            predictions["images"] = images  # keep original images for viz
        del images_eff

        return predictions
