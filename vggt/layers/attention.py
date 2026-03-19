import os
from pathlib import Path
import time
from PIL import Image
import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F
from tqdm.std import tqdm
from merging.merge import (
    token_merge_bipartite2d,
)
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

XFORMERS_AVAILABLE = False

# Global variables for attention visualization
vis_attn_map = False
current_images = []
attention_map = None


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        qk_norm: bool = False,
        kv_group_size: int = 1,
        fused_attn: bool = True,
        rope=None,
        global_merging=None,
        patch_width: int = 37,
        patch_height: int = 28,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.fused_attn = fused_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.rope = rope
        self.kv_group_size = kv_group_size

    def forward(self, x: Tensor, pos=None, global_merging=None) -> Tensor:
        merge_num = list(range(24))

        B, N, C = x.shape

        if vis_attn_map and global_merging is not None:
            # s1 Chunk computation of q@k
            chunk_size = 4096
            attn_maps = []
            total_chunks = (
                q.size(-2) + chunk_size - 1
            ) // chunk_size  # Calculate total number of chunks
            start_time_total = time.time()
            with torch.no_grad():
                for start in tqdm(
                    range(0, q.size(-2), chunk_size),
                    total=total_chunks,
                    desc="Processing chunks",
                ):
                    end = min(start + chunk_size, q.size(-2))
                    q_chunk = q[:, :, start:end, :]  # (1, 16, chunk_size, 64)
                    start_time_chunk = time.time()
                    attn_chunk = q_chunk @ k.transpose(
                        -2, -1
                    )  # (1, 16, chunk_size, 34353)
                    attn_maps.append(attn_chunk.cpu())
                    end_time_chunk = time.time()
                    print(
                        f"Chunk {start}:{end} processed in {end_time_chunk - start_time_chunk:.4f} seconds"
                    )
                    del q_chunk, attn_chunk
            end_time_total = time.time()
            print(
                f"\nTotal processing time: {end_time_total - start_time_total:.4f} seconds"
            )

            attn_map = torch.cat(attn_maps, dim=-2)
            attn = attn_map[0].mean(0)
            frame_token_num = self.patch_height * self.patch_width + 5
            for target_token_idx in [
                0,
                self.patch_height * self.patch_width,
                self.patch_height * self.patch_width * 10,
            ]:  # Iterate through each image's target_token
                for image_idx in range(
                    len(current_images)
                ):  # Corresponding to which image to visualize
                    target_attn = attn[
                        target_token_idx,
                        image_idx * frame_token_num : (image_idx + 1) * frame_token_num,
                    ]
                    target_attn_map = target_attn[5:].reshape(
                        self.patch_height, self.patch_width
                    )
                    # 1) Read original image to get true size (H, W)
                    image_path = current_images[image_idx]
                    p = Path(image_path)
                    parts = p.parts
                    scene_name = parts[-4]

                    image = Image.open(image_path).convert("RGB")
                    img_width, img_height = image.size  # PIL size: (W, H)

                    # Upsample attention map to the original image size
                    target_attn_map = F.interpolate(
                        target_attn_map.unsqueeze(0).unsqueeze(0),  # (1,1,h,w)
                        size=(img_height, img_width),
                        mode="bilinear",
                    )
                    target_attn_map = target_attn_map.squeeze()

                    # Convert image to numpy for blending
                    img_np = np.array(image) / 255.0

                    # 2. Normalize attention map
                    target_attn_map = (target_attn_map - target_attn_map.min()) / (
                        target_attn_map.max() - target_attn_map.min()
                    )

                    # 3. Color attention map
                    cmap = plt.get_cmap("jet")
                    attn_color = cmap(target_attn_map.cpu().float().numpy())
                    attn_color = attn_color[:, :, :3]

                    # 4. Blend attention and original image
                    overlay = img_np * 0.5 + attn_color * 0.5

                    plt.imshow(overlay, cmap="viridis")
                    output_dir = f"attention_map/{scene_name}/block_{attention_map}/token_{target_token_idx}"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, f"color_{image_idx}.png")
                    plt.savefig(output_path)
                    plt.close()
            del attn_maps, attn_map, attn

        merge_cfg = None
        if isinstance(global_merging, dict):
            merge_cfg = global_merging
        trigger_merge = (isinstance(global_merging, int) and global_merging in merge_num) or (
            isinstance(merge_cfg, dict) and merge_cfg.get("block") in merge_num
        )

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        del qkv
        q, k = self.q_norm(q), self.k_norm(k)
        if self.rope is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)
        B_q, H_q, N_q, D_q = q.shape

        u_list = None
        if trigger_merge:
            generator = torch.Generator(device=x.device)
            generator.manual_seed(33)
            merge_ratio = (
                merge_cfg.get("ratio", 0.9) if isinstance(merge_cfg, dict) else 0.9
            )
            r = int(x.shape[1] * merge_ratio)

            q_list = []
            k_list = []
            v_list = []
            u_list = []
            for b in range(B):
                m_b, u_b = token_merge_bipartite2d(
                    x[b : b + 1],
                    self.patch_width,
                    self.patch_height,
                    2,
                    2,
                    r,
                    False,
                    generator,
                    enable_protection=True,
                )
                q_merge_in = q[b : b + 1].permute(0, 2, 1, 3).reshape(1, N_q, H_q * D_q)
                k_merge_in = k[b : b + 1].permute(0, 2, 1, 3).reshape(1, N_q, H_q * D_q)
                v_merge_in = v[b : b + 1].permute(0, 2, 1, 3).reshape(1, N_q, H_q * D_q)
                q_out, k_out, v_out = m_b(
                    q_merge_in,
                    mode="mean",
                    extra_tensors=k_merge_in,
                    extra_tensors_2=v_merge_in,
                )
                N_m = q_out.shape[1]
                q_b = q_out.reshape(1, N_m, H_q, D_q).permute(0, 2, 1, 3)
                k_b = k_out.reshape(1, N_m, H_q, D_q).permute(0, 2, 1, 3)
                v_b = v_out.reshape(1, N_m, H_q, D_q).permute(0, 2, 1, 3)
                q_list.append(q_b)
                k_list.append(k_b)
                v_list.append(v_b)
                u_list.append(u_b)
                del q_merge_in, k_merge_in, v_merge_in, q_out, k_out, v_out
            q = torch.cat(q_list, dim=0)
            k = torch.cat(k_list, dim=0)
            v = torch.cat(v_list, dim=0)
            N = q.shape[2]
            del q_list, k_list, v_list
        else:
            N = N_q

        # frame_token_num = self.patch_height * self.patch_width + 5
        # if N_q % frame_token_num != 0:
        #     raise ValueError(
        #         f"Token count {N_q} is not divisible by frame size {frame_token_num}"
        #     )
        # num_frames = N_q // frame_token_num

        # q_frames = q.reshape(B_q, H_q, num_frames, frame_token_num, D_q)
        # k_frames = k.reshape(B_q, H_q, num_frames, frame_token_num, D_q)
        # q_frame_mean = q_frames.mean(dim=3)
        # k_frame_mean = k_frames.mean(dim=3)
        # # Use frame-averaged q/k logits for frame analysis
        # frame_mean_logits_full = torch.matmul(
        #     q_frame_mean, k_frame_mean.transpose(-1, -2)
        # ) * self.scale
        # frame_mean_logits_full = frame_mean_logits_full.mean(dim=1)
        # if num_frames > 1:
        #     frame_mean_logits = frame_mean_logits_full[:, 1:, 1:]
        #     frame_importance = frame_mean_logits.mean(dim=1)
        # else:
        #     frame_mean_logits = frame_mean_logits_full.new_zeros(B_q, 0, 0)
        #     frame_importance = frame_mean_logits_full.new_zeros(B_q, 0)
        # if B_q > 0 and frame_importance.shape[1] > 0:
        #     keyframe_order = frame_importance[0].argsort(descending=True)
        #     keyframe_order = (keyframe_order + 1).detach().cpu().tolist()
        #     keyframe_scores = frame_importance[0].detach().cpu().tolist()
        #     print("keyframe_order_by_column:", keyframe_order)
        #     print("keyframe_scores:", keyframe_scores)
        #     print("frame_mean_logits.shape:", frame_mean_logits.shape)
        #     if frame_mean_logits.shape[1] > 0:
        #         os.makedirs("frame_mean_logits_vis", exist_ok=True)
        #         heatmap = frame_mean_logits[0].detach().cpu().float().numpy()
        #         plt.figure()
        #         plt.imshow(heatmap, cmap="coolwarm")
        #         plt.colorbar()
        #         timestamp = int(time.time() * 1000)
        #         plt.savefig(
        #             os.path.join(
        #                 "frame_mean_logits_vis", f"heatmap_{timestamp}.png"
        #             )
        #         )
        #         plt.close()

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        del q, k, v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if trigger_merge:
            x_list = []
            for b in range(B):
                x_list.append(u_list[b](x[b : b + 1]))
            x = torch.cat(x_list, dim=0)
            del x_list, u_list
        return x


class MemEffAttention(Attention):
    def forward(
        self, x: Tensor, attn_bias=None, pos=None, global_merging=None
    ) -> Tensor:
        assert (
            pos is None or self.rope is not None
        ), "Position encoding is only supported with RoPE"
        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x, pos=pos, global_merging=global_merging)

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = qkv.unbind(2)

        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            k = self.rope(k, pos)

        # Use scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * (self.head_dim**-0.5)
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = F.softmax(attn, dim=-1)
        x = torch.matmul(attn, v)
        x = x.reshape([B, N, C])

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
