import argparse
from pathlib import Path
import numpy as np
import torch
import time
import os
import sys

# Ensure project root is in sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import (
    load_poses,
    get_vgg_input_imgs,
    get_sorted_image_paths,
    get_all_scenes,
    build_frame_selection,
    load_images_rgb,
    infer_vggt_and_reconstruct,
    evaluate_scene_and_save,
    compute_average_metrics_and_save,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=Path, default="/home/jovyan/shared/xinzeli/scannetv2/process_scannet"
    )
    parser.add_argument(
        "--gt_ply_dir",
        type=Path,
        default="/home/jovyan/shared/xinzeli/scannetv2/scannet",
    )
    parser.add_argument("--output_path", type=Path, default="./eval_results")
    parser.add_argument("--merging", type=int, default=None)
    parser.add_argument(
        "--num_groups",
        type=int,
        default=None,
        help="Subscene groups: None disables grouping; -1 uses auto groups; positive integer uses fixed groups",
    )
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument(
        "--depth_conf_thresh",
        type=float,
        default=3.0,
        help="Depth confidence threshold for filtering low confidence depth values",
    )
    parser.add_argument(
        "--chamfer_max_dist",
        type=float,
        default=0.5,
        help="Maximum distance threshold in Chamfer Distance computation, distances exceeding this value will be clipped",
    )
    parser.add_argument(
        "--input_frame",
        type=int,
        default=200,
        help="Maximum number of frames selected for processing per scene",
    )
    parser.add_argument(
        "--num_scenes",
        type=int,
        default=50,
        help="Maximum number of scenes to evaluate",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="./ckpt/model_tracker_fixed_e20.pt",
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--vis_attn_map",
        action="store_true",
        help="Whether to visualize attention maps during inference",
    )
    args = parser.parse_args()
    torch.manual_seed(33)
    np.random.seed(0)

    # Scene sampling
    scannet_scenes = get_all_scenes(args.data_dir, args.num_scenes)
    print(f"Evaluate {len(scannet_scenes)} scenes")

    all_scenes_metrics = {"scenes": {}, "average": {}}
    from collections import defaultdict
    scene_infer_times = defaultdict(list)
    # Force use of bf16 data type
    dtype = torch.bfloat16
    # Load VGGT model
    model = VGGT(merging=args.merging, vis_attn_map=args.vis_attn_map)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    incompat = model.load_state_dict(ckpt, strict=False)
    model = model.cuda().eval()
    model = model.to(torch.bfloat16)

    # Process each scene
    for scene in scannet_scenes:
        scene_dir = args.data_dir / f"{scene}"
        output_scene_dir = args.output_path / f"input_frame_{args.input_frame}" / scene
        if (output_scene_dir / "metrics.json").exists():
            continue

        # Load scene data
        images_dir = scene_dir / "color"
        pose_path = scene_dir / "pose"
        image_paths = get_sorted_image_paths(images_dir)
        poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_path)
        if (
            poses_gt is None
            or first_gt_pose is None
            or available_pose_frame_ids is None
        ):
            print(f"Skipping scene {scene}: no pose data")
            continue

        # Frame filtering
        selected_frame_ids, selected_image_paths, selected_pose_indices = (
            build_frame_selection(
                image_paths, available_pose_frame_ids, args.input_frame
            )
        )

        # Get corresponding poses
        c2ws = poses_gt[selected_pose_indices]
        image_paths = selected_image_paths

        if len(image_paths) == 0:
            print(f"No images found in {images_dir}")
            continue

        print("🚩Processing", scene, f"Found {len(image_paths)} images")
        all_cam_to_world_mat = []
        all_world_points = []

        try:
            # Load images
            images = load_images_rgb(image_paths)

            if not images or len(images) < 3:
                print(f"Skipping {scene}: insufficient valid images")
                continue

            frame_ids = selected_frame_ids
            vgg_input, patch_width, patch_height = get_vgg_input_imgs(images)
            print(f"Patch dimensions: {patch_width}x{patch_height}")

            # Update model attention layers with dynamic patch dimensions
            model.update_patch_dimensions(patch_width, patch_height)

            # Inference + Reconstruction
            (
                extrinsic_np,
                intrinsic_np,
                all_world_points,
                all_point_colors,
                all_cam_to_world_mat,
                inference_time_ms,
            ) = infer_vggt_and_reconstruct(
                model,
                vgg_input,
                dtype,
                args.depth_conf_thresh,
                image_paths,
                num_groups=args.num_groups,
            )
            print(f"Inference time: {inference_time_ms:.2f}ms")
            frame_count = len(image_paths)
            total_time = inference_time_ms / 1000.0
            fps = frame_count / total_time if total_time > 0 else float("inf")
            print(f"Inference FPS (frames/s): {fps:.2f} [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
            scene_infer_times[scene].append(float(fps))

            max_points = 999999
            total_points = sum(p.shape[0] for p in all_world_points)
            if total_points <= max_points:
                merged_points = np.concatenate(all_world_points, axis=0)
            else:
                sizes = [p.shape[0] for p in all_world_points]
                alloc = [int(max_points * s / total_points) for s in sizes]
                short = max_points - sum(alloc)
                if short > 0:
                    order = np.argsort(-np.array(sizes))
                    for i in order[:short]:
                        alloc[i] += 1
                sampled = []
                for arr, k in zip(all_world_points, alloc):
                    if k <= 0:
                        continue
                    idx = np.random.choice(arr.shape[0], k, replace=False)
                    sampled.append(arr[idx])
                merged_points = np.concatenate(sampled, axis=0)
            all_world_points = [merged_points]

            # Process results
            if not all_cam_to_world_mat or not all_world_points:
                print(
                    f"Skipping {scene}: failed to obtain valid camera poses or point clouds"
                )
                continue

            # Evaluate and save
            metrics = evaluate_scene_and_save(
                scene,
                c2ws,
                first_gt_pose,
                frame_ids,
                all_cam_to_world_mat,
                all_world_points,
                output_scene_dir,
                args.gt_ply_dir,
                args.chamfer_max_dist,
                inference_time_ms,
                args.plot,
            )
            if metrics is not None:
                all_scenes_metrics["scenes"][scene] = {
                    key: float(value)
                    for key, value in metrics.items()
                    if key
                    in [
                        "chamfer_distance",
                        "ate",
                        "are",
                        "rpe_rot",
                        "rpe_trans",
                        "inference_time_ms",
                    ]
                }
                all_scenes_metrics["scenes"][scene]["fps"] = float(fps)
                print("Complete metrics", all_scenes_metrics["scenes"][scene])

        except Exception as e:
            print(f"Error processing scene {scene}: {e}")
            import traceback

            traceback.print_exc()

    for sid, times in scene_infer_times.items():
        if len(times) > 0:
            avg_fps = np.mean(times)
            print(f"Idx: {sid}, FPS_avg: {avg_fps:.3f}")
    # Summarize average metrics and save
    compute_average_metrics_and_save(
        all_scenes_metrics,
        args.output_path,
        args.input_frame,
    )
