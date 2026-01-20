import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import pathlib
import time
import glob
import tempfile
import shutil
from general_motion_retargeting import RobotMotionViewer
from rich import print
from tqdm import tqdm
import numpy as np
import imageio
from skimage.transform import resize


def vis_npz_motion(motion_npz_path, output_path, robot_type="unitree_g1", rate_limit=False, motion_fps=30):
    """渲染单个npz文件为视频"""
    motion_csv = np.load(motion_npz_path)['qpos']
    data_frames = motion_csv.shape[0]

    robot_motion_viewer = RobotMotionViewer(
        robot_type=robot_type,
        motion_fps=motion_fps,
        transparent_robot=0,
        record_video=True,
        video_path=output_path,
    )
    
    print(f"Rendering {motion_npz_path} -> {output_path} ({data_frames} frames)")
    pbar = tqdm(total=data_frames, desc="Rendering", leave=False)
    
    i = 0
    while i < data_frames:
        pbar.update(1)
        qpos = motion_csv[i]
        quat_wxyz = qpos[3:7]
        root_pos = qpos[:3]

        robot_motion_viewer.step(
            root_pos=root_pos,
            root_rot=quat_wxyz,
            dof_pos=qpos[7:],
            rate_limit=rate_limit,
        )
        i += 1
    
    pbar.close()
    robot_motion_viewer.close()
    del robot_motion_viewer


def concatenate_videos_horizontally(video1_path, video2_path, output_path):
    """将两个视频左右拼接"""
    # 使用imageio读取视频
    reader1 = imageio.get_reader(video1_path)
    reader2 = imageio.get_reader(video2_path)
    
    # 获取视频属性
    fps = reader1.get_meta_data()['fps']
    width1, height1 = reader1.get_meta_data()['size']
    width2, height2 = reader2.get_meta_data()['size']
    
    # 确保两个视频高度相同，如果不相同则调整
    target_height = max(height1, height2)
    target_width = width1 + width2
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 使用imageio写入视频
    writer = imageio.get_writer(output_path, fps=fps)
    
    print(f"Concatenating videos horizontally...")
    pbar = tqdm(desc="Concatenating", leave=False)
    
    frame_count = 0
    try:
        # 使用迭代器方式读取，直到任一视频结束
        for frame1, frame2 in zip(reader1, reader2):
            # 调整两个视频到相同高度
            if frame1.shape[0] != target_height:
                frame1 = resize(frame1, (target_height, width1), preserve_range=True, anti_aliasing=True).astype(frame1.dtype)
            if frame2.shape[0] != target_height:
                frame2 = resize(frame2, (target_height, width2), preserve_range=True, anti_aliasing=True).astype(frame2.dtype)
            
            # 左右拼接
            combined_frame = np.hstack([frame1, frame2])
            writer.append_data(combined_frame)
            frame_count += 1
            pbar.update(1)
    except (StopIteration, IndexError):
        # 视频读取结束
        pass
    
    pbar.close()
    reader1.close()
    reader2.close()
    writer.close()
    print(f"Concatenated {frame_count} frames")


def concatenate_videos_vertically(video_paths, output_path):
    """将多个视频前后拼接"""
    if not video_paths:
        return
    
    # 读取第一个视频获取属性
    reader_first = imageio.get_reader(video_paths[0])
    fps = reader_first.get_meta_data()['fps']
    width, height = reader_first.get_meta_data()['size']
    reader_first.close()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 使用imageio写入视频
    writer = imageio.get_writer(output_path, fps=fps)
    
    total_frames = 0
    print(f"Concatenating {len(video_paths)} videos vertically")
    
    for video_path in tqdm(video_paths, desc="Processing videos"):
        reader = imageio.get_reader(video_path)
        frame_count = 0
        
        for frame in reader:
            # 确保帧大小一致
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = resize(frame, (height, width), preserve_range=True, anti_aliasing=True).astype(frame.dtype)
            
            writer.append_data(frame)
            frame_count += 1
            total_frames += 1
        
        reader.close()
    
    writer.close()
    print(f"Total frames in final video: {total_frames}")


def process_gt_pred_comparison(input_dir, output_path, robot_type="unitree_g1", rate_limit=False, motion_fps=30):
    """处理gt和pred的对比视频"""
    
    # 查找所有gt和pred文件
    gt_files = sorted(glob.glob(os.path.join(input_dir, "*.gt.npz")))
    pred_files = sorted(glob.glob(os.path.join(input_dir, "*.pred.npz")))
    
    # 提取motion编号并匹配
    motion_dict = {}
    for gt_file in gt_files:
        basename = os.path.basename(gt_file)
        motion_id = basename.replace(".gt.npz", "")
        motion_dict[motion_id] = {"gt": gt_file}
    
    for pred_file in pred_files:
        basename = os.path.basename(pred_file)
        motion_id = basename.replace(".pred.npz", "")
        if motion_id in motion_dict:
            motion_dict[motion_id]["pred"] = pred_file
    
    # 过滤出同时有gt和pred的motion
    valid_motions = {k: v for k, v in motion_dict.items() if "gt" in v and "pred" in v}
    
    if not valid_motions:
        print("No valid motion pairs found!")
        return
    
    print(f"Found {len(valid_motions)} motion pairs to process")
    
    # 创建临时目录
    temp_dir = tempfile.mkdtemp()
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        combined_videos = []
        
        # 处理每个motion
        for motion_id, files in sorted(valid_motions.items()):
            print(f"\nProcessing {motion_id}...")
            
            # 渲染gt和pred为临时视频
            gt_temp_video = os.path.join(temp_dir, f"{motion_id}_gt.mp4")
            pred_temp_video = os.path.join(temp_dir, f"{motion_id}_pred.mp4")
            combined_temp_video = os.path.join(temp_dir, f"{motion_id}_combined.mp4")
            
            # 渲染gt视频
            vis_npz_motion(files["gt"], gt_temp_video, robot_type, rate_limit, motion_fps)
            
            # 渲染pred视频
            vis_npz_motion(files["pred"], pred_temp_video, robot_type, rate_limit, motion_fps)
            
            # 左右拼接gt和pred
            concatenate_videos_horizontally(gt_temp_video, pred_temp_video, combined_temp_video)
            
            combined_videos.append(combined_temp_video)
        
        # 前后拼接所有视频
        print(f"\nConcatenating all videos into final output...")
        concatenate_videos_vertically(combined_videos, output_path)
        
        print(f"\nFinal video saved to: {output_path}")
        
    finally:
        # 清理临时文件
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    HERE = pathlib.Path(__file__).parent
    
    parser = argparse.ArgumentParser(description="Render GT and Pred motion comparison videos")
    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01"],
        default="unitree_g1",
    )
    
    parser.add_argument(
        "--input_dir",
        type=str,
        default="/root/workspace/CAMDM/PyTorch/save/camdm_aistpp_g1_1/best",
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="videos/gt_pred_comparison.mp4",
    )
    
    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )
    
    parser.add_argument(
        "--motion_fps",
        type=int,
        default=30,
    )
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    process_gt_pred_comparison(
        args.input_dir,
        args.output_path,
        args.robot,
        args.rate_limit,
        args.motion_fps
    )

