import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_lafan1_file
from rich import print
from tqdm import tqdm


import numpy as np
import pickle
import shutil
import joblib
import torch
from diff_quat import vec6d_to_quat

from scipy import signal


def low_pass_filter(data, cutoff_freq=0.2, order=4):
    """
    Apply a low-pass Butterworth filter to the data.
    
    Parameters:
        data (numpy.ndarray): The input data to be filtered.
        cutoff_freq (float): The normalized cutoff frequency (0~1).
        order (int): The order of the Butterworth filter.
        
    Returns:
        numpy.ndarray: The filtered data.
    """
    b, a = signal.butter(order, cutoff_freq, 'low')
    filtered_data = data.copy()
    for idx in range(filtered_data.shape[1]):
        filtered_data[:, idx] = signal.filtfilt(b, a, data[:, idx])
    return filtered_data

def pkl_to_csv(motion_pkl_path):
    with open(motion_pkl_path, "rb") as file:
        data = joblib.load(file)

    for key in data.keys():
        data[key] = data[key].cpu().numpy() if isinstance(data[key], torch.Tensor) else data[key]

    robot_name = data["robot_name"]
    match robot_name:
    
        case "g1_29":
            csv_data = np.zeros((data["angles"].shape[0], 36))
            csv_data[:, :3] = data["global_translation"][:, :, 0]
            csv_data[:, 3:7] = vec6d_to_quat(torch.tensor(data['global_rotation'])).numpy()
            csv_data[:, 7:] = data["angles"]

        case "_":
            print("Undefined robot type: ", robot_name)
            raise ValueError('Invalid robot type')

    return csv_data


def vis_pkl_motion(motion_pkl_path, output_path="final_output.mp4", robot_type="unitree_g1", rate_limit=False, motion_fps=20):

    

    motion_csv = pkl_to_csv(motion_pkl_path)
    motion_csv = low_pass_filter(motion_csv)
    data_frames = motion_csv.shape[0]

    
    robot_motion_viewer = RobotMotionViewer(robot_type=robot_type,
                                            motion_fps=motion_fps,
                                            transparent_robot=0,
                                            record_video=True,
                                            video_path=output_path,
                                            # video_width=2080,
                                            # video_height=1170
                                            )
    
    # FPS measurement variables
    fps_counter = 0
    fps_start_time = time.time()
    fps_display_interval = 2.0  # Display FPS every 2 seconds
    
    print(f"mocap_frame_rate: {motion_fps}")
    
    # Create tqdm progress bar for the total number of frames
    pbar = tqdm(total=data_frames, desc="visualizing")
    
    # Start the viewer
    i = 0

    while i < data_frames:
        
        # FPS measurement
        fps_counter += 1
        current_time = time.time()
        if current_time - fps_start_time >= fps_display_interval:
            actual_fps = fps_counter / (current_time - fps_start_time)
            print(f"Actual rendering FPS: {actual_fps:.2f}")
            fps_counter = 0
            fps_start_time = current_time
            
        # Update progress bar
        pbar.update(1)

        qpos = motion_csv[i]

        # Convert quaternion from xyzw to wxyz format for MuJoCo
        quat_xyzw = qpos[3:7]  # [x, y, z, w]
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])  # [w, x, y, z]

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=quat_wxyz,
            dof_pos=qpos[7:],
            rate_limit=rate_limit,
            # human_pos_offset=np.array([0.0, 0.0, 0.0])
        )

        i += 1

        
    
    

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close()
    del robot_motion_viewer
       



if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()

    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01"],
        default="unitree_g1",
    )
        

    parser.add_argument(
        "--video_path",
        type=str,
        default="videos/example.mp4",
    )

    parser.add_argument(
        "--rate_limit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--motion_fps",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--pkl_path",
        type=str,
        default="pkl/example.pkl",
    )

    
    
    args = parser.parse_args()

    vis_pkl_motion(args.pkl_path, args.video_path, args.robot, args.rate_limit, args.motion_fps)
       
