import os
os.environ["MUJOCO_GL"] = "egl"

import argparse
import pathlib
import time
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_lafan1_file
from rich import print
from tqdm import tqdm


import numpy as np



def vis_npz_motion(motion_npz_path, output_path="final_output.mp4", robot_type="g1_branco", rate_limit=False, motion_fps=30):

    
    motion_csv = np.load(motion_npz_path)['qpos']



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

        quat_wxyz = qpos[3:7]


        # def quaternion_multiply(q1, q2):
        #     w1, x1, y1, z1 = q1
        #     w2, x2, y2, z2 = q2
        #     w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        #     x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        #     y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        #     z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        #     return np.array([w, x, y, z])
        
        # quat_wxyz = quaternion_multiply(quat_wxyz, np.array([0.0, 1.0, 0.0, 0.0]))
        # quat_wxyz = quaternion_multiply(quat_wxyz, np.array([0.70710678, 0.0, 0.0, -0.70710678]))  
        root_pos = qpos[:3]

        # visualize
        robot_motion_viewer.step(
            root_pos=root_pos,
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
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01", "g1_brainco"],
        default="g1_brainco",
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
        default=60,
    )

    parser.add_argument(
        "--npz_path",
        type=str,
        default="pkl/example.pkl",
    )

    
    
    args = parser.parse_args()

    vis_npz_motion(args.npz_path, args.video_path, args.robot, args.rate_limit, args.motion_fps)
       
