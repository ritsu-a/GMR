import argparse
import pathlib
import time
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.lafan1 import load_lafan1_file
from rich import print
from tqdm import tqdm
import os
import numpy as np

from moviepy.editor import VideoFileClip, AudioFileClip, clips_array, CompositeVideoClip
import os

def combine_videos_with_audio(video1_path, video2_path, audio_path, output_path):
    """
    将两个视频左右拼接并添加音频
    
    参数:
    video1_path: 第一个视频文件路径
    video2_path: 第二个视频文件路径
    audio_path: 音频文件路径
    output_path: 输出文件路径
    """
    # 加载视频文件
    clip1 = VideoFileClip(video1_path)
    # clip2 = VideoFileClip(video2_path)
    
    # # 确保两个视频高度相同（如果不相同，调整第二个视频的高度）
    # if clip1.h != clip2.h:
    #     # 计算缩放比例
    #     scaling_factor = clip1.h / clip2.h
    #     new_w = int(clip2.w * scaling_factor)
    #     clip2 = clip2.resize(width=new_w)
    # # 将两个视频并排拼接
    # final_clip = clips_array([[clip1, clip2]])
    
    final_clip = clip1

    # 加载音频文件
    audio = AudioFileClip(audio_path)
    
    # 设置视频的音频
    # 如果音频长度超过视频长度，截取音频；如果短于视频，循环音频
    if audio.duration > final_clip.duration:
        audio = audio.subclip(0, final_clip.duration)
    
    # 设置音频
    final_clip = final_clip.set_audio(audio)
    
    # 导出最终视频
    final_clip.write_videofile(
        output_path,
        codec='libx264',  # 使用H.264编码
        audio_codec='aac',  # 使用AAC音频编码
        temp_audiofile='temp-audio.m4a',  # 临时音频文件
        remove_temp=True  # 处理完成后删除临时文件
    )
    
    # 关闭所有剪辑以释放资源
    clip1.close()
    # clip2.close()
    audio.close()
    final_clip.close()


if __name__ == "__main__":
    
    HERE = pathlib.Path(__file__).parent

    parser = argparse.ArgumentParser()

    
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands", "booster_t1", "stanford_toddy", "fourier_n1", "engineai_pm01"],
        default="unitree_g1",
    )
        
    parser.add_argument(
        "--record_video",
        action="store_true",
        default=False,
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

    
    
    args = parser.parse_args()
    


    # load csv motion
    csv_path = "/home/pengyang/codebase/playground/GMR/videos/llm.csv"
    motion_csv = np.genfromtxt(csv_path, delimiter=',')
    data_frames = motion_csv.shape[0]

    motion_fps = 50
    
    robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
                                            motion_fps=motion_fps,
                                            transparent_robot=0,
                                            record_video=True,
                                            video_path=csv_path.replace(".csv", f".mp4"),
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

        ## fix lower body motion
        qpos[:7] *= 0
        qpos[2] += 0.8
        qpos[7:7+11] *= 0

        # visualize
        robot_motion_viewer.step(
            root_pos=qpos[:3],
            root_rot=qpos[3:7],
            dof_pos=qpos[7:],
            rate_limit=args.rate_limit,
            # human_pos_offset=np.array([0.0, 0.0, 0.0])
        )

        i += 1

        
    
    

    # Close progress bar
    pbar.close()
    
    robot_motion_viewer.close()
    del robot_motion_viewer
       







# # load csv motion
#     csv_path = "/home/pengyang/codebase/playground/GMR/videos/decoded.csv"
#     motion_csv = np.genfromtxt(csv_path, delimiter=',')
#     data_frames = motion_csv.shape[0]

#     motion_fps = 50
    
#     robot_motion_viewer = RobotMotionViewer(robot_type=args.robot,
#                                             motion_fps=motion_fps,
#                                             transparent_robot=0,
#                                             record_video=True,
#                                             video_path=csv_path.replace(".csv", f".mp4"),
#                                             # video_width=2080,
#                                             # video_height=1170
#                                             )
    
#     # FPS measurement variables
#     fps_counter = 0
#     fps_start_time = time.time()
#     fps_display_interval = 2.0  # Display FPS every 2 seconds
    
#     print(f"mocap_frame_rate: {motion_fps}")
    
#     # Create tqdm progress bar for the total number of frames
#     pbar = tqdm(total=data_frames, desc="visualizing")
    
#     # Start the viewer
#     i = 0

#     while i < data_frames:
        
#         # FPS measurement
#         fps_counter += 1
#         current_time = time.time()
#         if current_time - fps_start_time >= fps_display_interval:
#             actual_fps = fps_counter / (current_time - fps_start_time)
#             print(f"Actual rendering FPS: {actual_fps:.2f}")
#             fps_counter = 0
#             fps_start_time = current_time
            
#         # Update progress bar
#         pbar.update(1)

#         qpos = motion_csv[i]

#         ## fix lower body motion
#         qpos[:7] *= 0
#         qpos[2] += 0.8
#         qpos[7:7+11] *= 0

#         # visualize
#         robot_motion_viewer.step(
#             root_pos=qpos[:3],
#             root_rot=qpos[3:7],
#             dof_pos=qpos[7:],
#             rate_limit=args.rate_limit,
#             # human_pos_offset=np.array([0.0, 0.0, 0.0])
#         )

#         i += 1
    # # Close progress bar
    # pbar.close()
    
    # robot_motion_viewer.close()


    combine_videos_with_audio("videos/llm.mp4", "videos/llm.mp4", "videos/audio.wav", "videos/final_output.mp4")
       
