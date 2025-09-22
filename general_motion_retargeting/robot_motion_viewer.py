import os
import time
import mujoco as mj
import imageio
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT, VIEWER_CAM_DISTANCE_DICT
from loop_rate_limiters import RateLimiter
import numpy as np
from rich import print
import glfw


def draw_frame(
    renderer,
    pos,
    mat,
    size,
    joint_name=None,
    orientation_correction=R.from_euler("xyz", [0, 0, 0]),
    pos_offset=np.array([0, 0, 0]),
):
    """修改后的draw_frame函数，使用renderer而不是viewer"""
    rgba_list = [[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]]
    
    # 在离屏渲染中，我们需要通过其他方式绘制参考帧
    # 这里简化处理，只记录位置信息，实际应用中可能需要更复杂的处理
    # 或者使用MuJoCo的标记功能
    pass


class RobotMotionViewer:
    def __init__(self,
                robot_type,
                camera_follow=True,
                motion_fps=30,
                transparent_robot=0,
                # video recording
                record_video=True,  # 默认开启录制
                video_path=None,
                video_width=640,
                video_height=480):
        
        self.robot_type = robot_type
        self.xml_path = ROBOT_XML_DICT[robot_type]
        self.model = mj.MjModel.from_xml_path(str(self.xml_path))
        self.data = mj.MjData(self.model)
        self.robot_base = ROBOT_BASE_DICT[robot_type]
        self.viewer_cam_distance = VIEWER_CAM_DISTANCE_DICT[robot_type]
        mj.mj_step(self.model, self.data)
        
        self.motion_fps = motion_fps
        self.rate_limiter = RateLimiter(frequency=self.motion_fps, warn=False)
        self.camera_follow = camera_follow
        self.record_video = record_video

        # 检查是否在有显示的环境下运行
        try:
            # 尝试初始化GLFW来检查是否有显示设备
            if not glfw.init():
                raise RuntimeError("GLFW初始化失败")
            
            # 尝试创建离屏窗口
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            window = glfw.create_window(100, 100, "Offscreen", None, None)
            if not window:
                raise RuntimeError("无法创建离屏窗口")
            
            glfw.make_context_current(window)
            self.has_display = True
            glfw.destroy_window(window)
            
        except Exception as e:
            print(f"无显示环境检测: {e}")
            self.has_display = False
            # 设置环境变量以使用离屏渲染
            os.environ['MUJOCO_GL'] = 'egl'  # 或者 'egl'

        # 初始化渲染器
        self.renderer = mj.Renderer(self.model, height=video_height, width=video_width)
        
        # 初始化相机
        self.camera = mj.MjvCamera()
        self.camera.type = mj.mjtCamera.mjCAMERA_FREE
        self.camera.fixedcamid = -1
        self.camera.trackbodyid = self.model.body(self.robot_base).id
        self.camera.distance = self.viewer_cam_distance
        self.camera.elevation = -10
        self.camera.azimuth = 180
        
        # 初始化场景和上下文
        self.scene = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
        
        if self.record_video:
            assert video_path is not None, "Please provide video path for recording"
            self.video_path = video_path

            self.mp4_writer = imageio.get_writer(self.video_path, fps=self.motion_fps)
            print(f"Recording video to {self.video_path}")
    
    def step(self, 
            # robot data
            root_pos, root_rot, dof_pos, 
            # human data
            human_motion_data=None, 
            show_human_body_name=False,
            # scale for human point visualization
            human_point_scale=0.1,
            # human pos offset add for visualization    
            human_pos_offset=np.array([0.0, 0.0, 0]),
            # rate limit
            rate_limit=True, 
            follow_camera=True,
            ):
        """
        修改后的step函数，使用离屏渲染而不是交互式查看器
        """
        
        self.data.qpos[:3] = root_pos
        self.data.qpos[3:7] = root_rot  # quat need to be scalar first! for mujoco
        self.data.qpos[7:22+7] = dof_pos[0:22]
        self.data.qpos[7+22:7+29] = dof_pos[34:41]

        mj.mj_forward(self.model, self.data)
        
        if follow_camera:
            # 更新相机位置
            self.camera.lookat = self.data.xpos[self.model.body(self.robot_base).id]
            self.camera.distance = self.viewer_cam_distance
            self.camera.elevation = -10
            self.camera.azimuth = 180
        
        # 渲染场景
        mj.mjv_updateScene(self.model, self.data, mj.MjvOption(), mj.MjvPerturb(), 
                          self.camera, mj.mjtCatBit.mjCAT_ALL.value, self.scene)
        
        # 渲染到缓冲区
        self.renderer.update_scene(self.data, camera=self.camera)
        img = self.renderer.render()
        
        if self.record_video:
            self.mp4_writer.append_data(img)
        
        if rate_limit is True:
            self.rate_limiter.sleep()
    
    def close(self):
        if self.record_video:
            self.mp4_writer.close()
            print(f"Video saved to {self.video_path}")
