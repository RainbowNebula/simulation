import os
import numpy as np
import torch
from glob import glob
import genesis as gs
import sys

sys.path.append('./api')
from api.handtraj_api_dmp import HandTrajProcess  # 初始化处理器，可配置DMP和ProMP参数
import pickle

import cv2
from copy import deepcopy
import time
from utils import rand_pose, decompose_grasppose

from transform_grasp import transform_grasp_pose_euler

import faulthandler
faulthandler.enable()  # 运行程序后，段错误时会输出详细堆栈
import time
import warnings
import traceback


class BaseEnv:
    def __init__(self, **kwargs):
        super().__init__()

        # 解析配置参数
        self.vis = kwargs.get('vis', False)
        self.seed = kwargs.get('seed', 0)
        self.save_data = kwargs.get('save_data', True)
        self.save_dir = kwargs.get('save_dir', "./test_sim_data5")
        self.save_rgb_origin = kwargs.get('save_rgb_origin', True)
        self.save_depth_origin = kwargs.get('save_depth_origin', False)
        self.task_name = kwargs.get('task_name', 'default')
        self.ep_num = kwargs.get('ep_num', 0)

        self.handtraj_processor = kwargs.get('handtraj_processor', None)

        # np.random.seed(self.seed)
        # torch.manual_seed(self.seed)

        ########################## init ##########################
        #gs.init(backend=gs.cpu,precision="32",logging_level='debug')
        gs.init(backend=gs.gpu, precision="32") #point_cloud, mask = camera.render_pointcloud() double free or corruption (out)
        
        # 初始化场景配置
        viewer_options = gs.options.ViewerOptions(
            camera_pos=kwargs.get('camera_pos', (3, -1, 1.5)),
            camera_lookat=kwargs.get('camera_lookat', (0.0, 0.0, 0.0)),
            camera_fov=kwargs.get('camera_fov', 30),
            max_FPS=kwargs.get('max_FPS', 60),
        )

        rigid_options = gs.options.RigidOptions(
            box_box_detection=False,
            max_collision_pairs=1000,
            # use_gjk_collision=True,
            # enable_mujoco_compatibility=False,
            # dt=kwargs.get('dt', 0.01),
        )  
        
        # 创建场景实例
        self.scene = gs.Scene(
            viewer_options=viewer_options,
            rigid_options=rigid_options,
            show_viewer=self.vis,
            show_FPS=False,
        )

        # 加载机器人, 索引控制具体的关节
        self.robot, self.end_effector, self.motors_dof, self.fingers_dof = self.load_robot()

        # 初始化实体存储,存地板和物体
        self.entities = {}
        self.init_state = {}
        self.cameras = {
            'head_camera': None,
        }

        self.data_type = kwargs.get('data_type', {
            'rgb': True,
            'depth': False,
            'normal': False,
            'third_view': False,
        })
        
        # 加载基础场景元素
        self._load_plane()
        # self._load_default_table()
        self.add_objects()
        self.add_cameras()

        self.FRAME_IDX = 0
        self.play_counter = 0
        self.render_idx = 0
        self.record_freq = 10
        self.debug_dir = kwargs.get('debug_dir', None)


        # 构建场景
        self.scene.build()
        self.scene.reset()

        self.set_robot_dof()
        self.record_objects_init_state()

        os.makedirs(self.save_dir, exist_ok=True)
        # self.start_pos = np.array([0.0005, -0.78746, -0.00032, -2.351319, 0.00013, 1.57421, 0.789325,0.04,0.04])

    def add_objects(self, **kwargs):
        """添加对象"""
        self.entities['cube'] = self.scene.add_entity(
                                        morph=gs.morphs.Mesh(
                                            # file=f"{asset_path}/allegro_hand/allegro_hand_right_glb.urdf",
                                            file="./assets/wood_cube.glb",
                                            pos=(0.24623267, -0.04144618, 0.03),
                                            euler=(0.0, 0.0, 0.0),
                                            scale=0.06,
                                            # decimate=True,
                                            # convexify=False,
                                            # decimate_face_num=50,
                                        ),
                                        material=gs.materials.Rigid(
                                            rho=400,
                                            # friction=2.0,  # 增大基础摩擦系数
                                        ),
                                        # surface=gs.surfaces.Default(
                                        #     # color = (0.8, 0.8, 0.8),
                                        #     vis_mode = 'collision',
                                        #     # friction=0.5,
                                        # ),
                                        # visualize_contact=True
                                    )

        self.entities['box'] = self.scene.add_entity(
                                        morph=gs.morphs.Mesh(
                                            # file=f"{asset_path}/allegro_hand/allegro_hand_right_glb.urdf",
                                            file="./assets/blue_box.glb",
                                            pos=(0.21407672, -0.26041815,  0.04),
                                            euler=(90, 0.0, 0),
                                            scale=0.2,
                                            # decimate=True,
                                            convexify=True,
                                            fixed=True,
                                            # decimate_face_num=50,
                                        ),
                                        # material=gs.materials.Rigid(
                                        #     rho=400,
                                        #     # friction=2.0,  # 增大基础摩擦系数
                                        # ),
                                        # surface=gs.surfaces.Default(
                                        #     # color = (0.8, 0.8, 0.8),
                                        #     vis_mode = 'collision',
                                        #     # friction=0.5,
                                        # ),
                                        
                                        # visualize_contact=True
                                    )


        # self.entities["cube"] = self.scene.add_entity(
        #                                 #material=gs.materials.Rigid(rho=300),
        #                                 morph=gs.morphs.Box(
        #                                     pos=(0.1, 0.0, 0.02),
        #                                     size=(0.04, 0.04, 0.04),
        #                                 ),
        #                                 surface=gs.surfaces.Default(color=(0.5, 1, 0.5)),
        #                             )
    
    def record_objects_init_state(self, state_dict=None):
        for entity_name in self.entities.keys():
            if entity_name != 'plane' and entity_name != 'table':
                self.init_state[entity_name] = {}
                self.init_state[entity_name]['pos'] = self.entities[entity_name].get_pos().cpu().numpy()
                self.init_state[entity_name]['quat'] = self.entities[entity_name].get_quat().cpu().numpy()

        # 也可以人工传数state_dict，覆盖原来的
        if state_dict is not None:
            for entity_name in state_dict: 
                self.init_state[entity_name] = {}
                self.init_state[entity_name]['pos'] = state_dict[entity_name]['pos']
                self.init_state[entity_name]['quat'] = state_dict[entity_name]['quat']

    def set_objects_init_state(self):
        for entity_name in self.entities.keys():
            if entity_name != 'plane' and entity_name != 'table':
                self.entities[entity_name].set_pos(self.init_state[entity_name]['pos'])
                self.entities[entity_name].set_quat(self.init_state[entity_name]['quat'])
                
                            
    def set_robot_dof(self, robot_name="franka"):
        if robot_name=="franka":
            self.robot.set_dofs_kp(
                np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
                )
            self.robot.set_dofs_kv(
                np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
            )
            self.robot.set_dofs_force_range(
                np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
                np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
            )
        elif robot_name=="piper":
            self.robot.set_dofs_kp(np.array([6000, 6000, 5000, 5000, 3000, 3000, 200, 200]))
            self.robot.set_dofs_kv(np.array([150, 150, 120, 120, 80, 80, 10, 10]))
            self.robot.set_dofs_force_range(
                np.array([-87, -87, -87, -87, -12, -12, -100, -100]),
                np.array([87, 87, 87, 87, 12, 12, 100, 100]),
            )
        else:
            raise ValueError(f"不支持的机械臂类型: {robot_name}")


    def _load_plane(self):
        """加载地面平面"""
        # self.entities['plane'] = self.scene.add_entity(
        #     gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True),
        # )
        self.scene.add_entity(
            gs.morphs.URDF(file="./assets/plane/plane.urdf", fixed=True),
        )

    def _load_default_table(self):
        """加载默认桌子"""
        self.scene.add_entity(
            morph=gs.morphs.MJCF(
                file="./assets/table2.xml",
                pos=(0.45, 0.0, 0.0),
                euler=(0, 0, 0),
                collision=True,
                convexify=True,
            ),
        )

    def load_robot(self, robot_type='franka', **kwargs):
        """加载机械臂，默认是Franka"""
        if robot_type == 'franka':
            robot = self.scene.add_entity(
                morph=gs.morphs.MJCF(
                    file="./assets/franka_emika_panda/panda.xml",
                    pos=kwargs.get('robot_pos', (0.2, -0.75, 0.0)),
                    # pos=kwargs.get('robot_pos', (-0.2, -0.3, 0.0)),
                    euler=(0, 0, 90),
                    collision=True,
                ),
            )
            # set control gains
            end_effector = robot.get_link("hand_tcp")
            motors_dof = np.arange(7)
            fingers_dof = np.arange(7, 9)
            self.start_pos = np.array([0.0005, -0.78746, -0.00032, -2.351319, 0.00013, 1.57421, 0.789325,0.04,0.04])
        elif robot_type == 'piper':
            robot = self.scene.add_entity(
                morph=gs.morphs.MJCF(
                    file="/home/haichao/workspace/mujoco_menagerie/agilex_piper/piper.xml",
                    pos=kwargs.get('robot_pos', (0.2, -0.75, 0.0)),
                    # pos=kwargs.get('robot_pos', (-0.2, -0.1, 0.0)),
                    euler=(0, 0, 0),
                    collision=True,
                ),
            ) 
            end_effector = robot.get_link("hand_tcp")
            motors_dof = np.arange(8)
            fingers_dof = np.arange(6, 8)
            self.start_pos = np.array([0] * 6 + [0.04,0.04])
        else:
            raise ValueError(f"不支持的机械臂类型: {robot_type}")
    
        return robot, end_effector, motors_dof, fingers_dof

    def add_cameras(self, **kwargs):
        """添加相机"""
        self.cameras['head_camera'] = self.scene.add_camera(
                res=kwargs.get('res', (640, 480)),
                pos=kwargs.get('pos', (0, 0, 1.0)),
                lookat=kwargs.get('lookat', (0.2, 0.0, 0.0)),
                fov=kwargs.get('fov', 53),
                GUI=kwargs.get('GUI', False),
            )

        self.cameras['front_camera'] = self.scene.add_camera(
                res=kwargs.get('res', (640, 480)),
                pos=kwargs.get('pos', (1.2, 0, 1.0)),
                lookat=kwargs.get('lookat', (0.0, 0.0, 0.0)),
                fov=kwargs.get('fov', 53),
                GUI=kwargs.get('GUI', False),
            )
    
    def start_recording(self, debug_camera=None):
        os.makedirs(self.folder_path + "/videos",exist_ok=True)
        if debug_camera is None:
            for camera_name in self.cameras.keys():
                self.cameras[camera_name].start_recording()
        else:
            self.cameras[debug_camera].start_recording()
    
    def stop_recording(self, debug_camera=None):
        if debug_camera is None:
            for camera_name in self.cameras.keys():
                self.cameras[camera_name].stop_recording(self.folder_path + f"/videos/{camera_name}.mp4", fps=30)
        else:
            self.cameras[debug_camera].stop_recording(self.folder_path + f"/videos/{debug_camera}.mp4", fps=30)


    def reset_robot(self):
        self.move_to_start()
        self.update_scene(10)
        # self.open_gripper()
        # self.update_scene(10)


    def random_objects(self):
        # for entity_name, states in self.init_state.items():
        #     self.entities[entity_name].set_pos(states["pos"])
        #     self.entities[entity_name].set_quat(states["quat"])
        block_pose = rand_pose(
            xlim=[0.1, 0.35],
            ylim=[0.0, 0.3],
            zlim=[0.05],
            quat=[1, 0, 0, 0],
            rotate_rand=True,
            rotate_lim=[0, 0, 30],
        )
        # while abs(block_pose[0][0]) < 0.05 or np.sum(pow(block_pose.[0][:2], 2)) < 0.001:
        #     block_pose = rand_pose(
        #         xlim=[-0.25, 0.25],
        #         ylim=[-0.05, 0.15],
        #         zlim=[0.0],
        #         quat=[1, 0, 0, 0],
        #         rotate_rand=True,
        #         rotate_lim=[0, 0, 0.5],
        #     )

        self.init_state['cube']['pos'] = block_pose[0]
        self.init_state['cube']['quat'] = block_pose[1]

        # self.entities['cube'].set_pos(block_pose[0])
        # self.entities['cube'].set_quat(block_pose[1])


    def reset_env(self):
        self.scene.reset()
        self.reset_robot()
        self.set_robot_dof()
        self.set_objects_init_state()
        self.FRAME_IDX = 0
        self.render_idx = 0
        # self.scene.build(compile_kernels=False)

    def generate_data(self, human_traj, grasp_pose):
        
        rand_num = 30
        rand_pose_counter = {id:0 for id in range(rand_num)}

        # 1.先随机物体位置
        for i in range(rand_num):
            if i > 0:
                self.random_objects()
            # self.record_objects_init_state()

            # 2.再遍历抓取位姿
            for idx in range(len(grasp_pose)):
                pose = grasp_pose[idx]
                pose = transform_grasp_pose_euler(pose)
                position, quat, euler_angles = decompose_grasppose(pose)
                # 3.根据人类轨迹批量生成轨迹
                traj_list = self.generate_traj(human_traj)

                for traj in traj_list:
                    self.reset_env()
                    success = self.play_traj(traj.copy(), np.array([0,1,0,0]),save_data=False)
                    if success:
                        self.reset_env()
                        self.play_traj(traj.copy(), np.array([0,1,0,0]), save_data=True)
                        print(f"grasp [0,1,0,0] 成功")
                        rand_pose_counter[i] += 1

                    self.reset_env()
                    success = self.play_traj(traj.copy(), quat, save_data=False)
                    if success:
                        self.reset_env()
                        self.play_traj(traj.copy(), quat, save_data=True)
                        print(f"grasp {euler_angles} 成功")
                        rand_pose_counter[i] += 1       
                    
                    print(f"物体当前第{i}的pose, 已保存{self.ep_num}个运动轨迹")
                    time.sleep(1)
        
        print(f"每个rand pose下的成功次数统计: {rand_pose_counter}")


    def play_traj(self, traj, grasp_quat, save_data=False):
        '''
        首先回起始位置
        1. 先根据grasp_pose将机械臂移动到初始位置
        2. 再移动机械臂到目标位置
        '''
        self.save_data = save_data
        print(traj.shape)
        for idx in range(len(traj)):
            if idx == 0:
                traj[idx] = self.entities['cube'].get_pos().cpu().numpy()
                print(f"traj {traj[idx]}")
                path = self.plan_to_start(traj[idx] + np.array([0.0,0,0.08]), grasp_quat) # + np.array([0.1,-0.1,0.2])
                self.update_scene(10, record=False)
                self.record_data_once()
                for i in range(5):
                    self.move_to_target(traj[idx], grasp_quat, open_gripper=True)
                    self.update_scene(10, record=False)
                    if (i + 1) % 2 == 0:
                        self.record_data_once()
                self.record_data_once()
                for i in range(5):
                    self.close_gripper()
                    self.update_scene(10, record=False)
                    if (i + 1) % 2 == 0:
                        self.record_data_once()
            else:
                for i in range(2):
                    self.move_to_target(traj[idx], open_gripper=False)
                    self.update_scene(10, record=False)
                self.record_data_once()

        for i in range(5):
            self.open_gripper()
            self.update_scene(20, record=False)
            # self.record_data_once()
            if (i + 1) % 2 == 0:
                self.record_data_once()
        
        for i in range(5):
            self.move_to_target(traj[-1] + np.array([0.0,0,0.05]), open_gripper=True)
            self.update_scene(20, record=False)
            self.record_data_once()

        self.record_data_once(export_video=True)
        target_pos = traj[-1].copy()
        target_pos[-1] = 0

        if save_data:
            self.ep_num += 1

        self.play_counter += 1
        print(f"已保存 / 执行 {self.ep_num}/{ self.play_counter} 轨迹")

        return self.check_success(target_pos)
    
    def move_to_start(self):
        ## reset Franka Position
        self.robot.set_qpos(self.start_pos)
        self.robot.control_dofs_position(self.start_pos)

    def plan_to_start(self, pos, quat):
        # move to pre-grasp pose
        # wxyz
        qpos = self.robot.inverse_kinematics(
            link = self.end_effector,
            pos  = pos,
            quat = quat,
        )
        # gripper open pos
        # qpos[-3:] = 0.789325
        qpos[-2:] = 0.04
        path = self.robot.plan_path(
            qpos_goal     = qpos,
            num_waypoints = 100, # 2s duration
        )

        if len(path) == 0:
            return None

        # execute the planned path
        for i in range(len(path)):
            waypoint = path[i]
            self.robot.control_dofs_position(waypoint)
            self.update_scene(1, record=True)
            # if i % 5:
            #     self.record_data_once()
            # self.scene.step()

        return path

    def move_to_target(self, target_pos, target_quat=None, open_gripper=True):
        if target_quat is not None:
            """移动机械臂到指定位置"""
            qpos, err = self.robot.inverse_kinematics(
                link=self.end_effector,
                pos=target_pos,
                quat=target_quat,
                return_error=True,
            )
            # qpos[-3:] = 0.789325
            # print(f"逆解误差:{err.sum():.6f}")
            # self.robot.set_qpos(qpos[:-2], self.motors_dof)
        else:
            """移动机械臂到指定位置"""
            qpos, err = self.robot.inverse_kinematics(
                link=self.end_effector,
                pos=target_pos,
                return_error=True)

        if open_gripper:
            qpos[-2:] = 0.04
        else:
            qpos[-2:] = 0.0

        # self.robot.set_qpos(qpos)
        self.robot.control_dofs_position(qpos)
        # self.robot.control_dofs_position(qpos[:-2], self.motors_dof)
        # print(self.end_effector.get_quat(), target_quat)

        # self.target_entity.set_qpos(np.concatenate([target_pos, self.end_effector.get_quat().cpu().numpy()]))

        # if open_gripper:
        #     self.open_gripper()
        # else:
        #     self.close_gripper()


    def open_gripper(self, force=None):
        self.robot.set_dofs_position(np.array([0.04, 0.04]), self.fingers_dof)
        # self.robot.control_dofs_force(np.array([0.5, 0.5]), self.fingers_dof)
        

    def close_gripper(self, force=None):
        # self.robot.set_dofs_position(np.array([0.0, 0.0]), self.fingers_dof)
        self.robot.control_dofs_force(np.array([-0.6, -0.6]), self.fingers_dof)


    def save_image(self, image, image_path):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        is_success = cv2.imwrite(image_path, image)

        return is_success


    def update_scene(self, step=2, record=False, debug_camera=None):
        for i in range(1, step + 1):   
            self.scene.step()
            if record and self.render_idx % self.record_freq == 0:
                self.record_data_once()
                return
            if debug_camera is not None:
                self.cameras[debug_camera].render()
            self.render_idx += 1


    def record_data_once(self, export_video=False):  # save data
        if not self.save_data:
            return

        if self.FRAME_IDX == 0:
            # self.folder_path = {"cache": f"{self.save_dir}/episode{self.ep_num}/"}
            self.folder_path = f"{self.save_dir}/{self.task_name}/episode{self.ep_num}/"
            if self.debug_dir is not None:
                self.folder_path = f"{self.debug_dir}/{self.task_name}/episode{self.ep_num}/"
            self.start_recording()
            # for directory in self.folder_path.values():  # remove previous data
            # if os.path.exists(self.folder_path):
            #     file_list = os.listdir(self.folder_path)
            #     # for file in file_list:
            #     #     # os.remove(directory + file)
            #     #     os.remove(file)
            # else:
            for cam_name in self.cameras.keys():
                rgb_path = self.folder_path + f"/rgb/{cam_name}/"
                os.makedirs(rgb_path, exist_ok=True)
                
            os.makedirs(self.folder_path + "/pkl", exist_ok=True)
            

        pkl_dic = self.get_obs()
        # if self.save_rgb_origin:
        #     for cam_name in pkl_dic["observation"].keys():
        #         rgb_path = self.folder_path + f"/rgb/{cam_name}/"
        #         flag = self.save_image(pkl_dic["observation"][cam_name]['rgb'], f"{rgb_path}/{self.FRAME_IDX}.png")
        
        self.save_pkl(pkl_dic, self.folder_path + f"/pkl/{self.FRAME_IDX}.pkl")  # use cache
        print("saving: episode = ", self.ep_num, " index = ", self.FRAME_IDX, end="\r")
        
        self.FRAME_IDX += 1
        del pkl_dic
        if export_video:
            # self._export_camera_videos()
            self.stop_recording()
    
    def _export_camera_videos(self):
        """使用ffmpeg将每个相机的图片序列合成为视频"""
        import subprocess
        import glob

        # 获取所有相机名称
        cam_rgb_dir = self.folder_path + "/rgb/"
        if not os.path.exists(cam_rgb_dir):
            print("No RGB images found, skip video export")
            return

        cam_names = os.listdir(cam_rgb_dir)
        for cam in cam_names:
            img_dir = os.path.join(cam_rgb_dir, cam)
            img_pattern = os.path.join(img_dir, "%d.png")  # 匹配0.png, 1.png...
            video_path = os.path.join(self.folder_path, "videos", f"{cam}.mp4")
            os.makedirs(self.folder_path + "/videos",exist_ok=True)

            # 检查是否有图片
            if not glob.glob(os.path.join(img_dir, "*.png")):
                print(f"No images found for camera {cam}, skip video")
                continue

            # ffmpeg命令：图片转视频（25fps，H.264编码）
            cmd = [
                "ffmpeg",
                "-y",  # 覆盖已有文件
                "-framerate", "25",  # 帧率
                "-i", img_pattern,  # 输入图片路径模式
                "-c:v", "libx264",  # 视频编码器
                "-pix_fmt", "yuv420p",  # 像素格式（兼容多数播放器）
                "-crf", "23",  # 质量控制（0-51，23为默认）
                video_path
            ]

            try:
                # 执行命令
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(f"Video saved for camera {cam}: {video_path}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to generate video for camera {cam}: {e.stderr.decode()}")


    def get_obs(self):
        pkl_dic = {
            "observation": {},
            "pointcloud": [],
            "joint_action": {},
            "endpose": {},
        }
        qpos = self.robot.get_qpos()
        end_pose = self.end_effector.get_pos()
        end_quat = self.end_effector.get_quat()

        if isinstance(qpos,torch.Tensor):
            qpos = qpos.cpu().numpy()
            end_pose = end_pose.cpu().numpy()
            end_quat = end_quat.cpu().numpy()

        # 返回四个render(rgb=True, depth=False, segmentation=False, colorize_seg=False, normal=False)
        '''
        rgb_arr (np.ndarray) – The rendered RGB image.

        depth_arr (np.ndarray) – The rendered depth image.

        seg_arr (np.ndarray) – The rendered segmentation mask.

        normal_arr (np.ndarray) – The rendered surface normal.
        '''
        # pkl_dic["observation"] = self.cameras.get_config()
        # rgb
        if self.data_type.get("rgb", True):
            for camera_name in self.cameras.keys():
                if camera_name not in pkl_dic["observation"]:
                    pkl_dic["observation"][camera_name] = {}
                pkl_dic["observation"][camera_name]['rgb'] = self.cameras[camera_name].render(rgb=True)[0]

        if self.data_type.get("third_view", False):
            pkl_dic["observation"]["third_view_rgb"] = self.cameras["third_view"].render(rgb=True)[0]
            # pkl_dic["third_view_rgb"] = third_view_rgb

        # mesh_segmentation
        if self.data_type.get("normal", False):
            for camera_name in self.cameras.keys():
                pkl_dic["observation"][camera_name]['normal'] = self.cameras[camera_name].render(rgb=True, normal=True)[3]

        # depth
        if self.data_type.get("depth", False):
            for camera_name in self.cameras.keys():
                pkl_dic["observation"][camera_name]['depth'] = self.cameras[camera_name].render(rgb=True, depth=True)[1]

        # endpose
        if self.data_type.get("endpose", True):
            # qpos = self.robot.get_qpos()
            # end_pose = self.end_effector.get_pos()
            # end_quat = self.end_effector.get_quat()

            pkl_dic["endpose"]["endpose"] = np.concatenate([end_pose, end_quat])
            pkl_dic["endpose"]["gripper"] = qpos[-2:]

            # norm_gripper_val = [
            #     self.robot.get_left_gripper_val(),
            #     self.robot.get_right_gripper_val(),
            # ]
            # left_endpose = self.robot["right_arm"]['end_effector'].get_pos()
            # right_endpose = self.robot["left_arm"]['end_effector'].get_pos()
            # pkl_dic["endpose"]["left_endpose"] = left_endpose
            # pkl_dic["endpose"]["left_gripper"] = norm_gripper_val[0]
            # pkl_dic["endpose"]["right_endpose"] = right_endpose
            # pkl_dic["endpose"]["right_gripper"] = norm_gripper_val[1]

        if self.data_type.get("qpos", True):
            pkl_dic["joint_action"] = qpos
            # left_jointstate = self.robot.get_left_arm_jointState()
            # right_jointstate = self.robot.get_right_arm_jointState()
            # pkl_dic["joint_action"]["left_arm"] = left_jointstate[:-1]
            # pkl_dic["joint_action"]["left_gripper"] = left_jointstate[-1]
            # pkl_dic["joint_action"]["right_arm"] = right_jointstate[:-1]
            # pkl_dic["joint_action"]["right_gripper"] = right_jointstate[-1]
            # pkl_dic["joint_action"]["vector"] = np.array(left_jointstate + right_jointstate)
            
        # pointcloud
        # if self.data_type.get("pointcloud", False):
        #     pkl_dic["pointcloud"] = self.cameras.get_pcd(self.data_type.get("conbine", False))

        # self.now_obs = deepcopy(pkl_dic)
        return pkl_dic


    def check_success(self, target_pos):
        # 获取方块的AABB
        cube_aabb = self.entities['cube'].get_AABB()
        box_aabb = self.entities['box'].get_AABB()

        # 解包并仅比较XY坐标
        (cx1, cy1, _), (cx2, cy2, _) = cube_aabb
        (bx1, by1, _), (bx2, by2, _) = box_aabb

        return (cx1 >= bx1 - 1e-2 and cx2 <= bx2 + 1e-2 and
            cy1 >= by1 - 1e-2 and cy2 <= by2 + 1e-2)

        cur_pos = self.entities['cube'].get_pos()
        if isinstance(cur_pos, torch.Tensor):
            cur_pos = cur_pos.cpu().numpy()

        cur_pos[-1] = 0
        if abs((cur_pos - target_pos).sum()) < 0.05:
            return True
  
        return False

    def save_pkl(self, pkl_dic, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(pkl_dic, f)

    def generate_traj(self, hand_traj):
        hand_traj, _ = self.handtraj_processor.get_center_trajectory(hand_traj, is_list_format=True)
        traj_tmp = []
        if self.handtraj_processor is None:
            raise ValueError("handtraj_processor is None, can't get traj")

        start_pos = self.init_state['cube']['pos']
        end_pos = self.init_state['box']['pos']

        if isinstance(start_pos, torch.Tensor):
            start_pos = start_pos.cpu().numpy().astype(np.float64)
            end_pos = end_pos.cpu().numpy().astype(np.float64) + np.array([0,0,0.04])

        # if isinstance(end_pos, torch.Tensor):

        # promp_traj, _ = self.handtraj_processor.apply_promp_old(hand_traj.copy(), start_pos=start_pos, end_pos=end_pos)
        # if not np.isnan(promp_traj).any():
        #     traj_tmp.append(promp_traj)

        promp_traj, _ = self.handtraj_processor.apply_promp_old(hand_traj.copy(), start_pos=start_pos, end_pos=end_pos, random_samples=2)

        for i in range(len(promp_traj)):
            if not np.isnan(promp_traj[i]).any():
                traj_tmp.append(promp_traj[i])
            else:
                print(start_pos, end_pos)
                print("存在Nan 轨迹")

        # 让原来的轨迹穿过人类噪声点，换起始点和终点时就没用了 
        # diff_points, diff_times = self.handtraj_processor.find_trajectory_diff(
        #                                     hand_traj, promp_traj, 
        #                                     tolerance=1e-2, 
        #                                     sample_sets=2, 
        #                                     samples_set_points=3, 
        #                                     vis=False
        #                                 )
        
        # for idx in range(len(diff_points)):
        #     via_points_data = {
        #             'ts': diff_times[idx],
        #             'y_cond': diff_points[idx]
        #         }
        #     # try:
            
        #     # promp_traj, _ = self.handtraj_processor.apply_promp_old(hand_traj.copy(), start_pos=start_pos, end_pos=end_pos,via_points_data=via_points_data)

            
            
        #     if not np.isnan(promp_traj).any():
        #         traj_tmp.append(promp_traj)
        #     # except Exception as e:
        #     #     print(traceback.format_exc())
        #     #     import ipdb
        #     #     ipdb.set_trace()
        #     #     continue
                
        return traj_tmp
    
      
    def test_env(self):
        for i in range(1000):
            self.update_scene(10, record=False)


   
# 示例用法
if __name__ == "__main__":
    import argparse


    hand_path = './stero_hand_3d.pkl'
    pose_path = "../GraspGen/valid_pose.npy"
    finger_verts_path = "../GraspGen/valid_vertices.npy"

    parser = argparse.ArgumentParser()
    parser.add_argument("--hand_traj",type=str, required=False, default="hand,mug,tomato")
    parser.add_argument("--grasp_path",type=str, required=False, default="hand,mug,tomato")
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    args = parser.parse_args()


    # 加载数据（根据实际情况修改路径）
    hand3d = pickle.load(open(hand_path, 'rb'))
    hand3d = hand3d[60:165]

    handtraj_processor = HandTrajProcess(
        dmp_execution_time=2.0,
        dmp_n_weights=100,
        promp_n_weights=10
    )


    # 获取手部中心轨迹
    # center_traj, valid_frames = handtraj_processor.get_center_trajectory(hand3d, is_list_format=True,smooth=5)
    # print(f"提取到{len(center_traj)}帧有效轨迹数据")

    # 加载目标位姿
    grasp_pose = np.load(pose_path)
    finger_verts = np.load(finger_verts_path)

    # 创建环境实例
    env_kwargs = {
        "seed": 42,
        "task_name": "object_manipulation",
        "save_path": "task_data",
        "save_data":False,
        "handtraj_processor":handtraj_processor
    }

    ########################## env ##########################
    env = BaseEnv(vis=args.vis, **env_kwargs)



    env.generate_data(hand3d, grasp_pose)
    i = 0
    # 使用原始起点生成轨迹
    # for idx in range(len(grasp_pose)):
    #     pose = grasp_pose[idx]
    #     pose = transform_grasp_pose_euler(pose)
    #     position, quat, euler_angles = decompose_grasppose(pose)


    #     promp_traj, _ = handtraj_processor.apply_promp_old(center_traj, start_pos=position.astype(np.float64))
    #     dmp_traj = handtraj_processor.apply_dmp(promp_traj, start_pos=position.astype(np.float64))
        
    #     print(promp_traj.shape)
    #     # promp_original, _ = traj_processor.apply_promp(center_traj)

    #     handtraj_processor.visualize_primitives(
    #         original_traj=center_traj,
    #         dmp_traj=dmp_traj,
    #         promp_traj=promp_traj,
    #         output_path="results/original_starts.png",
    #         show_plot=True
    #     )

    #     print(f"第{i}个目标位姿")
    #     i += 1
    #     diff_points, diff_times = handtraj_processor.find_trajectory_diff(center_traj, promp_traj, tolerance=1e-2, sample_sets=2, samples_set_points=3, vis=False)

    #     for idx in range(len(diff_points)):
    #         via_points_data = {
    #             'ts': diff_times[idx],
    #             'y_cond': diff_points[idx]
    #         }

    #         success = env.play_traj(promp_traj, np.array([0,1,0,0]))
    #         if success:
    #             env.play_traj(promp_traj, np.array([0,1,0,0]), save_data=True)
    #             print(f"[0,1,0,0] 成功")
        
    #         success = env.play_traj(promp_traj, quat)
    #         if success:
    #             env.play_traj(promp_traj, quat, save_data=True)
    #             print(f"{euler_angles} 成功")


    #         promp_traj, _ = handtraj_processor.apply_promp(center_traj, start_pos=position.astype(np.float64), via_points_data=via_points_data)
    #         print(promp_traj.shape)
    #         success = env.play_traj(promp_traj, quat)
    #         if success:
    #             env.play_traj(promp_traj, quat, save_data=True)
    #             print(f"sample [0,1,0,0] 成功")            
            
    #         print(f"已保存{env.ep_num}个运动轨迹")

        #  测试位置对不对
        # print(finger_verts[idx])
        # env.move_to_start()
        # env.move_to_fingers(finger_verts[idx], quat[[3,0,1,2]])
        # env.play_traj([position], quat_euler_converter(np.array([0,1,0,0]), np.array([0,0,45])))

    gs.destroy()
    import gc
    gc.collect()

    




    

