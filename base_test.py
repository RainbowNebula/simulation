import os
import numpy as np
import torch
from glob import glob
import genesis as gs
import sys

sys.path.append('/home/haichao/workspace/real2sim')
from api.handtraj_api_dmp import HandTrajProcess  # 初始化处理器，可配置DMP和ProMP参数
import pickle
from scipy.spatial.transform import Rotation as R

import cv2
from copy import deepcopy
import time

from transform_grasp import transform_grasp_poses
def quat_euler_converter(quaternion, euler_angles_deg, return_type='quat'):
    """
    将四元数与角度制欧拉角组合，返回旋转矩阵或四元数
    
    参数:
        quaternion: 四元数，格式为[x, y, z, w]
        euler_angles_deg: 角度制XYZ欧拉角，格式为[rx_deg, ry_deg, rz_deg]
        return_type: 返回类型，'matrix'返回旋转矩阵，'quaternion'返回四元数[x, y, z, w]
    
    返回:
        3x3旋转矩阵或四元数列表
    """
    # 从四元数创建旋转对象
    rot_quat = R.from_quat(quaternion)
    
    # 将角度制欧拉角转换为弧度
    euler_angles_rad = np.radians(euler_angles_deg)
    
    # 从欧拉角创建旋转对象 (xyz顺序)
    rot_euler = R.from_euler('xyz', euler_angles_rad)
    
    # 组合两个旋转: 先应用四元数旋转，再应用欧拉角旋转
    total_rot = rot_quat * rot_euler 
    
    # 根据返回类型选择输出
    if return_type == 'matrix':
        return total_rot.as_matrix()
    elif return_type == 'quat':
        return total_rot.as_quat()[[3, 0, 1, 2]] # wxyz
    else:
        raise ValueError("return_type必须是'matrix'或'quaternion'")


def decompose_grasppose(grasppose):
    """
    使用库函数将4x4 grasppose变换矩阵分解为四元数、旋转角和平移向量
    
    参数:
        matrix: 4x4 numpy数组，表示变换矩阵
        
    返回:
        元组: (四元数(x,y,z,w), 旋转角(roll,pitch,yaw,弧度), 平移向量(x,y,z))
    """
    # 验证输入
    if grasppose.shape != (4, 4):
        raise ValueError("输入必须是4x4矩阵")
    
    # 提取平移向量
    position = grasppose[:3, 3]
    
    # 提取旋转矩阵并转换
    rot_mat = grasppose[:3, :3]
    rotation = R.from_matrix(rot_mat)
    
    # 四元数 (x,y,z,w)
    quaternion = rotation.as_quat() # scalar_first=False
    
    # 欧拉角 (roll, pitch, yaw)，使用ZYX顺序
    euler_angles = rotation.as_euler('zyx')[::-1]  # 转换为roll, pitch, yaw顺序

    
    return position, quaternion, euler_angles


class BaseEnv:
    def __init__(self, **kwargs):
        super().__init__()

        # 解析配置参数
        self.vis = kwargs.get('vis', False)
        self.seed = kwargs.get('seed', 0)
        self.save_data = kwargs.get('save_data', True)
        self.save_dir = kwargs.get('save_dir', "./sim_data")
        self.save_rgb_origin = kwargs.get('save_rgb_origin', True)
        self.save_depth_origin = kwargs.get('save_depth_origin', False)
        self.task_name = kwargs.get('task_name', 'default')
        self.ep_num = kwargs.get('ep_num', 0)

        # np.random.seed(self.seed)
        # torch.manual_seed(self.seed)

        ########################## init ##########################
        # gs.init(backend=gs.gpu,precision="32",logging_level='debug', debug=False)
        gs.init(backend=gs.gpu)
        
        # 初始化场景配置
        viewer_options = gs.options.ViewerOptions(
            camera_pos=kwargs.get('camera_pos', (3, -1, 1.5)),
            camera_lookat=kwargs.get('camera_lookat', (0.0, 0.0, 0.0)),
            camera_fov=kwargs.get('camera_fov', 30),
            # max_FPS=kwargs.get('max_FPS', 60),
        )
        
        rigid_options = gs.options.RigidOptions(
            dt=kwargs.get('dt', 0.01),
        )
        
        # 创建场景实例
        self.scene = gs.Scene(
            viewer_options=viewer_options,
            rigid_options=rigid_options,
            show_viewer=True,
            show_FPS=False,
        )

        # 加载机器人, 索引控制具体的关节
        self.robot, self.end_effector, self.motors_dof, self.fingers_dof = self.load_robot()

        # 初始化实体存储,存地板和物体
        self.entities = {
            'plane': None,
            'table': None,
            'objects': []
        }

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

        self.sphere = self.scene.add_entity(
            material=gs.materials.Rigid(rho=300, friction=0.5, gravity_compensation=1),
            # morph=gs.morphs.Box(
            #     size=(0.04, 0.04, 0.04),
            #     pos=(0.6, 0.0, 0.01),
            # )
            morph=gs.morphs.Sphere(
                radius=0.03,
                pos=(0.6, 0.0, 0.01),
                collision=False,
                fixed=True,
            ),
            surface=gs.surfaces.Default(
                color=(0.9, 0.8, 0.2, 1.0),
            )
        )
        self.FRAME_IDX = 0

        # 构建场景
        self.scene.build()

        self.set_robot_dof()
        self.start_pos = np.array([0.0005, -0.78746, -0.00032, -2.351319, 0.00013, 1.57421, 0.789325,0.04,0.04])



    def add_objects(self, **kwargs):
        """添加对象"""
        self.entities['tomato'] = self.scene.add_entity(
                                        morph=gs.morphs.Mesh(
                                            # file=f"{asset_path}/allegro_hand/allegro_hand_right_glb.urdf",
                                            file="/home/haichao/workspace/real2sim/data/2025-07-12_15:39/3d_assets/tomato_2.glb",
                                            pos=(0.5 , 0 ,  0.0),
                                            euler=(0.0, 0.0, 0.0),
                                            scale=0.1,
                                        ),
                                        material=gs.materials.Rigid(
                                            rho=10,
                                            friction=2.0,  # 增大基础摩擦系数
                                        ),
                                    )

        self.entities["cube"] = self.scene.add_entity(
                                        material=gs.materials.Rigid(rho=300),
                                        morph=gs.morphs.Box(
                                            pos=(0.1, 0.0, 0.00),
                                            size=(0.04, 0.04, 0.04),
                                        ),
                                        surface=gs.surfaces.Default(color=(0.5, 1, 0.5)),
                                    )
                            
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
        self.entities['plane'] = self.scene.add_entity(
            gs.morphs.URDF(file="/home/haichao/workspace/real2sim/third_party/Genesis/genesis/assets/urdf/plane/plane.urdf", fixed=True),
        )

    def _load_default_table(self):
        """加载默认桌子"""
        self.scene.add_entity(
            morph=gs.morphs.MJCF(
                file="/home/haichao/workspace/real2sim/simulation/assets/table2.xml",
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
                    file="/home/haichao/workspace/real2sim/simulation/assets/franka_emika_panda/panda.xml",
                    pos=kwargs.get('robot_pos', (0.2, -0.75, 0.0)),
                    euler=(0, 0, 90),
                    collision=True,
                ),
            )
            # set control gains
            end_effector = robot.get_link("hand_tcp")
            motors_dof = np.arange(7)
            fingers_dof = np.arange(7, 9)
        else:
            raise ValueError(f"不支持的机械臂类型: {robot_type}")
        
        return robot, end_effector, motors_dof, fingers_dof

    def add_cameras(self, **kwargs):
        """添加相机"""
        self.cameras['head_camera'] = self.scene.add_camera(
                res=kwargs.get('res', (1280, 720)),
                pos=kwargs.get('pos', (0, 0, 0.5)),
                lookat=kwargs.get('lookat', (0.07, 0.0, 0.4292)),
                fov=kwargs.get('fov', 53),
                GUI=kwargs.get('GUI', False),
            )

        # def add_camera(self, res, pos, lookat, up, model, fov, aperture, focus_dist, GUI, spp, denoise, env_idx):
        #     cam_idx = len(self._cameras)
        #     camera = Camera(
        #         self, cam_idx, model, res, pos, lookat, up, fov, aperture, focus_dist, GUI, spp, denoise, env_idx=env_idx
        #     )
        #     self._cameras.append(camera)
        #     return camera

        # self.cameras['left_camera'] = self.scene.add_camera(
        #         res=kwargs.get('res', (1280, 720)),
        #         pos=kwargs.get('pos', (0, 0, 0.8)),
        #         lookat=kwargs.get('lookat', (1.0, 0.0, 0.0)),
        #         fov=kwargs.get('fov', 53),
        #         GUI=kwargs.get('GUI', False),
        #     )

        # self.cameras['right_camera'] = self.scene.add_camera(
        #         res=kwargs.get('res', (1280, 720)),
        #         pos=kwargs.get('pos', (0, 0, 0.8)),
        #         lookat=kwargs.get('lookat', (1.0, 0.0, 0.0)),
        #         fov=kwargs.get('fov', 53),
        #         GUI=kwargs.get('GUI', False),
        #     )

        # 摄像机设置
        # cam_0 = scene.add_camera(res=(1280, 960), pos=(0, 0.5, 1.5), lookat=(0.5, 0.5, 0.0), fov=55, GUI=False)
        # cam_1 = scene.add_camera(res=(1280, 960), pos=(1.5, -0.5, 1.5), lookat=(0.5, 0.5, 0), fov=55, GUI=False)
        # cam_2 = scene.add_camera(res=(1280, 960), pos=(0.5, 1.0, 1.5), lookat=(0.5, 0.5, 0.0), fov=55, GUI=False)
        # return cam


    def _load_assets(self, json_path, table_height=0.85):
        """遍历指定文件夹加载glb物体到桌子上"""
        if not os.path.exists(json_path ):
            raise FileNotFoundError(f"资产配置文件夹不存在: {folder_path}")
            
        # 获取所有glb和stl文件
        # asset_files = glob(os.path.join(folder_path, "*.glb")) + glob(os.path.join(folder_path, "*.stl"))
        
        if not asset_files:
            print(f"在 {folder_path} 中未找到glb或stl文件")
            return

    def reset_env(self):
        self.move_to_start()
        self.open_gripper()
        self.update_scene(50)
        self.FRAME_IDX = 0


    def play_traj(self, traj=None, grasp_quat=None, save_data=False):
        '''
        首先回起始位置
        1. 先根据grasp_pose将机械臂移动到初始位置
        2. 再移动机械臂到目标位置
        '''
        # self.robot.set_dofs_position(self.start_pos, self.motors_dof)
        self.reset_env()
        self.save_data = save_data

        for idx, pos in enumerate(traj):
            if idx == 0:
                self.plan_to_start(pos + np.array([0.0,0,0.03]), grasp_quat) # + np.array([0.1,-0.1,0.2])
                self.sphere.set_pos(pos + np.array([0.0,0,0.03]))
                # self.robot.control_dofs_force(np.array([0.789325]), np.array([self.motors_dof[-1]]))
                # self.move_to_target(pos + np.array([0.0,0,0.1]), grasp_quat) # + np.array([0.1,0,0.1])
                self.move_to_target(pos - np.array([0.0,0,0.03]), open_gripper=True)
                self.entities['cube'].set_pos(traj[0])
                self.update_scene(200) 
                # self.update_scene(100) 
                for i in range(50):
                    self.close_gripper()
                    self.update_scene(1) 
            # else:
            #     self.move_to_target(pos, open_gripper=False)
            #     self.update_scene(10)
            
            self.record_data_once()
                
        self.open_gripper()
        self.update_scene(10)

        self.record_data_once()
       
        target_pos = traj[-1]
        target_pos[-1] = 0
        return self.check_success(target_pos)
    
    def move_to_start(self):
        ## reset Franka Position
        self.robot.set_qpos(self.start_pos)
        self.robot.control_dofs_position(self.start_pos)

    def plan_to_start(self, pos, quat):
        # move to pre-grasp pose
        qpos = self.robot.inverse_kinematics(
            link = self.end_effector,
            pos  = pos,
            quat = quat,
        )
        # gripper open pos
        qpos[-3:] = 0.789325
        qpos[-2:] = 0.04
        path = self.robot.plan_path(
            qpos_goal     = qpos,
            num_waypoints = 100, # 2s duration
        )
        # execute the planned path
        for waypoint in path:
            self.robot.control_dofs_position(waypoint)
            self.scene.step()

    def move_to_target(self, target_pos, target_quat=None, open_gripper=True):
        if target_quat is not None:
            """移动机械臂到指定位置"""
            qpos, err = self.robot.inverse_kinematics(
                link=self.end_effector,
                pos=target_pos,
                quat=target_quat,
                return_error=True,
            )
            qpos[-3:] = 0.789325
            print(f"逆解误差:{err.sum():.6f}")
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

        if open_gripper:
            self.open_gripper()
        else:
            self.close_gripper()


    def open_gripper(self, force=None):
        self.robot.control_dofs_force(np.array([0.5, 0.5]), self.fingers_dof)
        

    def close_gripper(self, force=None):
        # self.robot.set_dofs_position(np.array([0.0, 0.0]), self.fingers_dof)
        self.robot.control_dofs_force(np.array([-0.5, -0.5]), self.fingers_dof)


    def save_image(self, image, image_path):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        is_success = cv2.imwrite(image_path, image)


    def update_scene(self, step=2):
        for i in range(step):   
            self.scene.step()


    def record_data_once(self):  # save data
        if not self.save_data:
            return

        print("saving: episode = ", self.ep_num, " index = ", self.FRAME_IDX, end="\r")
        if self.FRAME_IDX == 0:
            # self.folder_path = {"cache": f"{self.save_dir}/episode{self.ep_num}/"}
            self.folder_path = f"{self.save_dir}/{self.task_name}/episode{self.ep_num}/"
            # for directory in self.folder_path.values():  # remove previous data
            # if os.path.exists(self.folder_path):
            #     file_list = os.listdir(self.folder_path)
            #     # for file in file_list:
            #     #     # os.remove(directory + file)
            #     #     os.remove(file)
            # else:
            os.makedirs(self.folder_path + "/pkl", exist_ok=True)
            

        pkl_dic = self.get_obs()
        if self.save_rgb_origin:
            for cam_name in pkl_dic["observation"].keys():
                rgb_path = self.folder_path + f"/rgb/{cam_name}/"
                os.makedirs(rgb_path, exist_ok=True)
                self.save_image(pkl_dic["observation"][cam_name]['rgb'], f"{rgb_path}/{self.FRAME_IDX}.png")

        
        self.save_pkl(pkl_dic, self.folder_path + f"/pkl/{self.FRAME_IDX}.pkl")  # use cache
        # save_pkl(self.folder_path["cache"] + f"{self.FRAME_IDX}.pkl", pkl_dic)  # use cache
        self.FRAME_IDX += 1


    def get_obs(self):
        pkl_dic = {
            "observation": {},
            "pointcloud": [],
            "joint_action": {},
            "endpose": {},
        }

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
            pkl_dic["third_view_rgb"] = third_view_rgb

        # mesh_segmentation
        if self.data_type.get("normal", False):
            for camera_name in self.cameras.keys():
                pkl_dic["observation"][camera_name]['normal'] = self.cameras[camera_name].render(rgb=False, normal=True)[3]

        # depth
        if self.data_type.get("depth", False):
            for camera_name in self.cameras.keys():
                pkl_dic["observation"][camera_name]['depth'] = self.cameras[camera_name].render(rgb=False, depth=False)[1]

        # endpose
        if self.data_type.get("endpose", False):
            qpos = self.robot.get_pos()
            end_pose = self.end_effector.get_pos()
            end_quat = self.end_effector.get_quat()

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

        if self.data_type.get("qpos", False):
            qpos = self.robot.get_pos()
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

        self.now_obs = deepcopy(pkl_dic)
        return pkl_dic


    def check_success(self, target_pos):

        cur_pos = self.entities['cube'].get_pos()

        if isinstance(cur_pos, torch.Tensor):
            cur_pos = cur_pos.cpu().numpy()

        if abs((cur_pos - target_pos).sum()) < 0.05:
            self.ep_num += 1
            return True
  
        return False

    def save_pkl(self, pkl_dic, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(pkl_dic, f)
   

# 示例用法
if __name__ == "__main__":
    import argparse


    hand_path = '/home/haichao/workspace/real2sim/simulation/data/20250716-2018/video/demo_stero/stero_hand_3d.pkl'
    pose_path = "/mnt/nas/liuqipeng/workspace/GraspGen/valid_pose.npy"
    finger_verts_path = "/mnt/nas/liuqipeng/workspace/GraspGen/valid_vertices.npy"

    parser = argparse.ArgumentParser()
    parser.add_argument("--hand_traj",type=str, required=False, default="hand,mug,tomato")
    parser.add_argument("--grasp_path",type=str, required=False, default="hand,mug,tomato")
    parser.add_argument("-v", "--vis", action="store_true", default=True)
    args = parser.parse_args()

    # 创建环境实例
    env_kwargs = {
        "seed": 42,
        "task_name": "object_manipulation",
        "save_path": "task_data",
        "asset_folder": "/home/haichao/workspace/real2sim/data/2025-06-27/3d_assets/",
        "save_data":False 
    }



    ########################## env ##########################
    env = BaseEnv(vis=args.vis, **env_kwargs)

    handtraj_processor = HandTrajProcess(
        dmp_execution_time=2.0,
        dmp_n_weights=100,
        promp_n_weights=10
    )

    # 加载数据（根据实际情况修改路径）
    hand3d = pickle.load(open('/home/haichao/workspace/real2sim/simulation/data/20250716-2018/video/demo_stero/stero_hand_3d.pkl', 'rb'))
    hand3d = hand3d[60:165]

    # 获取手部中心轨迹
    center_traj, valid_frames = handtraj_processor.get_center_trajectory(hand3d, is_list_format=True)
    print(f"提取到{len(center_traj)}帧有效轨迹数据")

    # 加载目标位姿
    grasp_pose = np.load(pose_path)
    finger_verts = np.load(finger_verts_path)

    base_position = [0.2, -0.75, 0.0]  # 基座位置
    base_rpy = [0, 0, np.radians(90)]  # 基座旋转角 (弧度)

    grasp_poses = transform_grasp_poses(
        base_position, base_rpy, grasp_pose
    )

    i = 0
    # 使用原始起点生成轨迹
    for idx in range(len(grasp_pose)):
        pose = grasp_pose[idx]
        position, quat, _ = decompose_grasppose(pose)

        # if idx == 0:
        #     position = np.array([0.6 , 0.0 ,  0.3])
        #     quat = np.array([1 ,  0, 0, 0])

        dmp_traj = handtraj_processor.apply_dmp(center_traj, start_pos=position.astype(np.float64))
        promp_traj, _ = handtraj_processor.apply_promp(center_traj, start_pos=position.astype(np.float64))
        # promp_original, _ = traj_processor.apply_promp(center_traj)
        print(f"第{i}个目标位姿")
        i += 1
        # handtraj_processor.visualize_primitives(
        #     original_traj=center_traj,
        #     # dmp_traj=dmp_traj,
        #     promp_traj=promp_traj,
        #     output_path="results/original_starts.png",
        #     show_plot=True
        # )
        # success = env.play_traj(promp_traj, np.array([0,1,0,0]))
        # if success:
        #     env.play_traj(promp_traj, np.array([0,1,0,0]), save_data=True)
        #     print(f"[0,1,0,0] 成功")
      
        success = env.play_traj(promp_traj, quat[[3,0,1,2]])
        # if success:
        #     env.play_traj(promp_traj, np.array([0,1,0,0]), save_data=True)
        #     print(f"{quat[[3,0,1,2]]} 成功")
        
        print(f"已保存{env.ep_num}个运动轨迹")
        # print(finger_verts[idx])
        # env.move_to_start()
        # env.move_to_fingers(finger_verts[idx], quat[[3,0,1,2]])
        # env.play_traj([position], quat_euler_converter(np.array([0,1,0,0]), np.array([0,0,45])))

    gs.destroy()

    




    

