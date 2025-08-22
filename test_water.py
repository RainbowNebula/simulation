
import numpy as np
import pickle

from base import *
from utils import augment_grasps_with_interpolation, seed_everything
# from utils import rand_pose, decompose_grasppose
# from transform_grasp import transform_grasp_pose_euler
# from api.handtraj_api_dmp import HandTrajProcess
from scipy.spatial.transform import Rotation as R
from transform_grasp import quat_to_rotation_matrix
import time
seed_everything(60)

# gs.destroy()
# import gc
# gc.collect()

class PullWater(BaseEnv):
    def __init__(self, **kwargs):
        super(PullWater, self).__init__(**kwargs)
        self.save_dir = kwargs.get('save_dir', "./pull_water_h100")


    def add_objects(self, **kwargs):
        """添加对象"""
        frictionless_rigid = gs.materials.Rigid(needs_coup=True, coup_friction=0.0)
        # self.entities['cup'] = self.scene.add_entity(
        #                                 morph=gs.morphs.Mesh(
        #                                     # file=f"{asset_path}/allegro_hand/allegro_hand_right_glb.urdf",
        #                                     file="./assets/redcup.glb",
        #                                     pos=(0.1, 0.1, 0.03),
        #                                     euler=(90, 0.0, 0.0),
        #                                     scale=0.07,
        #                                     # decimate=True,
        #                                     # convexify=False,
        #                                     # decimate_face_num=50,
        #                                 ),
        #                                 material=gs.materials.Rigid(
        #                                     rho=750,
        #                                     needs_coup=True, 
        #                                     coup_friction=0.08,
        #                                     coup_softness=0.02,
        #                                     # friction=2.0,  # 增大基础摩擦系数
        #                                 ),
        #                                 )
        
        self.entities['water'] = self.scene.add_entity(
                                        material=gs.materials.SPH.Liquid(),
                                        morph=gs.morphs.Box(
                                            pos=(0.1, 0.1, 0.1),
                                            size=(0.05, 0.05, 0.05),
                                        ),
                                        surface=gs.surfaces.Default(
                                            color=(0.2, 0.6, 1.0, 1.0),
                                            # vis_mode="particle",
                                        ),
                                    )
        
        self.entities['kuang'] = self.scene.add_entity(
                                        morph=gs.morphs.Mesh(
                                            # file=f"{asset_path}/allegro_hand/allegro_hand_right_glb.urdf",
                                            file="./assets/kuang.glb",
                                            pos=(0.1, 0.1, 0.03),
                                            euler=(90, 0.0, 0.0),
                                            scale=0.1,
                                            # decimate=True,
                                            # convexify=False,
                                            # decimate_face_num=50,
                                        ),
                                        material=gs.materials.Rigid(
                                            rho=1000,
                                            needs_coup=True, 
                                            coup_friction=0.1,
                                            coup_softness=0.01,
                                            # friction=2.0,  # 增大基础摩擦系数
                                        ),

                                        )
        

        self.entities['nailong'] = self.scene.add_entity(
                                        morph=gs.morphs.Mesh(
                                            # file=f"{asset_path}/allegro_hand/allegro_hand_right_glb.urdf",
                                            file="./assets/nailong.glb",
                                            pos=(0.5, 0.5, 0.6),
                                            euler=(0, 0.0, 0.0),
                                            scale=0.1,
                                            # decimate=True,
                                            # convexify=False,
                                            # decimate_face_num=50,
                                            ),
                                        material=gs.materials.Rigid(
                                                    rho=200.0,               # 适中密度，模拟填充毛绒的质量
                                                    friction=1.2,            # 高摩擦，表面不易滑动
                                                    needs_coup=True,         # 允许与其他物体（如手、地面）交互
                                                    coup_friction=0.8,       # 耦合时摩擦较大，抓握感强
                                                    coup_softness=0.01,      # 轻微柔度，碰撞有缓冲感
                                                    coup_restitution=0.1,    # 极低回弹，模拟毛绒吸收冲击
                                                    sdf_cell_size=0.005,     # 中等网格精度，平衡效率与细节
                                                    sdf_min_res=32,          # 基础分辨率
                                                    sdf_max_res=128,         # 足够捕捉玩偶轮廓细节
                                                    # gravity_compensation=0.0 # 正常受重力影响
                                                )   
                                    )



        
        

    def is_grasp_success(self):
        cube_pos = self.entities['cube'].get_pos()
        gripper_pos = self.end_effector.get_pos()
        print(abs((cube_pos - gripper_pos).sum()), cube_pos[2])
        return abs((cube_pos - gripper_pos).sum()) and cube_pos[2] > 0.2
    

    def check_grasp(self, grasp_pose, debug=False):
        final_pos = []
        if debug:
            self.save_dir = "./test_grasp2"
            debug_camera = "front_camera"
        else:
            debug_camera = None

        # 调用插值函数扩展抓取位姿
        grasp_pose = augment_grasps_with_interpolation(grasp_pose, num_interpolations=5)

        # return grasp_pose
        
        for idx in range(len(grasp_pose)):
            if debug:
                self.folder_path = f"{self.save_dir}/{self.task_name}/episode{self.ep_num}/"
            
            pose = grasp_pose[idx]
            pose = transform_grasp_pose_euler(pose)
            position, quat, euler_angles = decompose_grasppose(pose)

            self.reset_env()
            # self.reset_robot()
            # self.set_objects_init_state()
            if debug_camera is not None:
                self.start_recording(debug_camera=debug_camera)

            position = self.entities['cube'].get_pos().cpu().numpy()
            self.plan_to_start(position + np.array([0, 0, 0.1]), quat)
            self.update_scene(30, debug_camera=debug_camera)
            self.move_to_target(position, quat, open_gripper=True)
            self.update_scene(30, debug_camera=debug_camera)
            self.close_gripper()
            self.update_scene(30, debug_camera=debug_camera)
            self.move_to_target(position + np.array([0, 0, 0.25]), quat, open_gripper=False)
            self.update_scene(100, debug_camera=debug_camera)

            if self.is_grasp_success():
                final_pos.append(pose)
                print(f"grasp success {idx + 1} / {len(grasp_pose)}")

            if debug_camera is not None:
                self.stop_recording(debug_camera=debug_camera)
        
        return final_pos
    
    def generate_data(self, human_traj, grasp_pose):
        fail_ep = []
        rand_num = 100
        rand_pose_counter = {id:0 for id in range(rand_num)}

        # 1.先随机物体位置
        for i in range(rand_num):
            if i > 0:
                self.reset_env()
                self.random_objects()
                self.update_scene(10)
            
            # 2.再遍历抓取位姿
            for idx in range(len(grasp_pose)):
                pose = grasp_pose[idx]
                # pose = transform_grasp_pose_euler(pose)
                position, quat, euler_angles = decompose_grasppose(pose)
                # 3.根据人类轨迹批量生成轨迹
                traj_list = self.generate_traj(human_traj)
                print(f"物体当前第{i}的pose, 已生成{len(traj_list)}个运动轨迹")


                for idx in range(len(traj_list)):
                    traj = traj_list[idx]
                    self.reset_env()
                    # if idx == 0:
                    success = self.play_traj(traj.copy(), quat, save_data=True)
                    if not success:
                        self.ep_num -= 1
                    else:
                        rand_pose_counter[i] += 1
                        print(f"{quat} 抓取成功")

                    print(f"物体当前第{i}的pose, 已保存{self.ep_num}个运动轨迹")
                    time.sleep(0.5)
        
        print(f"每个rand pose下的成功次数统计: {rand_pose_counter}")
        print(f"失败的episode: {fail_ep}")

    
    def load_robot(self, robot_type='franka', **kwargs):
        """加载机械臂，默认是Franka"""
        if robot_type == 'franka':
            robot = self.scene.add_entity(
                morph=gs.morphs.MJCF(
                    file="./assets/franka_emika_panda/panda.xml",
                    pos=kwargs.get('robot_pos', (-0.2, -0.25, 0.0)),
                    # pos=kwargs.get('robot_pos', (-0.2, -0.3, 0.0)),
                    euler=(0, 0, 0),
                    collision=True,
                ),
            )
            # set control gains
            end_effector = robot.get_link("hand_tcp")
            motors_dof = np.arange(7)
            fingers_dof = np.arange(7, 9)
            self.start_pos = np.array([0.0005, -0.78746, -0.00032, -2.351319, 0.00013, 1.57421, 0.789325,0.04,0.04])

        return robot, end_effector, motors_dof, fingers_dof
    
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
            self.update_scene(10, record=False)
            # self.record_data_once()
            if (i + 1) % 2 == 0:
                self.record_data_once()
        
        for i in range(5):
            self.move_to_target(traj[-1] + np.array([0.0,0,0.05]), open_gripper=True)
            self.update_scene(10, record=False)
            self.record_data_once()

        self.record_data_once(export_video=True)
        target_pos = traj[-1].copy()
        target_pos[-1] = 0

        if save_data:
            self.ep_num += 1

        self.play_counter += 1
        print(f"已保存 / 执行 {self.ep_num}/{ self.play_counter} 轨迹")

        return self.check_success(target_pos)
    

    def test_env(self):
        for i in range(700):
            self.update_scene(1, record=False)

        # for i in range(500):
        #     self.entities['cup'].set_pos(np.array([0.1, 0.1, 0.4]))
        #     self.update_scene(1, record=False)

        

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

    zero_grasp = np.eye(4)
    zero_grasp[:3, :3] = quat_to_rotation_matrix([1,0,0,0])

    # 获取手部中心轨迹
    # center_traj, valid_frames = handtraj_processor.get_center_trajectory(hand3d, is_list_format=True,smooth=5)
    # print(f"提取到{len(center_traj)}帧有效轨迹数据")

    # 加载目标位姿
    grasp_pose = np.load(pose_path)
    finger_verts = np.load(finger_verts_path)
    
    # 创建环境实例
    env_kwargs = {
        "seed": 42,
        "task_name": "grasp_cube",
        "save_path": "task_data",
        "save_data":False,
        "handtraj_processor":handtraj_processor,
        "vis": True
    }

    ########################## env ##########################
    env = PullWater(**env_kwargs)

    env.test_env()


    # print(f"生成数据耗时: {time.time() - start_time}")
    # print(f"已保存 {env.ep_num} / {env.play_counter} 个轨迹, 成功率 {env.ep_num / env.play_counter} ")

    gs.destroy()
    import gc
    gc.collect()


