
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

class GraspCube(BaseEnv):
    def __init__(self, **kwargs):
        super(GraspCube, self).__init__(**kwargs)
        self.save_dir = kwargs.get('save_dir', "./grasp_cube_h100")

    def add_objects(self, **kwargs):
        """添加对象"""
        self.entities['cube'] = self.scene.add_entity(
                                        morph=gs.morphs.Mesh(
                                            # file=f"{asset_path}/allegro_hand/allegro_hand_right_glb.urdf",
                                            file="./assets/wood_cube.glb",
                                            pos=(0.24623267, -0.04144618, 0.03),
                                            euler=(0.0, 0.0, 0.0),
                                            scale=0.08,
                                            # decimate=True,
                                            # convexify=False,
                                            # decimate_face_num=50,
                                        ),
                                        material=gs.materials.Rigid(
                                            rho=400,
                                            # friction=2.0,  # 增大基础摩擦系数
                                        ))
        
        self.entities['box'] = self.scene.add_entity(
                                morph=gs.morphs.Mesh(
                                    # file=f"{asset_path}/allegro_hand/allegro_hand_right_glb.urdf",
                                    file="./assets/blue_box.glb",
                                    pos=(0.21407672, -0.26041815,  0.02),
                                    euler=(90, 0.0, 90),
                                    scale=(0.22, 0.1, 0.22), 
                                    # decimate=True,
                                    convexify=True,
                                    fixed=True,
                                    # decimate_face_num=50,
                                ),
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
        rand_num = 10
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

                # 保存两个异常用例
                # if np.isnan(self.entities['cube'].get_pos().cpu().numpy()).any() and len(fail_ep) < 2:
                #     print(f"当前第 {i} 个random, cube 位置异常：",self.entities['cube'].get_pos().cpu().numpy())
                #     # self.reset_env()
                #     break

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

                    # if success:
                    #     self.reset_env()
                    #     self.play_traj(traj.copy(), np.array([0,1,0,0]), save_data=True)
                    #     print(f"grasp [0,1,0,0] 成功")

                    # self.reset_env()
                    # success = self.play_traj(traj.copy(), quat, save_data=False)
                    
                    # if not success:
                    #     self.reset_env()
                    #     self.play_traj(traj.copy(), quat, save_data=True)
                    #     print(f"grasp {euler_angles} 成功")
                    #     rand_pose_counter[i] += 1

                    print(f"物体当前第{i}的pose, 已保存{self.ep_num}个运动轨迹")
                    time.sleep(0.5)
        
        print(f"每个rand pose下的成功次数统计: {rand_pose_counter}")
        print(f"失败的episode: {fail_ep}")

    
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
        "task_name": "object_manipulation",
        "save_path": "task_data",
        "save_data":False,
        "handtraj_processor":handtraj_processor
    }

    ########################## env ##########################
    env = GraspCube(vis=args.vis, **env_kwargs)

    if not os.path.exists("env_valid_pose.npy"):
        grasp_pose = env.check_grasp(grasp_pose)
        print(f"final grasp pose: {len(grasp_pose)}")
        np.save("env_valid_pose.npy", grasp_pose)
    else:
        grasp_pose = np.load("env_valid_pose.npy")

    print(f"已获取 {len(grasp_pose)} 个抓取位姿")

    grasp_pose = np.concatenate([zero_grasp[np.newaxis, ...], grasp_pose], axis=0)
    start_time = time.time()
    env.generate_data(hand3d, grasp_pose)
    print(f"生成数据耗时: {time.time() - start_time}")
    print(f"已保存 {env.ep_num} / {env.play_counter} 个轨迹, 成功率 {env.ep_num / env.play_counter} ")

    gs.destroy()
    import gc
    gc.collect()


