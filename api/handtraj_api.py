import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle

class HandTrajProcess:
    def __init__(self):
        """初始化手部轨迹处理器"""
        # 手部关节连接关系
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),    # 拇指
            (0, 5), (5, 6), (6, 7), (7, 8),    # 食指
            (0, 9), (9, 10), (10, 11), (11, 12), # 中指
            (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
            (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
        ]
        # 确保中文正常显示
        plt.rcParams["font.family"] = ["sans-serif"]
        plt.rcParams['axes.unicode_minus'] = False

    def get_joint_trajectories(self, hand3d_data):
        """
        提取所有关节的轨迹
        
        参数:
            hand3d_data: 形状为 (N, 21, 3) 的numpy数组，包含N帧的21个3D关键点坐标
            
        返回:
            字典，包含每个关节的轨迹数据
        """
        trajectories = {}
        for joint_idx in range(21):
            trajectories[joint_idx] = hand3d_data[:, joint_idx, :]
        return trajectories

    def get_center_trajectory(self, hand3d_data, is_list_format=False):
        """
        提取手部中心轨迹（所有关键点的平均位置）
        
        参数:
            hand3d_data: 若is_list_format=True，则为嵌套列表；否则为形状为 (N, 21, 3) 的numpy数组
            is_list_format: 是否为列表格式数据
            
        返回:
            包含手部中心轨迹的numpy数组 (M, 3)，M为有效帧数
            有效帧的索引列表
        """
        if is_list_format:
            return self._get_center_from_list(hand3d_data)
        else:
            return self._get_center_from_array(hand3d_data)

    def _get_center_from_array(self, hand3d_data):
        """从numpy数组格式数据中提取中心轨迹"""
        hand_centers = np.mean(hand3d_data, axis=1)  # 形状: (N, 3)
        valid_frames = list(range(len(hand3d_data)))
        return hand_centers, valid_frames

    def _get_center_from_list(self, hand3d_data):
        """从列表格式数据中提取中心轨迹"""
        hand_centers = []
        valid_frames = []
        
        for i, frame in enumerate(hand3d_data):
            if frame is None:
                continue
                
            try:
                frame_points = np.array(frame)
            except:
                print(f"警告: 第{i}帧无法转换为有效的3D坐标数组，已跳过")
                continue
                
            if frame_points.ndim != 2 or frame_points.shape[1] != 3:
                print(f"警告: 第{i}帧的坐标格式不正确 (shape={frame_points.shape})，已跳过")
                continue
                
            frame_center = np.mean(frame_points, axis=0)
            hand_centers.append(frame_center)
            valid_frames.append(i)
        
        if not hand_centers:
            raise ValueError("没有找到有效的3D关键点数据")
            
        return np.array(hand_centers), valid_frames

    def visualize_joint_trajectories(self, hand3d_data, output_path=None, title="手部所有关键点3D轨迹", show_plot=False):
        """
        可视化所有关节的轨迹
        
        参数:
            hand3d_data: 形状为 (N, 21, 3) 的numpy数组
            output_path: 保存图像的路径，为None则不保存
            title: 图像标题
            show_plot: 是否显示图像
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x_min, y_min, z_min = np.min(hand3d_data, axis=(0, 1))
        x_max, y_max, z_max = np.max(hand3d_data, axis=(0, 1))
        margin = 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min)
        
        colors = plt.cm.rainbow(np.linspace(0, 1, 21))
        
        for joint_idx in range(21):
            joint_trajectory = hand3d_data[:, joint_idx, :]
            ax.plot(joint_trajectory[:, 0], joint_trajectory[:, 1], joint_trajectory[:, 2], 
                    color=colors[joint_idx], alpha=0.6, linewidth=1.5, 
                    label=f'Joint {joint_idx}' if joint_idx < 3 else "")
            
            ax.scatter(joint_trajectory[-1, 0], joint_trajectory[-1, 1], joint_trajectory[-1, 2],
                       color=colors[joint_idx], s=50, alpha=1.0)
        
        # 绘制最后一帧的骨架连接
        for connection in self.hand_connections:
            start_idx, end_idx = connection
            ax.plot([hand3d_data[-1, start_idx, 0], hand3d_data[-1, end_idx, 0]],
                    [hand3d_data[-1, start_idx, 1], hand3d_data[-1, end_idx, 1]],
                    [hand3d_data[-1, start_idx, 2], hand3d_data[-1, end_idx, 2]],
                    color='black', linestyle='-', linewidth=2.0, alpha=0.8)
        
        ax.set_xlabel('X (mm)', fontsize=12)
        ax.set_ylabel('Y (mm)', fontsize=12)
        ax.set_zlabel('Z (mm)', fontsize=12)
        ax.set_title(title, fontsize=16)
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)
        
        ax.view_init(elev=30, azim=45)
        plt.legend(loc='upper right')
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"轨迹图像已保存至: {output_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()

    def visualize_center_trajectory(self, hand3d_data, output_path=None, title="手部中心3D轨迹可视化", 
                                   show_plot=False, is_list_format=False):
        """
        可视化手部中心轨迹
        
        参数:
            hand3d_data: 输入的3D关键点数据
            output_path: 保存图像的路径，为None则不保存
            title: 图像标题
            show_plot: 是否显示图像
            is_list_format: 是否为列表格式数据
        """
        hand_centers, valid_frames = self.get_center_trajectory(hand3d_data, is_list_format)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x_min, y_min, z_min = np.min(hand_centers, axis=0)
        x_max, y_max, z_max = np.max(hand_centers, axis=0)
        margin = 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min)
        
        ax.plot(hand_centers[:, 0], hand_centers[:, 1], hand_centers[:, 2], 
                color='blue', alpha=0.8, linewidth=2.0)
        
        ax.scatter(hand_centers[0, 0], hand_centers[0, 1], hand_centers[0, 2],
                   color='green', s=100, alpha=1.0, label=f'Start (Frame {valid_frames[0]})')
        ax.scatter(hand_centers[-1, 0], hand_centers[-1, 1], hand_centers[-1, 2],
                   color='red', s=100, alpha=1.0, label=f'End (Frame {valid_frames[-1]})')
        
        # 绘制最后一帧的所有关键点
        if is_list_format:
            last_valid_frame_idx = valid_frames[-1]
            last_frame = hand3d_data[last_valid_frame_idx]
            if last_frame is not None:
                try:
                    last_frame_points = np.array(last_frame)
                    ax.scatter(last_frame_points[:, 0], last_frame_points[:, 1], last_frame_points[:, 2],
                              color='gray', s=30, alpha=0.7, label=f'Frame {last_valid_frame_idx} Keypoints')
                except:
                    print(f"警告: 无法绘制第{last_valid_frame_idx}帧的关键点")
        else:
            ax.scatter(hand3d_data[-1, :, 0], hand3d_data[-1, :, 1], hand3d_data[-1, :, 2],
                       color='gray', s=30, alpha=0.7, label='Final Frame')
        
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(f"{title} ({len(valid_frames)} frames)", fontsize=16)
        ax.set_xlim(x_min - margin, x_max + margin)
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_zlim(z_min - margin, z_max + margin)
        
        ax.view_init(elev=10, azim=-73)
        plt.legend(loc='upper right')
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"手部中心轨迹图像已保存至: {output_path}")
        
        if show_plot:
            plt.show()
        
        plt.close()


# 示例用法
if __name__ == "__main__":
    # 初始化处理器
    traj_processor = HandTrajProcess()
    hand_path = '/home/haichao/workspace/real2sim/simulation/data/20250716-2018/video/demo_stero/stero_hand_3d.pkl'

    # 加载数据（根据实际情况修改路径）
    hand3d = pickle.load(open('/home/haichao/workspace/real2sim/simulation/data/20250716-2018/video/demo_stero/stero_hand_3d.pkl', 'rb'))
    hand3d = hand3d[60:165]

    # 获取手部中心轨迹
    center_traj, valid_frames = traj_processor.get_center_trajectory(hand3d, is_list_format=True)
    print(f"提取到{len(center_traj)}帧有效轨迹数据")

    base_dir = os.path.dirname(hand_path)
    np.save(os.path.join(base_dir, 'hand_center_traj_processed.npy'), center_traj)

    
    # 可视化手部中心轨迹（可选）
    traj_processor.visualize_center_trajectory(
        hand3d_data=hand3d,
        output_path="results/demo_vggt.png",
        title="Hand (center of 21 keypoints) 3D Traj",
        show_plot=True,
        is_list_format=True                                                                                                                                                                                                                                
    )
    
    # 如果有数组格式的数据，也可以处理
    # array_data = np.load('hand_3d.npy')
    # joint_traj = traj_processor.get_joint_trajectories(array_data)
    # traj_processor.visualize_joint_trajectories(array_data, show_plot=True)