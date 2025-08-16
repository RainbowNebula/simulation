import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle
from movement_primitives.dmp import DMP
from movement_primitives.promp import ProMP,via_points
from scipy.linalg import eigh
from scipy.signal import savgol_filter

def symmetric_moving_average(signal, window):
    """实现对称移动平均，更好地处理边界"""
    window_size = len(window)
    half_window = window_size // 2
    smoothed = np.convolve(signal, window, mode='same')
    
    # 修正开头部分
    for i in range(half_window):
        k = i + half_window + 1
        smoothed[i] = np.mean(signal[:k])
        
    # 修正结尾部分
    n = len(signal)
    for i in range(n-half_window, n):
        k = n - i + half_window
        smoothed[i] = np.mean(signal[-k:])
        
    return smoothed

#  Savitzky-Golay 滤波
def smooth_3d_trajectory(trajectory, window_size, method='savgol'):
    """
    对3D轨迹进行平滑处理
    :param trajectory: 原始轨迹数组，形状: (N, 3)
    :param window_size: 平滑窗口大小，必须为正奇数
    :param method: 平滑方法，可选 'savgol', 'gaussian', 'moving_average'
    :return: 平滑后的轨迹数组
    """
    if window_size < 2:
        return trajectory
        
    # 确保窗口大小不超过轨迹长度且为奇数（对Savitzky-Golay很重要）
    n_frames = len(trajectory)
    window_size = min(window_size, n_frames)
    if window_size % 2 == 0:
        window_size -= 1  # 确保是奇数
    if window_size < 3:
        return trajectory
        
    smoothed = np.zeros_like(trajectory)
    
    if method == 'savgol':
        # Savitzky-Golay滤波，适合保留趋势
        # 使用3阶多项式拟合
        for i in range(3):
            # 正确写法
            smoothed[:, i] = savgol_filter(
                trajectory[:, i], 
                window_length=window_size,  # 改为 window_length
                polyorder=3,
                mode='nearest'
            )
            
    elif method == 'gaussian':
        # 高斯滤波，使用标准差为窗口大小1/6的高斯核
        from scipy.ndimage import gaussian_filter1d
        sigma = window_size / 6.0  # 经验值，6σ覆盖99.7%的分布
        for i in range(3):
            smoothed[:, i] = gaussian_filter1d(trajectory[:, i], sigma=sigma, mode='nearest')
            
    elif method == 'moving_average':
        # 改进的移动平均，使用对称窗口处理边界
        window = np.ones(window_size) / window_size
        for i in range(3):
            # 对开头和结尾使用较小的窗口避免边界效应
            smoothed[:, i] = symmetric_moving_average(trajectory[:, i], window)
            
    return smoothed



class HandTrajProcess:
    def __init__(self, **kwargs):
        """初始化手部轨迹处理器，支持DMP和ProMP的参数配置
        
        参数:
           ** kwargs: 可包含DMP和ProMP的初始化参数，如:
                dmp_execution_time: DMP执行时间
                dmp_dt: DMP时间步长
                dmp_n_weights: DMP每个维度的权重数量
                promp_n_weights: ProMP每个维度的权重数量
        """
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
        
        # DMP和ProMP参数配置
        self.dmp_params = {
            "execution_time": 2.0,
            "dt": 0.01,
            "n_weights_per_dim": 100
        }
        self.promp_params = {
            "n_weights_per_dim": 10
        }
        
        # 更新参数
        if "dmp_execution_time" in kwargs:
            self.dmp_params["execution_time"] = kwargs["dmp_execution_time"]
        if "dmp_dt" in kwargs:
            self.dmp_params["dt"] = kwargs["dmp_dt"]
        if "dmp_n_weights" in kwargs:
            self.dmp_params["n_weights_per_dim"] = kwargs["dmp_n_weights"]
        if "promp_n_weights" in kwargs:
            self.promp_params["n_weights_per_dim"] = kwargs["promp_n_weights"]
        
        # 存储训练好的模型和原始轨迹信息
        self.dmp_model = None
        self.promp_model = None
        self._original_start = None  # 原始轨迹起点
        self._original_goal = None   # 原始轨迹终点

    def get_joint_trajectories(self, hand3d_data):
        """提取所有关节的轨迹"""
        trajectories = {}
        for joint_idx in range(21):
            trajectories[joint_idx] = hand3d_data[:, joint_idx, :]
        return trajectories

    # def get_center_trajectory(self, hand3d_data, is_list_format=False):
    #     """提取手部中心轨迹（所有关键点的平均位置）"""
    #     if is_list_format:
    #         return self._get_center_from_list(hand3d_data)
    #     else:
    #         return self._get_center_from_array(hand3d_data)
        
    def get_center_trajectory(self, hand3d_data, is_list_format=False, smooth=0):
        """提取手部中心轨迹（所有关键点的平均位置）"""
        if is_list_format:
            hand_centers, valid_frames = self._get_center_from_list(hand3d_data)
        else:
            hand_centers, valid_frames = self._get_center_from_array(hand3d_data)
        
        # 如果需要平滑且平滑窗口大小有效
        if smooth > 1:
            hand_centers = smooth_3d_trajectory(hand_centers, smooth)
        
        return hand_centers, valid_frames

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
    
    

    def apply_dmp(self, trajectory, start_pos=None, goal_pos=None, return_time=False, retrain=False):
        """
        应用DMP学习并生成轨迹，支持自定义起点和终点
        
        参数:
            trajectory: 输入轨迹，形状为 (N, 3) 的numpy数组
            start_pos: 可选，自定义起点坐标 (3,)，None则使用原始轨迹起点
            goal_pos: 可选，自定义终点坐标 (3,)，None则使用原始轨迹终点
            return_time: 是否返回时间轴
            retrain: 是否强制重新训练模型，即使已有训练好的模型
            
        返回:
            生成的轨迹 (M, 3)
            若return_time=True，同时返回时间轴 (M,)
        """
        # 记录原始轨迹的起点和终点
        self._original_start = trajectory[0]
        self._original_goal = trajectory[-1]
        
        # 确定实际使用的起点和终点
        actual_start = start_pos if start_pos is not None else self._original_start
        actual_goal = goal_pos if goal_pos is not None else self._original_goal
        
        # 训练模型（如果需要）
        if self.dmp_model is None or retrain:
            self.dmp_model = DMP(
                n_dims=3,
                execution_time=self.dmp_params["execution_time"],
                dt=self.dmp_params["dt"],
                n_weights_per_dim=self.dmp_params["n_weights_per_dim"]
            )
            # 生成时间轴
            T = np.linspace(0, self.dmp_params["execution_time"], len(trajectory))
            self.dmp_model.imitate(T, trajectory)
        
        # 如果需要修改起点或终点，创建模型副本并重新配置
        if start_pos is not None or goal_pos is not None:
            current_dmp = self.dmp_model
            current_dmp.configure(start_y=actual_start, goal_y=actual_goal)
        else:
            current_dmp = self.dmp_model
        
        # 生成轨迹
        T_gen, Y_gen = current_dmp.open_loop()
        
        if return_time:
            return Y_gen, T_gen
        return Y_gen

    def apply_promp(self, trajectory, start_pos=None, via_points_data=None, 
                    return_stats=False, n_steps=101, retrain=False):
        """
        应用ProMP学习并生成轨迹，支持自定义起点和通过点约束
        """
        # 记录原始轨迹的起点
        self._original_start = trajectory[0]
        
        # 训练模型（如果需要）
        if self.promp_model is None or retrain:
            self.promp_model = ProMP(
                n_dims=3,
                n_weights_per_dim=self.promp_params["n_weights_per_dim"]
            )
            # 生成时间轴
            T = np.linspace(0, 1, len(trajectory))
            self.promp_model.imitate(T[None, ...], trajectory[None, ...])
        
        # 生成时间轴
        T_gen = np.linspace(0, 1, n_steps)
        current_promp = self.promp_model
        
        # 应用通过点约束（如果提供）
        if via_points_data is not None:
            ts = np.array(via_points_data['ts'])
            y_cond = np.array(via_points_data['y_cond'])
            y_conditional_cov = np.array(via_points_data.get(
                'y_conditional_cov', [0.01]*len(ts)
            ))
            
            if len(ts) != len(y_cond):
                raise ValueError("ts和y_cond的长度必须相同")
            if y_cond.ndim != 2 or y_cond.shape[1] != 3:
                raise ValueError("y_cond必须是形状为 (n_points, 3) 的数组")
            
            # 为整个3D轨迹创建单一条件化模型（而非按维度拆分）
            cpromp = via_points(
                promp=current_promp,
                y_cond=y_cond,  # 展平为一维数组
                y_conditional_cov=y_conditional_cov, #np.repeat(y_conditional_cov, 3),  # 每个维度重复cov
                ts=ts,
                # ts=np.repeat(ts, 3),  # 每个维度重复时间点
            )
            
            Y_mean = cpromp.mean_trajectory(T_gen).reshape(-1, 3)  # 重塑为(n_steps, 3)
            Y_var = None
            
            if return_stats:
                Y_var = cpromp.var_trajectory(T_gen).reshape(-1, 3)
                Y_conf = 1.96 * np.sqrt(Y_var)
                return Y_mean, Y_var, Y_conf, T_gen
            return Y_mean, T_gen
        
        # 应用起点约束（如果提供且没有通过点约束）
        elif start_pos is not None:
            y_cov = np.eye(3) * 0.01
            cpromp = current_promp.condition_position(
                start_pos, y_cov=y_cov, t=0.0, t_max=1.0
            )
            Y_mean = cpromp.mean_trajectory(T_gen).reshape(-1, 3)
            if return_stats:
                Y_var = cpromp.var_trajectory(T_gen).reshape(-1, 3)
                Y_conf = 1.96 * np.sqrt(Y_var)
                return Y_mean, Y_var, Y_conf, T_gen
            return Y_mean, T_gen
        
        # 无约束生成
        else:
            Y_mean = current_promp.mean_trajectory(T_gen).reshape(-1, 3)
            if return_stats:
                Y_var = current_promp.var_trajectory(T_gen).reshape(-1, 3)
                Y_conf = 1.96 * np.sqrt(Y_var)
                return Y_mean, Y_var, Y_conf, T_gen
            return Y_mean, T_gen

    def apply_promp_old(self, trajectory, start_pos=None, end_pos=None, via_points_data=None, return_stats=False, n_steps=101, random_samples=0, retrain=False):
        """
        应用ProMP学习并生成轨迹，支持自定义起点
        
        参数:
            trajectory: 输入轨迹，形状为 (N, 3) 的numpy数组
            start_pos: 可选，自定义起点坐标 (3,)，None则使用原始轨迹起点
            return_stats: 是否返回方差
            n_steps: 生成轨迹的步数
            retrain: 是否强制重新训练模型，即使已有训练好的模型
            
        返回:
            生成的平均轨迹 (n_steps, 3)
            若return_stats=True，同时返回方差 (n_steps, 3) 和时间轴 (n_steps,)
        """
        # 记录原始轨迹的起点
        self._original_start = trajectory[0]
        
        # 训练模型（如果需要）
        if self.promp_model is None or retrain:
            self.promp_model = ProMP(
                n_dims=3,
                n_weights_per_dim=self.promp_params["n_weights_per_dim"]
            )
            # 生成时间轴
            T = np.linspace(0, 1, len(trajectory))
            self.promp_model.imitate(T[None, ...], trajectory[None, ...])
        
        cpromp = self.promp_model
        # 生成时间轴
        T_gen = np.linspace(0, 1, n_steps)
        
        # 应用起点约束（如果提供）
        if start_pos is not None:
            # 添加起点约束
            # y_cov = np.eye(3) * 0.025  # 位置方差（控制约束强度）
            y_cov = np.array([0.05] * 3)
            cpromp = cpromp.condition_position(
                start_pos, y_cov=y_cov, t=0.0, t_max=1.0
            )
            # Y_mean = cpromp.mean_trajectory(T_gen)

        if end_pos is not None:
            # 添加终点约束
            y_cov = np.array([0.05] * 3)
            # y_cov = np.eye(3) * 0.025  # 位置方差（控制约束强度）
            cpromp = cpromp.condition_position(
                end_pos, y_cov=y_cov, t=1.0, t_max=1.0
            )

        # 应用通过点约束（如果提供）
        if via_points_data is not None:
            ts = np.array(via_points_data['ts'])
            y_cond = np.array(via_points_data['y_cond'])
            y_conditional_cov = np.array(via_points_data.get(
                'y_conditional_cov', [0.1]*len(ts)
            ))
            
            if len(ts) != len(y_cond):
                raise ValueError("ts和y_cond的长度必须相同")
            if y_cond.ndim != 2 or y_cond.shape[1] != 3:
                raise ValueError("y_cond必须是形状为 (n_points, 3) 的数组")
            
            # 为整个3D轨迹创建单一条件化模型（而非按维度拆分）
            cpromp = via_points(
                promp=cpromp,
                y_cond=y_cond,  # 展平为一维数组
                y_conditional_cov=y_conditional_cov, #np.repeat(y_conditional_cov, 3),  # 每个维度重复cov
                ts=ts,
                # ts=np.repeat(ts, 3),  # 每个维度重复时间点
            )
            # Y_mean = cpromp.mean_trajectory(T_gen).reshape(-1, 3)  # 重塑为(n_steps, 3)

        Y_mean = cpromp.mean_trajectory(T_gen)
        # 处理采样轨迹
        if random_samples > 0:
            # 初始化随机数生成器
            random_state = np.random.RandomState()
            # 从条件化后的ProMP模型中采样轨迹
            sampled_trajectories = cpromp.sample_trajectories(
                T_gen, random_samples, random_state
            )
            Y_mean = np.concatenate([Y_mean[np.newaxis, ...], sampled_trajectories], axis=0)

        
        if return_stats:
            Y_var = cpromp.var_trajectory(T_gen)
            return Y_mean, Y_var, T_gen
        return Y_mean, T_gen

    def visualize_primitives(self, original_traj, dmp_traj=None, promp_traj=None, 
                            output_path=None, show_plot=True):
        """可视化原始轨迹、DMP和ProMP生成的轨迹"""
        fig = plt.figure(figsize=(15, 5))
        
        # 原始轨迹
        ax1 = fig.add_subplot(131, projection='3d')
        ax1.plot(original_traj[:, 0], original_traj[:, 1], original_traj[:, 2], 
                color='black', alpha=0.8, label='Original')
        ax1.scatter(original_traj[0, 0], original_traj[0, 1], original_traj[0, 2],
                   color='green', s=80, label='Start')
        ax1.scatter(original_traj[-1, 0], original_traj[-1, 1], original_traj[-1, 2],
                   color='red', s=80, label='End')
        ax1.set_title('Original Trajectory')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.legend()
        
        # DMP轨迹
        ax2 = fig.add_subplot(132, projection='3d')
        ax2.plot(original_traj[:, 0], original_traj[:, 1], original_traj[:, 2], 
                color='gray', alpha=0.3, label='Original')
        if dmp_traj is not None:
            ax2.plot(dmp_traj[:, 0], dmp_traj[:, 1], dmp_traj[:, 2], 
                    color='blue', alpha=0.8, label='DMP')
            ax2.scatter(dmp_traj[0, 0], dmp_traj[0, 1], dmp_traj[0, 2],
                       color='green', s=80)
            ax2.scatter(dmp_traj[-1, 0], dmp_traj[-1, 1], dmp_traj[-1, 2],
                       color='red', s=80)
        ax2.set_title('DMP Generated')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.legend()
        
        # ProMP轨迹
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot(original_traj[:, 0], original_traj[:, 1], original_traj[:, 2], 
                color='gray', alpha=0.3, label='Original')
        if promp_traj is not None:
            ax3.plot(promp_traj[:, 0], promp_traj[:, 1], promp_traj[:, 2], 
                    color='red', alpha=0.8, label='ProMP')
            ax3.scatter(promp_traj[0, 0], promp_traj[0, 1], promp_traj[0, 2],
                       color='green', s=80)
            ax3.scatter(promp_traj[-1, 0], promp_traj[-1, 1], promp_traj[-1, 2],
                       color='red', s=80)
        ax3.set_title('ProMP Generated')
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        ax3.legend()
        
        plt.tight_layout()
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"轨迹对比图已保存至: {output_path}")
        
        if show_plot:
            plt.show()
        plt.close()

    def visualize_multiple_starts(self, original_traj, new_starts, output_path=None, show_plot=True):
        """可视化多个新起点生成的轨迹对比"""
        if self.dmp_model is None or self.promp_model is None:
            raise ValueError("请先使用apply_dmp和apply_promp训练模型")
            
        fig = plt.figure(figsize=(12, 6))
        ax_dmp = fig.add_subplot(121, projection='3d')
        ax_promp = fig.add_subplot(122, projection='3d')
        
        # 绘制原始轨迹作为参考
        ax_dmp.plot(original_traj[:, 0], original_traj[:, 1], original_traj[:, 2],
                   color='gray', alpha=0.3, label='Original')
        ax_promp.plot(original_traj[:, 0], original_traj[:, 1], original_traj[:, 2],
                    color='gray', alpha=0.3, label='Original')
        
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        for idx, start_pos in enumerate(new_starts):
            color = colors[idx % len(colors)]
            
            # DMP新起点轨迹
            dmp_traj = self.apply_dmp(original_traj, start_pos=start_pos)
            ax_dmp.plot(dmp_traj[:, 0], dmp_traj[:, 1], dmp_traj[:, 2], 
                       color=color, label=f'Start: {start_pos.round(2)}')
            ax_dmp.scatter(*start_pos, marker='*', s=100, c=color)
            
            # ProMP新起点轨迹
            promp_traj, _ = self.apply_promp(original_traj, start_pos=start_pos)
            ax_promp.plot(promp_traj[:, 0], promp_traj[:, 1], promp_traj[:, 2], 
                         color=color, label=f'Start: {start_pos.round(2)}')
            ax_promp.scatter(*start_pos, marker='*', s=100, c=color)
        
        ax_dmp.set_title('DMP with Different Starts')
        ax_dmp.set_xlabel('X')
        ax_dmp.set_ylabel('Y')
        ax_dmp.set_zlabel('Z')
        ax_dmp.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        ax_promp.set_title('ProMP with Different Starts')
        ax_promp.set_xlabel('X')
        ax_promp.set_ylabel('Y')
        ax_promp.set_zlabel('Z')
        ax_promp.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"多起点轨迹图已保存至: {output_path}")
        
        if show_plot:
            plt.show()
        plt.close()

    # 保留原有的可视化关节轨迹和中心轨迹方法
    def visualize_joint_trajectories(self, hand3d_data, output_path=None, title="手部所有关键点3D轨迹", show_plot=False):
        """可视化所有关节的轨迹"""
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
        """可视化手部中心轨迹"""
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

    def find_trajectory_diff(self, traj1, traj2, traj1_times=None, vis=False, output_path=None, 
                            tolerance=1e-2, sample_sets=3, samples_set_points=3, duplicate_tol=1e-2):
        """
        计算轨迹差异，多组采样并将坐标与时间分离返回
        
        参数:
            traj1/traj2: 输入轨迹 (N,3)/(M,3)
            traj1_times: 轨迹1时间轴，默认0-1均匀分布
            vis: 是否可视化
            output_path: 可视化保存路径
            tolerance: 差异点判定阈值
            sample_sets: 采样组数
            samples_set_points: 每组采样点数（0返回所有差异点）
            duplicate_tol: 组内去重阈值
            
        返回:
            单组: (差异点坐标数组, 时间数组)
            多组: ([坐标数组列表], [时间数组列表])
        """
        # 生成时间轴
        traj1_times = np.linspace(0, 1, len(traj1)) if traj1_times is None else traj1_times
        if len(traj1_times) != len(traj1):
            raise ValueError("traj1_times长度与traj1不符")
        
        # 提取所有差异点
        diff_points, diff_times = [], []
        for i, p in enumerate(traj1):
            if np.min(np.linalg.norm(traj2 - p, axis=1)) > tolerance:
                diff_points.append(p)
                diff_times.append(traj1_times[i])
        diff_points, diff_times = np.array(diff_points), np.array(diff_times)
        total = len(diff_points)
        
        # 处理无差异点情况
        empty = (np.array([]), np.array([]))
        if total == 0:
            if sample_sets == 1:
                return empty
            return ([empty[0]]*sample_sets, [empty[1]]*sample_sets)
        
        # 多组采样（分离坐标和时间列表）
        all_points, all_times = [], []
        for _ in range(sample_sets):
            if samples_set_points <= 0:
                all_points.append(diff_points.copy())
                all_times.append(diff_times.copy())
                continue
            
            # 组内采样去重
            sampled_p, sampled_t = [], []
            remaining = list(range(total))
            while len(sampled_p) < samples_set_points and remaining:
                idx = np.random.choice(remaining)
                p, t = diff_points[idx], diff_times[idx]
                # 检查重复
                if not any(np.linalg.norm(p - sp) < duplicate_tol for sp in sampled_p):
                    sampled_p.append(p)
                    sampled_t.append(t)
                remaining.remove(idx)
            
            all_points.append(np.array(sampled_p))
            all_times.append(np.array(sampled_t))
        
        # 可视化
        if vis and len(all_points[0]) > 0:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.plot(traj1[:,0], traj1[:,1], traj1[:,2], 'b-', alpha=0.5, label='traj1')
            ax.plot(traj2[:,0], traj2[:,1], traj2[:,2], 'g-', alpha=0.5, label='traj2')
            ax.scatter(all_points[0][:,0], all_points[0][:,1], all_points[0][:,2], 
                    'r*', s=100, label='diff points')
            ax.set_xlabel('X'), ax.set_ylabel('Y'), ax.set_zlabel('Z')
            ax.set_title(f'sets={sample_sets}, points/set={samples_set_points}')
            ax.legend()
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=300)
            plt.show(), plt.close()
        
        # 单组返回元组，多组返回两个列表 (all_points[0], all_times[0]) if sample_sets == 1 else (all_points, all_times)
        return all_points, all_times


    def find_trajectory_diff_old(self, traj1, traj2, traj1_times=None, vis=False, output_path=None, tolerance=1e-2, random_sample=0):
        """
        计算轨迹1与轨迹2的差异，提取轨迹1中不在轨迹2中的点及其对应时间点
        
        参数:
            traj1: 轨迹1，形状为 (N, 3) 的numpy数组
            traj2: 轨迹2，形状为 (M, 3) 的numpy数组
            traj1_times: 轨迹1各点对应的时间点，形状为 (N,)，若为None则生成默认时间轴
            vis: 是否可视化差异
            output_path: 可视化结果保存路径（vis=True时有效）
            tolerance: 点匹配的容差阈值
            random_sample: 随机采样数量，0返回所有差异点，1-N返回随机采样的N个点
            
        返回:
            差异点坐标数组 (K, 3) 和对应的时间点数组 (K,)
        """
        # 生成轨迹1的默认时间轴（0到1之间均匀分布）
        if traj1_times is None:
            traj1_times = np.linspace(0, 1, len(traj1))
        
        # 确保时间轴长度与轨迹匹配
        if len(traj1_times) != len(traj1):
            raise ValueError("traj1_times长度必须与traj1一致")
        
        # 提取轨迹1中不在轨迹2中的点及其时间
        diff_points = []
        diff_times = []
        for i, point in enumerate(traj1):
            # 计算与轨迹2所有点的欧氏距离
            distances = np.linalg.norm(traj2 - point, axis=1)
            if np.min(distances) > tolerance:
                diff_points.append(point)
                diff_times.append(traj1_times[i])
        
        # 转换为numpy数组
        diff_points = np.array(diff_points)
        diff_times = np.array(diff_times)
        
        # 随机采样处理
        if random_sample > 0 and len(diff_points) > 0:
            sample_size = min(random_sample, len(diff_points))
            indices = np.random.choice(len(diff_points), sample_size, replace=False)
            diff_points = diff_points[indices]
            diff_times = diff_times[indices]
        
        # 可视化差异
        if vis and len(diff_points) > 0:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 绘制原始轨迹
            ax.plot(traj1[:, 0], traj1[:, 1], traj1[:, 2], 'b-', alpha=0.5, label='traj1')
            ax.plot(traj2[:, 0], traj2[:, 1], traj2[:, 2], 'g-', alpha=0.5, label='traj2')
            
            # 标记差异点
            ax.scatter(diff_points[:, 0], diff_points[:, 1], diff_points[:, 2], 
                    'r*', s=100, label='traj1 diff points')
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('traj diff points')
            ax.legend()
            
            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                plt.savefig(output_path, dpi=300)
            plt.show()
            plt.close()
        
        return diff_points, diff_times




# 示例用法
if __name__ == "__main__":
    # 初始化处理器，可配置DMP和ProMP参数
    traj_processor = HandTrajProcess(
        dmp_execution_time=2.0,
        dmp_n_weights=100,
        promp_n_weights=10
    )
    
    # 生成示例轨迹（螺旋线）
    np.random.seed(42)
    n_frames = 100
    t = np.linspace(0, 2*np.pi, n_frames)
    x = np.sin(t) * 0.5
    y = np.cos(t) * 0.5
    z = t / (2*np.pi) * 0.3
    center_traj = np.column_stack((x, y, z))
    
    # 1. 使用原始起点生成轨迹
    dmp_original = traj_processor.apply_dmp(center_traj)
    promp_original, _ = traj_processor.apply_promp(center_traj)
    
    # 2. 使用新起点生成轨迹
    new_start = np.array([-0.6, -0.6, 0.0])
    dmp_new_start = traj_processor.apply_dmp(center_traj, start_pos=new_start)
    promp_new_start, _ = traj_processor.apply_promp(center_traj, start_pos=new_start)
    
    
    # 3. 可视化对比
    print("原始起点轨迹对比:")
    traj_processor.visualize_primitives(
        original_traj=center_traj,
        dmp_traj=dmp_original,
        promp_traj=promp_original,
        output_path="results/original_starts.png",
        show_plot=True
    )
    
    print("新起点轨迹对比:")
    traj_processor.visualize_primitives(
        original_traj=center_traj,
        dmp_traj=dmp_new_start,
        promp_traj=promp_new_start,
        output_path="results/new_start.png",
        show_plot=True
    )
    
    # 4. 多起点对比
    new_starts = np.array([
        [-0.6, -0.6, 0.0],
        [0.0, 0.0, 0.1],
        [0.5, -0.5, 0.2]
    ])
    traj_processor.visualize_multiple_starts(
        original_traj=center_traj,
        new_starts=new_starts,
        output_path="results/multiple_starts.png",
        show_plot=True
    )
    
    
    # via_points_data = {
    # 'ts': [0.2, 0.5, 0.8],  # 时间点（0到1之间）
    # 'y_cond': [             # 对应时间点的目标位置
    #     [-0.3, 0.2, 0.05],
    #     [0.0, -0.4, 0.15],
    #     [0.3, 0.1, 0.25]
    # ]
    # # 无需指定y_conditional_cov，将自动使用默认值0.01
    # }

    # # 获取包含置信区间的结果
    # promp_via, Y_var, Y_conf, T_gen = traj_processor.apply_promp(
    #     center_traj, 
    #     via_points_data=via_points_data,
    #     return_stats=True
    # )