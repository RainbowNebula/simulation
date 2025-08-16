import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import pickle

def visualize_all_joint_trajectories(hand3d_data, output_path="hand_trajectory.png", 
                                     title="手部3D关键点轨迹可视化", show_plot=False):
    """
    可视化手部所有21个关键点的3D轨迹
    
    参数:
        hand3d_data: 形状为 (N, 21, 3) 的numpy数组，包含N帧的21个3D关键点坐标
        output_path: 保存图像的路径
        title: 图像标题
        show_plot: 是否显示图像
    """
    # 确保中文正常显示
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    # 手部关节连接关系
    hand_connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),    # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),    # 食指
        (0, 9), (9, 10), (10, 11), (11, 12), # 中指
        (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20)  # 小指
    ]
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴范围
    x_min, y_min, z_min = np.min(hand3d_data, axis=(0, 1))
    x_max, y_max, z_max = np.max(hand3d_data, axis=(0, 1))
    margin = 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min)
    
    # 为每个关键点绘制轨迹
    colors = plt.cm.rainbow(np.linspace(0, 1, 21))
    
    for joint_idx in range(21):
        # 获取该关节在所有帧中的坐标
        joint_trajectory = hand3d_data[:, joint_idx, :]
        ax.plot(joint_trajectory[:, 0], joint_trajectory[:, 1], joint_trajectory[:, 2], 
                color=colors[joint_idx], alpha=0.6, linewidth=1.5, 
                label=f'Joint {joint_idx}' if joint_idx < 3 else "")  # 只标记前几个关节避免图例过多
        
        # 在轨迹末尾添加当前位置标记
        ax.scatter(joint_trajectory[-1, 0], joint_trajectory[-1, 1], joint_trajectory[-1, 2],
                   color=colors[joint_idx], s=50, alpha=1.0)
    
    # 绘制最后一帧的骨架连接
    for connection in hand_connections:
        start_idx, end_idx = connection
        ax.plot([hand3d_data[-1, start_idx, 0], hand3d_data[-1, end_idx, 0]],
                [hand3d_data[-1, start_idx, 1], hand3d_data[-1, end_idx, 1]],
                [hand3d_data[-1, start_idx, 2], hand3d_data[-1, end_idx, 2]],
                color='black', linestyle='-', linewidth=2.0, alpha=0.8)
    
    # 设置坐标轴和标题
    ax.set_xlabel('X (mm)', fontsize=12)
    ax.set_ylabel('Y (mm)', fontsize=12)
    ax.set_zlabel('Z (mm)', fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)
    
    # 设置视角和图例
    ax.view_init(elev=30, azim=45)
    plt.legend(loc='upper right')
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"轨迹图像已保存至: {output_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def visualize_hand_center_trajectory_list(hand3d_data, output_path="hand_center_trajectory.png", 
                                    title="手部中心3D轨迹可视化", show_plot=False):
    """
    可视化手部中心的3D轨迹（以所有关键点的中心作为手部中心）
    
    参数:
        hand3d_data: 形状为 (N, M_i, 3) 的嵌套列表，包含N帧的M_i个3D关键点坐标
        output_path: 保存图像的路径
        title: 图像标题
        show_plot: 是否显示图像
    """
    # 确保中文正常显示
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    # 转换嵌套列表为NumPy数组并计算每帧的手部中心
    hand_centers = []
    valid_frames = []
    
    for i, frame in enumerate(hand3d_data):
        # 跳过空帧
        if frame is None:
            continue
            
        # 将当前帧的关键点列表转换为numpy数组
        try:
            frame_points = np.array(frame)
        except:
            print(f"警告: 第{i}帧无法转换为有效的3D坐标数组，已跳过")
            continue
            
        # 验证数组形状是否为 (M, 3)
        if frame_points.ndim != 2 or frame_points.shape[1] != 3:
            print(f"警告: 第{i}帧的坐标格式不正确 (shape={frame_points.shape})，已跳过")
            continue
            
        # 计算并存储当前帧的手部中心
        frame_center = np.mean(frame_points, axis=0)
        hand_centers.append(frame_center)
        valid_frames.append(i)
    
    # 检查是否有有效帧
    if not hand_centers:
        print("错误: 没有找到有效的3D关键点数据")
        return
    
    hand_centers = np.array(hand_centers)
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴范围
    x_min, y_min, z_min = np.min(hand_centers, axis=0)
    x_max, y_max, z_max = np.max(hand_centers, axis=0)
    margin = 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min)
    
    # 绘制手部中心轨迹
    ax.plot(hand_centers[:, 0], hand_centers[:, 1], hand_centers[:, 2], 
            color='blue', alpha=0.8, linewidth=2.0)
    
    # 标记轨迹的起点和终点
    ax.scatter(hand_centers[0, 0], hand_centers[0, 1], hand_centers[0, 2],
               color='green', s=100, alpha=1.0, label=f'Start (Frame {valid_frames[0]})')
    ax.scatter(hand_centers[-1, 0], hand_centers[-1, 1], hand_centers[-1, 2],
               color='red', s=100, alpha=1.0, label=f'End (Frame {valid_frames[-1]})')
    
    # 绘制最后一帧的所有关键点（如果存在）
    last_valid_frame_idx = valid_frames[-1]
    last_frame = hand3d_data[last_valid_frame_idx]
    
    if last_frame is not None:
        try:
            last_frame_points = np.array(last_frame)
            ax.scatter(last_frame_points[:, 0], last_frame_points[:, 1], last_frame_points[:, 2],
                       color='gray', s=30, alpha=0.7, label=f'Frame {last_valid_frame_idx} Keypoints')
        except:
            print(f"警告: 无法绘制第{last_valid_frame_idx}帧的关键点")
    
    # 设置坐标轴和标题
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(f"{title} ({len(valid_frames)} frames)", fontsize=16)
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)
    
    # 设置视角和图例
    ax.view_init(elev=10, azim=-73)
    plt.legend(loc='upper right')
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"手部中心轨迹图像已保存至: {output_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()


def visualize_hand_center_trajectory(hand3d_data, output_path="hand_center_trajectory.png", 
                                    title="手部中心3D轨迹可视化", show_plot=False):
    """
    可视化手部中心的3D轨迹（以21个关键点的中心作为手部中心）
    
    参数:
        hand3d_data: 形状为 (N, 21, 3) 的numpy数组，包含N帧的21个3D关键点坐标
        output_path: 保存图像的路径
        title: 图像标题
        show_plot: 是否显示图像
    """
    # 确保中文正常显示
    plt.rcParams["font.family"] = ["sans-serif"]
    plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
    
    # 计算每帧的手部中心（21个关键点的平均位置）
    hand_centers = np.mean(hand3d_data, axis=1)  # 形状: (N, 3)
    
    # 创建3D图形
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 设置坐标轴范围
    x_min, y_min, z_min = np.min(hand_centers, axis=0)
    x_max, y_max, z_max = np.max(hand_centers, axis=0)
    margin = 0.1 * max(x_max - x_min, y_max - y_min, z_max - z_min)
    
    # 绘制手部中心轨迹
    ax.plot(hand_centers[:, 0], hand_centers[:, 1], hand_centers[:, 2], 
            color='blue', alpha=0.8, linewidth=2.0)
    
    # 标记轨迹的起点和终点
    ax.scatter(hand_centers[0, 0], hand_centers[0, 1], hand_centers[0, 2],
               color='green', s=100, alpha=1.0, label='Start')
    ax.scatter(hand_centers[-1, 0], hand_centers[-1, 1], hand_centers[-1, 2],
               color='red', s=100, alpha=1.0, label='End')
    
    # 绘制手部中心在最后一帧的位置及所有关键点
    ax.scatter(hand3d_data[-1, :, 0], hand3d_data[-1, :, 1], hand3d_data[-1, :, 2],
               color='gray', s=30, alpha=0.7, label='Final Frame')
    
    # 设置坐标轴和标题
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title(title, fontsize=16)

    
    ax.set_xlim(x_min - margin, x_max + margin)
    ax.set_ylim(y_min - margin, y_max + margin)
    ax.set_zlim(z_min - margin, z_max + margin)

    # 反转Y轴和Z轴，使Y轴从下到上递增，Z轴从屏幕内指向屏幕外
    ax.invert_yaxis()
    ax.invert_xaxis()
    
    # 设置视角和图例
    ax.view_init(elev=10, azim=-73)
    # 关键：让X轴面向自己（azim=180°）
    # ax.view_init(elev=30, azim=180)
    plt.legend(loc='upper right')
    
    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"手部中心轨迹图像已保存至: {output_path}")
    
    if show_plot:
        plt.show()
    
    plt.close()


# 示例用法
if __name__ == "__main__":
    # 加载手部3D关键点数据
    # hand3d = np.load('/mnt/nas/liuqipeng/data/20250710-2126/video/demo_vggt/hand_3d.npy')
    # print(hand3d.shape) 

    hand3d = pickle.load(open('/home/haichao/workspace/real2sim/simulation/data/20250716-2018/video/demo_stero/stero_hand_3d.pkl', 'rb'))
    # hand3d = [i for i in hand3d if len(i) > 0]
    # hand3d = hand3d[60:165]
    
    # 可视化所有关键点轨迹
    # visualize_all_joint_trajectories(
    #     hand3d_data=hand3d,
    #     output_path="results/hand_trajectory.png",
    #     title="手部所有关键点3D轨迹",
    #     show_plot=True
    # )
    # print(len(hand3d))
    
    # 可视化手部中心轨迹
    visualize_hand_center_trajectory_list(
        hand3d_data=hand3d,
        output_path="results/demo_vggt.png",
        title="Hand (center of 21 keypoints) 3D Traj",
        show_plot=True
    )