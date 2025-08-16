import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import argparse
import os
import pickle
from matplotlib.gridspec import GridSpec

# 关节名称（英文）
JOINT_NAMES = [
    "Wrist",          # 手腕
    "Thumb Base",     # 拇指根
    "Index Base",     # 食指根
    "Middle Base",    # 中指根
    "Ring Base",      # 无名指根
    "Pinky Base"      # 小拇指根
]

def load_hand_3d_data(data_folder, args):
    """加载数据并计算速度、位移、窗口统计及中心偏移"""
    hand3d_path = os.path.join(data_folder, 'stero_hand_3d.pkl')
    if not os.path.exists(hand3d_path):
        print(f"File {hand3d_path} not found")
        return None, None, None, None, None, None
    
    # 加载手部3D关键点数据
    with open(hand3d_path, "rb") as f:
        hand_3d_data = pickle.load(f)
    
    # 过滤无效帧
    valid_frames = []
    for frame_data in hand_3d_data:
        if len(frame_data) >= len(JOINT_NAMES):
            valid_frames.append([frame_data[i] for i in range(len(JOINT_NAMES))])
    
    # 检查最小帧数（需要至少window_size帧）
    if len(valid_frames) < args.window_size + 1:
        print(f"Insufficient valid frames (need at least {args.window_size + 1})")
        return None, None, None, None, None, None
    
    # 转换为numpy数组 (帧数 x 关节数 x 3)
    positions = np.array(valid_frames)
    
    # 计算每帧的关键点中心（平均位置）
    joint_centers = np.mean(positions, axis=1)  # 帧数 x 3
    
    # 计算中心偏移量（相对于前一帧）
    center_offsets = []
    for i in range(1, len(joint_centers)):
        center_offsets.append(np.linalg.norm(joint_centers[i] - joint_centers[i-1]))
    center_offsets = np.array(center_offsets)
    
    # 计算每帧之间的位移和瞬时速度
    fps = args.fps
    time_interval = 1.0 / fps
    displacements = []
    instant_velocities = []
    
    for i in range(1, len(positions)):
        # 计算位移（3D向量）
        displacement = positions[i] - positions[i-1]
        displacements.append(displacement)
        
        # 计算瞬时速度大小
        speed = np.linalg.norm(displacement, axis=1) / time_interval
        instant_velocities.append(speed)
    
    displacements = np.array(displacements)
    instant_velocities = np.array(instant_velocities)
    
    # 计算速度变化（相对于前一帧）
    velocity_changes = []
    for i in range(1, len(instant_velocities)):
        velocity_changes.append(np.linalg.norm(instant_velocities[i] - instant_velocities[i-1]))
    velocity_changes = np.array(velocity_changes)
    
    # 计算基于滑动窗口的速度统计（当前帧向前4帧）
    window_velocities = []
    window_size = args.window_size  # 窗口大小（包括当前帧共5帧）
    
    for i in range(len(instant_velocities)):
        # 确定窗口范围（当前帧向前4帧）
        start = max(0, i - (window_size - 1))
        end = i + 1  # 包含当前帧
        
        # 计算窗口内的平均速度
        window = instant_velocities[start:end]
        window_mean = np.mean(window, axis=0)
        window_velocities.append(window_mean)
    
    return (positions, displacements, instant_velocities, 
            np.array(window_velocities), center_offsets, velocity_changes)

def plot_summary(displacements, instant_velocities, window_velocities, 
                center_offsets, velocity_changes, args):
    """绘制综合统计图表"""
    if instant_velocities is None or displacements is None:
        return
    
    num_joints = instant_velocities.shape[1]
    num_frames = instant_velocities.shape[0]
    window_size = args.window_size
    
    # 创建图形
    plt.figure(figsize=(14, 18))
    gs = GridSpec(5, 1, height_ratios=[3, 3, 2, 2, 2])
    
    # 1. 瞬时速度曲线
    ax1 = plt.subplot(gs[0])
    for i in range(num_joints):
        ax1.plot(instant_velocities[:, i], label=JOINT_NAMES[i], alpha=0.7)
        ax1.plot(window_velocities[:, i], label=f'{JOINT_NAMES[i]} (window avg)', linestyle='--')
    
    ax1.set_title(f'Instantaneous vs Window-Averaged Velocity')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Velocity (units/second)')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True)
    ax1.text(0.02, 0.98, f'Window: current frame + previous {window_size-1} frames', 
             transform=ax1.transAxes, verticalalignment='top', 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. 位移大小曲线
    ax2 = plt.subplot(gs[1])
    displacement_magnitudes = np.linalg.norm(displacements, axis=2)
    for i in range(num_joints):
        ax2.plot(displacement_magnitudes[:, i], label=JOINT_NAMES[i])
    
    ax2.set_title('Displacement Magnitude (Frame-to-Frame)')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Displacement (units)')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True)
    
    # 3. 关键点中心偏移量
    ax3 = plt.subplot(gs[2])
    ax3.plot(center_offsets, color='purple', label='Joint Center Offset')
    ax3.set_title('Key Point Center Offset (from previous frame)')
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Offset (units)')
    ax3.legend()
    ax3.grid(True)
    
    # 4. 速度变化量
    ax4 = plt.subplot(gs[3])
    ax4.plot(velocity_changes, color='red', label='Velocity Change')
    ax4.set_title('Velocity Change (from previous frame)')
    ax4.set_xlabel('Frame Index')
    ax4.set_ylabel('Change Magnitude')
    ax4.legend()
    ax4.grid(True)
    
    # 5. 速度标准差对比
    ax5 = plt.subplot(gs[4])
    instant_std = np.std(instant_velocities, axis=1)
    window_std = np.std(window_velocities, axis=1)
    
    ax5.plot(instant_std, color='blue', label='Instantaneous Std Dev')
    ax5.plot(window_std, color='orange', label=f'Window ({window_size} frames) Std Dev')
    ax5.set_xlabel('Frame Index')
    ax5.set_ylabel('Std Deviation')
    ax5.legend()
    ax5.grid(True)
    
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(args.data_folder, 'joint_metrics_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Metrics summary plot saved to: {save_path}")
    
    if args.show:
        plt.show()

def create_interactive_visualization(displacements, instant_velocities, window_velocities, 
                                    center_offsets, velocity_changes, args):
    """创建交互式可视化界面"""
    if instant_velocities is None or displacements is None:
        return
    
    num_joints = instant_velocities.shape[1]
    num_frames = instant_velocities.shape[0]
    displacement_magnitudes = np.linalg.norm(displacements, axis=2)
    
    # 创建图形和轴
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    plt.subplots_adjust(bottom=0.25, hspace=0.4)
    ax1, ax2, ax3, ax4 = axes
    
    # 1. 瞬时速度条形图
    ax1.set_title('Instantaneous Velocity')
    ax1.set_ylabel('Velocity (units/second)')
    instant_bars = ax1.bar(JOINT_NAMES, instant_velocities[0], color='skyblue')
    max_instant = np.max(instant_velocities)
    ax1.set_ylim(0, max_instant * 1.1)
    
    # 2. 窗口平均速度条形图
    ax2.set_title(f'Window-Averaged Velocity (current + previous {args.window_size-1} frames)')
    ax2.set_ylabel('Velocity (units/second)')
    window_bars = ax2.bar(JOINT_NAMES, window_velocities[0], color='salmon')
    max_window = np.max(window_velocities)
    ax2.set_ylim(0, max_window * 1.1)
    
    # 3. 位移条形图
    ax3.set_title('Displacement Magnitude')
    ax3.set_ylabel('Displacement (units)')
    disp_bars = ax3.bar(JOINT_NAMES, displacement_magnitudes[0], color='lightgreen')
    max_displacement = np.max(displacement_magnitudes)
    ax3.set_ylim(0, max_displacement * 1.1)
    
    # 4. 中心偏移和速度变化
    ax4.set_title('Key Metrics')
    ax4.set_ylabel('Magnitude')
    
    # 为了在同一轴上显示两个指标，使用双轴
    ax4_twin = ax4.twinx()
    offset_line, = ax4.plot([0], [center_offsets[0]], color='purple', label='Center Offset')
    change_line, = ax4_twin.plot([0], [velocity_changes[0]], color='red', label='Velocity Change')
    
    ax4.set_ylim(0, np.max(center_offsets) * 1.1)
    ax4_twin.set_ylim(0, np.max(velocity_changes) * 1.1)
    
    ax4.set_xlabel('Frame Index')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    
    # 添加滑块
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    frame_slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=num_frames-1,
        valinit=0,
        valstep=1
    )
    
    # 更新函数
    def update(val):
        frame = int(frame_slider.val)
        
        # 更新瞬时速度条
        for i, bar in enumerate(instant_bars):
            bar.set_height(instant_velocities[frame, i])
        
        # 更新窗口速度条
        for i, bar in enumerate(window_bars):
            bar.set_height(window_velocities[frame, i])
        
        # 更新位移条
        for i, bar in enumerate(disp_bars):
            bar.set_height(displacement_magnitudes[frame, i])
        
        # 更新中心偏移和速度变化曲线
        offset_line.set_data(range(frame+1), center_offsets[:frame+1])
        if frame > 0:  # 速度变化比帧索引晚1步
            change_line.set_data(range(frame), velocity_changes[:frame])
        
        # 调整坐标轴范围
        ax4.relim()
        ax4.autoscale_view()
        ax4_twin.relim()
        ax4_twin.autoscale_view()
        
        fig.suptitle(f'Frame: {frame}/{num_frames-1}', fontsize=14)
        fig.canvas.draw_idle()
    
    # 初始化标题
    fig.suptitle(f'Frame: 0/{num_frames-1}', fontsize=14)
    
    # 注册更新函数
    frame_slider.on_changed(update)
    
    # 添加播放/暂停按钮
    ax_button = plt.axes([0.8, 0.02, 0.15, 0.04])
    button = Button(ax_button, 'Play/Pause')
    
    anim_running = False
    animation_obj = None
    
    def animate(frame):
        frame_slider.set_val(frame)
        return instant_bars + window_bars + disp_bars + [offset_line, change_line]
    
    def toggle_animation(event):
        nonlocal anim_running, animation_obj
        if anim_running:
            animation_obj.event_source.stop()
            anim_running = False
        else:
            animation_obj = animation.FuncAnimation(
                fig, animate, frames=num_frames, interval=100, blit=True
            )
            anim_running = True
    
    button.on_clicked(toggle_animation)
    
    # 保存动画
    if args.save_animation:
        anim = animation.FuncAnimation(
            fig, animate, frames=num_frames, interval=100, blit=True
        )
        save_path = os.path.join(args.data_folder, 'joint_metrics_animation.gif')
        anim.save(save_path, writer='pillow', fps=10)
        print(f"Metrics animation saved to: {save_path}")
    
    if args.show:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Joint motion visualization tool with window stats')
    parser.add_argument('--data_folder', type=str, 
                      default="/home/haichao/workspace/real2sim/simulation/data/20250716-2018/video/demo_stero/",
                      required=False, help='Path to folder containing stero_hand_3d.pkl')
    parser.add_argument('--show', action='store_true', default=False, help='Whether to display plots')
    parser.add_argument('--save_animation', action='store_true', default=False, help='Whether to save animation')
    parser.add_argument('--fps', type=float, default=30.0, help='Video frame rate for velocity calculation')
    parser.add_argument('--window_size', type=int, default=5, help='Sliding window size (current + previous frames), default 5')
    
    args = parser.parse_args()
    
    # 加载数据并计算各项指标
    data = load_hand_3d_data(args.data_folder, args)
    if data[0] is None:
        print("Cannot calculate metrics data, program exiting")
        return
    
    (positions, displacements, instant_velocities, window_velocities, 
     center_offsets, velocity_changes) = data
    
    # 绘制静态summary图
    plot_summary(displacements, instant_velocities, window_velocities, 
                center_offsets, velocity_changes, args)
    
    # 创建交互式可视化
    create_interactive_visualization(displacements, instant_velocities, window_velocities, 
                                    center_offsets, velocity_changes, args)

if __name__ == '__main__':
    main()