import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import argparse
import os
import pickle
from matplotlib.gridspec import GridSpec

def load_detection_results(data_path):
    """加载检测结果并提取中心点信息"""
    if not os.path.exists(data_path):
        print(f"文件 {data_path} 不存在")
        return None, None, None, None, None
    
    with open(data_path, "rb") as f:
        detect_results = pickle.load(f)
    
    # 提取物体ID和对应的类别名称
    object_ids = sorted(detect_results.keys())
    object_classes = {obj_id: detect_results[obj_id]['classname'] 
                     for obj_id in object_ids}
    
    # 确定最大帧数
    max_frames = max(len(detect_results[obj_id]['bbox']) for obj_id in object_ids)
    
    # 提取每个物体在每帧的中心点和边界框大小
    centers = {}
    bbox_sizes = {}  # 用于归一化的边界框对角线长度
    
    for obj_id in object_ids:
        bboxes = detect_results[obj_id]['bbox']
        obj_centers = []
        obj_sizes = []
        
        for bbox in bboxes:
            # 计算边界框中心点 (x_center, y_center)
            x1, y1, x2, y2 = bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            obj_centers.append((center_x, center_y))
            
            # 计算边界框对角线长度（用于归一化）
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            diagonal = np.sqrt(bbox_width**2 + bbox_height**2)
            obj_sizes.append(diagonal)
        
        # 用最后一个有效值填充缺失帧
        if len(obj_centers) < max_frames and len(obj_centers) > 0:
            last_center = obj_centers[-1]
            last_size = obj_sizes[-1] if obj_sizes else 1.0
            
            while len(obj_centers) < max_frames:
                obj_centers.append(last_center)
                obj_sizes.append(last_size)
        
        centers[obj_id] = np.array(obj_centers)
        bbox_sizes[obj_id] = np.array(obj_sizes)
    
    # 计算每个物体的平均边界框大小（用于全局归一化）
    avg_bbox_sizes = {obj_id: np.mean(sizes) for obj_id, sizes in bbox_sizes.items()}
    
    return object_ids, object_classes, centers, avg_bbox_sizes, max_frames

def calculate_metrics(centers, avg_bbox_sizes, object_ids, max_frames, 
                     diff_frames=1, window_size=5, fps=30.0, normalize=True):
    """计算中心点变化量和速度指标，支持指定与前几帧的差值求和"""
    # 确保差值帧数至少为1
    diff_frames = max(1, diff_frames)
    
    # 计算与前N帧的中心点变化量总和
    center_changes = {obj_id: [] for obj_id in object_ids}
    normalized_changes = {obj_id: [] for obj_id in object_ids}
    
    for obj_id in object_ids:
        obj_centers = centers[obj_id]
        avg_size = avg_bbox_sizes[obj_id] if avg_bbox_sizes[obj_id] > 0 else 1.0
        
        for i in range(diff_frames, len(obj_centers)):
            # 计算与前diff_frames帧的差值总和
            total_distance = 0
            for j in range(1, diff_frames + 1):
                dx = obj_centers[i-j+1][0] - obj_centers[i-j][0]
                dy = obj_centers[i-j+1][1] - obj_centers[i-j][1]
                total_distance += np.sqrt(dx**2 + dy**2)
            
            center_changes[obj_id].append(total_distance)
            
            # 归一化变化量
            if normalize:
                normalized = total_distance / avg_size
                normalized_changes[obj_id].append(normalized)
            else:
                normalized_changes[obj_id].append(total_distance)
    
    # 计算速度（像素/秒）
    time_interval = diff_frames / fps  # 总时间间隔
    instant_velocities = {
        obj_id: np.array(changes) / time_interval 
        for obj_id, changes in center_changes.items()
    }
    
    # 计算归一化速度
    normalized_velocities = {
        obj_id: np.array(changes) / time_interval 
        for obj_id, changes in normalized_changes.items()
    }
    
    # 计算窗口速度（当前帧和前N帧的平均速度）
    window_velocities = {}
    normalized_window_velocities = {}
    window_size = max(1, window_size)
    
    for obj_id in object_ids:
        # 原始速度窗口平均
        velocities = instant_velocities[obj_id]
        win_vel = []
        
        # 归一化速度窗口平均
        norm_velocities = normalized_velocities[obj_id]
        norm_win_vel = []
        
        for i in range(len(velocities)):
            start = max(0, i - (window_size - 1))
            end = i + 1
            win_vel.append(np.mean(velocities[start:end]))
            norm_win_vel.append(np.mean(norm_velocities[start:end]))
        
        window_velocities[obj_id] = np.array(win_vel)
        normalized_window_velocities[obj_id] = np.array(norm_win_vel)
    
    return (center_changes, normalized_changes, 
            instant_velocities, normalized_velocities,
            window_velocities, normalized_window_velocities,
            window_size, diff_frames)

def find_first_exceed_threshold(object_ids, changes, threshold):
    """找到首次超过阈值的帧序号"""
    first_exceed = {}
    for obj_id in object_ids:
        obj_changes = changes[obj_id]
        for frame_idx, value in enumerate(obj_changes):
            if value > threshold:
                # 实际帧号需要加上diff_frames的偏移（因为我们从diff_frames开始计算）
                first_exceed[obj_id] = frame_idx + diff_frames
                break
        if obj_id not in first_exceed:
            first_exceed[obj_id] = None  # 从未超过阈值
    
    return first_exceed


def find_motion_frames_with_slide_window(
        object_ids, 
        changes, 
        threshold, 
        start_window=3,  # 连续多少帧高于阈值视为开始
        stop_window=3    # 连续多少帧低于阈值视为停止
    ):
    """
    找到每个对象的运动起始和停止帧，支持自定义滑动窗口大小：
    
    参数:
        object_ids: 对象ID列表
        changes: 包含每个对象偏移量变化的字典
        threshold: 判断阈值
        start_window: 连续多少帧高于阈值视为运动开始（默认3）
        stop_window: 连续多少帧低于阈值视为运动停止（默认3）
    
    返回值:
        字典，每个键是对象ID，值是包含两个元素的元组:
        (start_frame, stop_frame)
        其中：
            start_frame: 运动起始帧（窗口中第一帧的序号）
            stop_frame: 运动停止帧（窗口中最后一帧的序号）
    """
    motion_frames = {}
    
    for obj_id in object_ids:
        obj_changes = changes[obj_id]
        start_frame = None
        stop_frame = None
        sequence_length = len(obj_changes)
        
        # 寻找运动起始帧：连续start_window帧高于阈值
        # 确保有足够的帧来形成窗口
        max_start_index = sequence_length - start_window
        for i in range(max_start_index + 1):
            # 检查窗口内所有帧是否都超过阈值
            window_exceeds = all(
                obj_changes[i + j] > threshold 
                for j in range(start_window)
            )
            if window_exceeds:
                start_frame = i + diff_frames  # 起始帧为窗口的第一帧
                break
        
        # 如果找到了起始帧，再寻找运动停止帧
        if start_frame is not None:
            # 计算起始帧在列表中的索引（减去偏移）
            start_idx = start_frame - diff_frames
            # 从起始窗口之后开始寻找停止窗口
            max_stop_index = sequence_length - stop_window
            # 确保从起始窗口之后开始搜索
            start_search_idx = max(start_idx + start_window, 0)
            
            for i in range(start_search_idx, max_stop_index + 1):
                # 检查窗口内所有帧是否都低于阈值
                window_below = all(
                    obj_changes[i + j] < threshold 
                    for j in range(stop_window)
                )
                if window_below:
                    # 停止帧为窗口的最后一帧
                    stop_frame = i + (stop_window - 1) + diff_frames
                    break
        
        motion_frames[obj_id] = (start_frame, stop_frame)
    
    return motion_frames
    



def plot_summary(object_ids, object_classes, center_changes, normalized_changes,
                instant_velocities, normalized_velocities,
                window_velocities, normalized_window_velocities, 
                window_size, diff_frames, normalize, threshold, first_exceed):
    """绘制静态汇总图表，包含阈值线和首次超过标记"""
    if not object_ids:
        return
    
    # 根据是否归一化确定显示内容
    if normalize:
        fig = plt.figure(figsize=(14, 12))
        gs = GridSpec(2, 1, height_ratios=[1, 1])
        
        # 1. 归一化中心点变化量曲线
        ax1 = plt.subplot(gs[0])
        for obj_id in object_ids:
            ax1.plot(normalized_changes[obj_id], label=f"{object_classes[obj_id]} (ID: {obj_id})")
            # 标记首次超过阈值的点
            if first_exceed.get(obj_id) is not None:
                frame_idx = first_exceed[obj_id] - diff_frames  # 转换为相对索引
                if frame_idx < len(normalized_changes[obj_id]):
                    ax1.scatter(frame_idx, normalized_changes[obj_id][frame_idx], 
                               color='red', zorder=5)
                    ax1.text(frame_idx, normalized_changes[obj_id][frame_idx], 
                             f'  Frame {first_exceed[obj_id]}', color='red')
        
        # 添加阈值线
        if threshold is not None:
            ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
        
        ax1.set_title(f'Normalized Center Point Displacement (sum of {diff_frames} frames)')
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Normalized Displacement')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 归一化速度曲线
        ax2 = plt.subplot(gs[1])
        for obj_id in object_ids:
            ax2.plot(normalized_velocities[obj_id], 
                    label=f"{object_classes[obj_id]} (Instantaneous)", alpha=0.6)
            ax2.plot(normalized_window_velocities[obj_id], 
                    label=f"{object_classes[obj_id]} (Window-averaged)", linestyle='--')
        
        ax2.set_title(f'Normalized Velocity - Window size: {window_size}, Sum of {diff_frames} frames')
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Normalized Velocity')
        ax2.legend()
        ax2.grid(True)
    
    else:
        fig = plt.figure(figsize=(14, 12))
        gs = GridSpec(2, 1, height_ratios=[1, 1])
        
        # 1. 原始中心点变化量曲线
        ax1 = plt.subplot(gs[0])
        for obj_id in object_ids:
            ax1.plot(center_changes[obj_id], label=f"{object_classes[obj_id]} (ID: {obj_id})")
            # 标记首次超过阈值的点
            if first_exceed.get(obj_id) is not None:
                frame_idx = first_exceed[obj_id] - diff_frames  # 转换为相对索引
                if frame_idx < len(center_changes[obj_id]):
                    ax1.scatter(frame_idx, center_changes[obj_id][frame_idx], 
                               color='red', zorder=5)
                    ax1.text(frame_idx, center_changes[obj_id][frame_idx], 
                             f'  Frame {first_exceed[obj_id]}', color='red')
        
        # 添加阈值线
        if threshold is not None:
            ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
        
        ax1.set_title(f'Original Center Point Displacement (sum of {diff_frames} frames)')
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Displacement (pixels)')
        ax1.legend()
        ax1.grid(True)
        
        # 2. 原始速度曲线
        ax2 = plt.subplot(gs[1])
        for obj_id in object_ids:
            ax2.plot(instant_velocities[obj_id], 
                    label=f"{object_classes[obj_id]} (Instantaneous)", alpha=0.6)
            ax2.plot(window_velocities[obj_id], 
                    label=f"{object_classes[obj_id]} (Window-averaged)", linestyle='--')
        
        ax2.set_title(f'Original Velocity - Window size: {window_size}, Sum of {diff_frames} frames')
        ax2.set_xlabel('Frame Index')
        ax2.set_ylabel('Velocity (pixels/second)')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    
    # 保存图像
    save_path = os.path.join(os.path.dirname(args.result_path), 'detection_metrics_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"汇总图表已保存至: {save_path}")
    
    if args.show:
        plt.show()

def create_interactive_visualization(object_ids, object_classes, 
                                    center_changes, normalized_changes,
                                    instant_velocities, normalized_velocities,
                                    window_velocities, normalized_window_velocities, 
                                    max_frames, window_size, diff_frames, normalize,
                                    threshold, first_exceed):
    """创建交互式可视化界面"""
    if not object_ids:
        return
    
    num_objects = len(object_ids)
    max_frames = min(max(len(center_changes[obj_id]) for obj_id in object_ids), max_frames)
    if max_frames == 0:
        return
    
    # 创建图形和轴
    if normalize:
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        plt.subplots_adjust(bottom=0.25, hspace=0.5)
        ax1, ax2 = axes
        
        # 1. 归一化中心点变化量条形图
        ax1.set_title(f'Normalized Center Point Displacement (sum of {diff_frames} frames)')
        ax1.set_ylabel('Normalized Displacement')
        norm_change_bars = ax1.bar([str(obj_id) for obj_id in object_ids], 
                                  [normalized_changes[obj_id][0] for obj_id in object_ids], 
                                  color='green')
        
        # 添加阈值线
        if threshold is not None:
            ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
            ax1.legend()
        
        max_norm_change = max(np.max(normalized_changes[obj_id]) for obj_id in object_ids)
        # 确保阈值可见
        if threshold is not None:
            max_norm_change = max(max_norm_change, threshold * 1.1)
        ax1.set_ylim(0, max_norm_change * 1.1)
        ax1.set_xticklabels([f"{object_classes[obj_id]}\n(ID: {obj_id})" 
                            for obj_id in object_ids], rotation=30, ha='right')
        
        # 2. 归一化速度条形图
        ax2.set_title(f'Normalized Velocity Comparison - Window size: {window_size}')
        ax2.set_ylabel('Normalized Velocity')
        
        bar_width = 0.35
        x = np.arange(num_objects)
        norm_instant_bars = ax2.bar(x - bar_width/2, 
                                  [normalized_velocities[obj_id][0] for obj_id in object_ids],
                                  bar_width, label='Instantaneous', color='blue')
        norm_window_bars = ax2.bar(x + bar_width/2, 
                                 [normalized_window_velocities[obj_id][0] for obj_id in object_ids],
                                 bar_width, label='Window-averaged', color='red')
        
        max_norm_velocity = max(np.max(normalized_velocities[obj_id]) for obj_id in object_ids)
        max_norm_velocity = max(max_norm_velocity, max(np.max(normalized_window_velocities[obj_id]) for obj_id in object_ids))
        ax2.set_ylim(0, max_norm_velocity * 1.1)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"ID: {obj_id}" for obj_id in object_ids])
        ax2.legend()
        
        all_bars = list(norm_change_bars) + list(norm_instant_bars) + list(norm_window_bars)
    
    else:
        fig, axes = plt.subplots(2, 1, figsize=(14, 12))
        plt.subplots_adjust(bottom=0.25, hspace=0.5)
        ax1, ax2 = axes
        
        # 1. 原始中心点变化量条形图
        ax1.set_title(f'Original Center Point Displacement (sum of {diff_frames} frames)')
        ax1.set_ylabel('Displacement (pixels)')
        change_bars = ax1.bar([str(obj_id) for obj_id in object_ids], 
                             [center_changes[obj_id][0] for obj_id in object_ids], 
                             color='lightgreen')
        
        # 添加阈值线
        if threshold is not None:
            ax1.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold}')
            ax1.legend()
        
        max_change = max(np.max(center_changes[obj_id]) for obj_id in object_ids)
        # 确保阈值可见
        if threshold is not None:
            max_change = max(max_change, threshold * 1.1)
        ax1.set_ylim(0, max_change * 1.1)
        ax1.set_xticklabels([f"{object_classes[obj_id]}\n(ID: {obj_id})" 
                            for obj_id in object_ids], rotation=30, ha='right')
        
        # 2. 原始速度条形图
        ax2.set_title(f'Original Velocity Comparison - Window size: {window_size}')
        ax2.set_ylabel('Velocity (pixels/second)')
        
        bar_width = 0.35
        x = np.arange(num_objects)
        instant_bars = ax2.bar(x - bar_width/2, 
                              [instant_velocities[obj_id][0] for obj_id in object_ids],
                              bar_width, label='Instantaneous', color='skyblue')
        window_bars = ax2.bar(x + bar_width/2, 
                             [window_velocities[obj_id][0] for obj_id in object_ids],
                             bar_width, label='Window-averaged', color='salmon')
        
        max_velocity = max(np.max(instant_velocities[obj_id]) for obj_id in object_ids)
        max_velocity = max(max_velocity, max(np.max(window_velocities[obj_id]) for obj_id in object_ids))
        ax2.set_ylim(0, max_velocity * 1.1)
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"ID: {obj_id}" for obj_id in object_ids])
        ax2.legend()
        
        all_bars = list(change_bars) + list(instant_bars) + list(window_bars)
    
    # 添加滑块
    ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
    frame_slider = Slider(
        ax=ax_slider,
        label='Frame',
        valmin=0,
        valmax=max_frames - 1,
        valinit=0,
        valstep=1
    )
    
    # 更新函数
    def update(val):
        frame = int(frame_slider.val)
        if frame >= max_frames:
            return
        
        if normalize:
            # 更新归一化中心点变化量条
            for i, obj_id in enumerate(object_ids):
                if frame < len(normalized_changes[obj_id]):
                    height = normalized_changes[obj_id][frame]
                    norm_change_bars[i].set_height(height)
                    # 如果超过阈值，改变颜色
                    if threshold is not None and height > threshold:
                        norm_change_bars[i].set_color('red')
                    else:
                        norm_change_bars[i].set_color('green')
            
            # 更新归一化速度条
            for i, obj_id in enumerate(object_ids):
                if frame < len(normalized_velocities[obj_id]):
                    norm_instant_bars[i].set_height(normalized_velocities[obj_id][frame])
                    norm_window_bars[i].set_height(normalized_window_velocities[obj_id][frame])
        else:
            # 更新原始中心点变化量条
            for i, obj_id in enumerate(object_ids):
                if frame < len(center_changes[obj_id]):
                    height = center_changes[obj_id][frame]
                    change_bars[i].set_height(height)
                    # 如果超过阈值，改变颜色
                    if threshold is not None and height > threshold:
                        change_bars[i].set_color('red')
                    else:
                        change_bars[i].set_color('lightgreen')
            
            # 更新原始速度条
            for i, obj_id in enumerate(object_ids):
                if frame < len(instant_velocities[obj_id]):
                    instant_bars[i].set_height(instant_velocities[obj_id][frame])
                    window_bars[i].set_height(window_velocities[obj_id][frame])
        
        fig.suptitle(f'Frame: {frame + diff_frames}/{max_frames + diff_frames - 1} (sum of {diff_frames} frames)', fontsize=14)
        fig.canvas.draw_idle()
    
    # 初始化标题
    fig.suptitle(f'Frame: {diff_frames}/{max_frames + diff_frames - 1} (sum of {diff_frames} frames)', fontsize=14)
    
    # 注册更新函数
    frame_slider.on_changed(update)
    
    # 添加播放/暂停按钮
    ax_button = plt.axes([0.8, 0.02, 0.15, 0.04])
    button = Button(ax_button, 'Play/Pause')
    
    anim_running = False
    animation_obj = None
    
    def animate(frame):
        if frame < max_frames:
            frame_slider.set_val(frame)
        return all_bars
    
    def toggle_animation(event):
        nonlocal anim_running, animation_obj
        if anim_running:
            animation_obj.event_source.stop()
            anim_running = False
        else:
            animation_obj = animation.FuncAnimation(
                fig, animate, frames=max_frames, interval=100, blit=True
            )
            anim_running = True
    
    button.on_clicked(toggle_animation)
    
    # 保存动画
    if args.save_animation:
        anim = animation.FuncAnimation(
            fig, animate, frames=max_frames, interval=100, blit=True
        )
        save_path = os.path.join(os.path.dirname(args.result_path), 'detection_metrics_animation.gif')
        anim.save(save_path, writer='pillow', fps=10)
        print(f"动画已保存至: {save_path}")
    
    if args.show:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='物体检测中心点变化和速度可视化工具')
    parser.add_argument('--result_path', type=str, required=False, 
                      default="/home/haichao/workspace/real2sim/simulation/data/20250716-2018/video/detect_results.pkl",   
                      help='检测结果pkl文件路径')
    parser.add_argument('--show', action='store_true', default=False, 
                      help='是否显示图形')
    parser.add_argument('--save_animation', action='store_true', default=False, 
                      help='是否保存动画')
    parser.add_argument('--fps', type=float, default=30.0, 
                      help='视频帧率，用于计算速度')
    parser.add_argument('--ids', type=int, nargs='+', default=None,
                      help='指定要可视化的物体ID，例如 --ids 1 3 5')
    parser.add_argument('--window_size', type=int, default=6,
                      help='速度计算的窗口大小（当前帧+向前的帧数），默认5帧')
    parser.add_argument('--diff_frames', type=int, default=5,
                      help='计算差值的帧数，即与前几帧的变化量求和，默认1帧')
    parser.add_argument('--threshold', type=float, default=0.05,
                      help='设定阈值，用于检测首次超过该值的帧序号')
    parser.add_argument('--no_normalize', action='store_true', default=False,
                      help='不进行归一化处理，使用原始像素值')
    
    global args, diff_frames
    args = parser.parse_args()
    
    # 验证参数
    if args.window_size < 1:
        print(f"警告: 窗口大小必须至少为1，已自动调整为1")
        args.window_size = 1
    if args.diff_frames < 1:
        print(f"警告: 差值帧数必须至少为1，已自动调整为1")
        args.diff_frames = 1
    diff_frames = args.diff_frames
    
    # 验证阈值
    if args.threshold is not None and args.threshold < 0:
        print(f"警告: 阈值不能为负数，已自动调整为0")
        args.threshold = 0
    
    # 加载检测结果
    data = load_detection_results(args.result_path)
    if data[0] is None:
        print("无法加载检测结果，程序退出")
        return
    
    all_object_ids, object_classes, centers, avg_bbox_sizes, max_frames = data
    
    # 处理指定ID
    if args.ids is not None:
        # 验证指定的ID是否存在
        valid_ids = [obj_id for obj_id in args.ids if obj_id in all_object_ids]
        invalid_ids = [obj_id for obj_id in args.ids if obj_id not in all_object_ids]
        
        if invalid_ids:
            print(f"警告: 以下ID不存在于检测结果中: {invalid_ids}")
        
        if not valid_ids:
            print("没有有效的物体ID可供可视化，程序退出")
            return
            
        object_ids = valid_ids
        print(f"将可视化以下物体ID: {object_ids}")
    else:
        object_ids = all_object_ids
        print(f"将可视化所有物体ID: {object_ids}")
    
    # 计算各项指标（带归一化选项）
    metrics = calculate_metrics(
        centers, avg_bbox_sizes, object_ids, max_frames, 
        args.diff_frames, args.window_size, args.fps, not args.no_normalize
    )
    
    (center_changes, normalized_changes, 
     instant_velocities, normalized_velocities,
     window_velocities, normalized_window_velocities,
     window_size, diff_frames) = metrics
    
    # 确定使用哪种变化量进行阈值检测
    if not args.no_normalize:
        changes_for_threshold = normalized_changes
    else:
        changes_for_threshold = center_changes
    
    # 查找首次超过阈值的帧序号
    # first_exceed = {}
    # if args.threshold is not None:
    #     first_exceed = find_first_exceed_threshold(object_ids, changes_for_threshold, args.threshold)
        
    #     # 输出结果
    #     print("\n首次超过阈值的帧序号:")
    #     for obj_id, frame in first_exceed.items():
    #         if frame is not None:
    #             print(f"物体 {object_classes[obj_id]} (ID: {obj_id}): 第 {frame} 帧")
    #         else:
    #             print(f"物体 {object_classes[obj_id]} (ID: {obj_id}): 从未超过阈值")
    # else:
    #     print("\n未设置阈值，跳过阈值检测")

    first_exceed = {}
    if args.threshold is not None:
        frames = find_motion_frames_with_slide_window(object_ids, changes_for_threshold, args.threshold)
        for obj_id, (start_frame, stop_frame) in frames.items():
            first_exceed[obj_id] = start_frame
            print(f"对象 {obj_id}: 开始于 {start_frame}, 停止于 {stop_frame}")

    
    # 绘制静态汇总图
    plot_summary(object_ids, object_classes, 
                center_changes, normalized_changes,
                instant_velocities, normalized_velocities,
                window_velocities, normalized_window_velocities, 
                window_size, diff_frames, not args.no_normalize,
                args.threshold, first_exceed)
    
    # 创建交互式可视化
    create_interactive_visualization(object_ids, object_classes, 
                                    center_changes, normalized_changes,
                                    instant_velocities, normalized_velocities,
                                    window_velocities, normalized_window_velocities, 
                                    max_frames, window_size, diff_frames, not args.no_normalize,
                                    args.threshold, first_exceed)

if __name__ == '__main__':
    main()
