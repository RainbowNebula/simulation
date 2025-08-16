import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import argparse
import pickle
import os  # 添加os模块用于路径处理

class TrajProcessor:
    def __init__(self, fps=30.0, window_size=6, threshold=0.05, no_normalize=False):
        """
        初始化轨迹处理器（处理单个物体的边界框序列）
        :param fps: 视频帧率，用于计算速度
        :param window_size: 计算窗口大小，使用当前帧的前N帧进行计算（N=window_size）
        :param threshold: 检测运动的阈值
        :param no_normalize: 是否不进行归一化处理
        """
        self.fps = fps
        self.window_size = max(1, window_size)  # 窗口大小至少为1
        self.threshold = max(0, threshold)
        self.no_normalize = no_normalize
        
        # 存储处理结果
        self.centers = None
        self.bbox_sizes = []  # 存储每个边界框的大小
        self.avg_bbox_size = 1.0
        self.total_frames = 0
        self.metrics = None
        self.start_frame = None
        self.end_frame = None
        self.trajectory = None

    def _extract_centers(self, bboxes):
        """从边界框序列中提取中心点，边界框格式为(4,)：[x1, y1, x2, y2]"""
        self.total_frames = len(bboxes)
        obj_centers = []
        self.bbox_sizes = []
        
        for bbox in bboxes:
            # 确保输入是边界框格式(4,)
            if len(bbox) != 4:
                raise ValueError(f"边界框格式错误，期望(4,)，实际{bbox.shape}")
            
            x1, y1, x2, y2 = bbox
            # 计算边界框中心点
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            obj_centers.append((center_x, center_y))
            
            # 计算边界框对角线长度作为大小（用于归一化）
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            size = np.sqrt(bbox_width**2 + bbox_height**2)
            self.bbox_sizes.append(size)
        
        self.centers = np.array(obj_centers)
        self.avg_bbox_size = np.mean(self.bbox_sizes) if self.bbox_sizes else 1.0

    def _calculate_metrics(self):
        """计算各项运动指标，使用统一的窗口大小"""
        if self.centers is None or len(self.centers) == 0:
            return
            
        center_changes = []
        normalized_changes = []
        
        # 计算中心点变化量（与前window_size帧的变化总和）
        # 从第window_size帧开始计算，因为需要前window_size帧的数据
        for i in range(self.window_size, len(self.centers)):
            total_distance = 0
            # 累加当前帧与前window_size帧的变化量
            for j in range(1, self.window_size + 1):
                dx = self.centers[i-j+1][0] - self.centers[i-j][0]
                dy = self.centers[i-j+1][1] - self.centers[i-j][1]
                total_distance += np.sqrt(dx**2 + dy**2)
            
            center_changes.append(total_distance)
            
            # 计算归一化变化量
            if self.no_normalize:
                normalized = total_distance
            else:
                normalized = total_distance / self.avg_bbox_size if self.avg_bbox_size > 0 else 0
            normalized_changes.append(normalized)
        
        # 计算速度 (像素/秒)
        time_interval = self.window_size / self.fps  # 使用窗口大小计算时间间隔
        instant_velocities = np.array(center_changes) / time_interval if center_changes else np.array([])
        normalized_velocities = np.array(normalized_changes) / time_interval if normalized_changes else np.array([])
        
        # 计算窗口平均速度（使用相同的窗口大小）
        window_velocities = []
        normalized_window_velocities = []
        
        for i in range(len(instant_velocities)):
            start = max(0, i - (self.window_size - 1))
            end = i + 1
            window_velocities.append(np.mean(instant_velocities[start:end]))
            normalized_window_velocities.append(np.mean(normalized_velocities[start:end]))
        
        self.metrics = (
            np.array(center_changes),
            np.array(normalized_changes),
            np.array(instant_velocities),
            np.array(normalized_velocities),
            np.array(window_velocities),
            np.array(normalized_window_velocities)
        )

    def _find_motion_frames(self):
        """找到运动的起始和终止帧"""
        if self.metrics is None:
            return
            
        # 确定使用哪种变化量进行阈值检测
        if self.no_normalize:
            changes = self.metrics[0]  # 原始变化量
        else:
            changes = self.metrics[1]  # 归一化变化量
        
        sequence_length = len(changes)
        start_idx = None
        stop_idx = None
        
        # 寻找运动起始帧：连续window_size帧高于阈值
        max_start_index = sequence_length - self.window_size
        for i in range(max_start_index + 1):
            if all(changes[i + j] > self.threshold for j in range(self.window_size)):
                start_idx = i
                break
        
        # 寻找运动停止帧：连续window_size帧低于阈值
        if start_idx is not None:
            max_stop_index = sequence_length - self.window_size
            start_search_idx = max(start_idx + self.window_size, 0)
            
            for i in range(start_search_idx, max_stop_index + 1):
                if all(changes[i + j] < self.threshold for j in range(self.window_size)):
                    stop_idx = i + (self.window_size - 1)  # 窗口最后一帧的索引
                    break
        
        # 转换为实际帧号（加上window_size偏移）
        self.start_frame = start_idx + self.window_size if start_idx is not None else None
        self.end_frame = stop_idx + self.window_size if stop_idx is not None else None
        
        # 提取运动轨迹
        if self.start_frame is not None and self.end_frame is not None:
            self.trajectory = self.centers[self.start_frame:self.end_frame+1]
        elif self.start_frame is not None:
            self.trajectory = self.centers[self.start_frame:]
        else:
            self.trajectory = np.array([])

    def _plot_summary(self, save_path=None):
        """绘制汇总图表"""
        if self.metrics is None:
            return
            
        center_changes, normalized_changes = self.metrics[0], self.metrics[1]
        instant_velocities, normalized_velocities = self.metrics[2], self.metrics[3]
        window_velocities, normalized_window_velocities = self.metrics[4], self.metrics[5]
        
        fig = plt.figure(figsize=(14, 12))
        gs = GridSpec(2, 1, height_ratios=[1, 1])
        
        # 标题显示当前窗口大小
        window_title = f' (window size: {self.window_size} frames)'
        
        if self.no_normalize:
            # 原始中心点变化量曲线
            ax1 = plt.subplot(gs[0])
            ax1.plot(center_changes, label='Center Point Displacement')
            
            # 标记起始和终止帧
            if self.start_frame is not None:
                start_idx = self.start_frame - self.window_size
                if start_idx < len(center_changes):
                    ax1.axvline(x=start_idx, color='r', linestyle='--')
                    ax1.text(start_idx, 0, f' Start: {self.start_frame}', color='r', rotation=90)
            
            if self.end_frame is not None:
                end_idx = self.end_frame - self.window_size
                if end_idx < len(center_changes):
                    ax1.axvline(x=end_idx, color='g', linestyle='--')
                    ax1.text(end_idx, 0, f' Stop: {self.end_frame}', color='g', rotation=90)
            
            ax1.axhline(y=self.threshold, color='k', linestyle='--', label=f'Threshold: {self.threshold}')
            ax1.set_title(f'Center Point Displacement{window_title}')
            ax1.set_xlabel('Frame Index')
            ax1.set_ylabel('Displacement (pixels)')
            ax1.legend()
            ax1.grid(True)
            
            # 原始速度曲线
            ax2 = plt.subplot(gs[1])
            ax2.plot(instant_velocities, label='Instantaneous Velocity', alpha=0.6)
            ax2.plot(window_velocities, label=f'Window-averaged{window_title}', linestyle='--')
            ax2.set_title(f'Velocity{window_title}')
            ax2.set_xlabel('Frame Index')
            ax2.set_ylabel('Velocity (pixels/second)')
            ax2.legend()
            ax2.grid(True)
        else:
            # 归一化中心点变化量曲线
            ax1 = plt.subplot(gs[0])
            ax1.plot(normalized_changes, label='Normalized Displacement')
            
            # 标记起始和终止帧
            if self.start_frame is not None:
                start_idx = self.start_frame - self.window_size
                if start_idx < len(normalized_changes):
                    ax1.axvline(x=start_idx, color='r', linestyle='--')
                    ax1.text(start_idx, 0, f' Start: {self.start_frame}', color='r', rotation=90)
            
            if self.end_frame is not None:
                end_idx = self.end_frame - self.window_size
                if end_idx < len(normalized_changes):
                    ax1.axvline(x=end_idx, color='g', linestyle='--')
                    ax1.text(end_idx, 0, f' Stop: {self.end_frame}', color='g', rotation=90)
            
            ax1.axhline(y=self.threshold, color='k', linestyle='--', label=f'Threshold: {self.threshold}')
            ax1.set_title(f'Normalized Displacement{window_title}')
            ax1.set_xlabel('Frame Index')
            ax1.set_ylabel('Normalized Value')
            ax1.legend()
            ax1.grid(True)
            
            # 归一化速度曲线
            ax2 = plt.subplot(gs[1])
            ax2.plot(normalized_velocities, label='Instantaneous Velocity', alpha=0.6)
            ax2.plot(normalized_window_velocities, label=f'Window-averaged{window_title}', linestyle='--')
            ax2.set_title(f'Normalized Velocity{window_title}')
            ax2.set_xlabel('Frame Index')
            ax2.set_ylabel('Normalized Velocity')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存至: {save_path}")
        
        plt.show()

    def process(self, bboxes, show_plot=False, save_plot_path=None):
        """
        处理边界框序列并返回结果
        :param bboxes: 边界框序列，每个元素是单帧的边界框，格式为(4,)：[x1, y1, x2, y2]
        :param show_plot: 是否显示图表
        :param save_plot_path: 图表保存路径
        :return: 字典，包含起始帧、终止帧和轨迹
        """
        # 提取中心点
        self._extract_centers(bboxes)
        
        # 计算运动指标
        self._calculate_metrics()
        
        # 找到运动帧和轨迹
        self._find_motion_frames()
        
        # 输出运动分析结果
        print(f"运动分析结果:")
        print(f"起始帧: {self.start_frame}")
        print(f"终止帧: {self.end_frame}")
        print(f"轨迹长度: {len(self.trajectory)} 帧")
        
        # 绘制图表
        if show_plot:
            self._plot_summary(save_plot_path)
            print(f"图表已保存至: {save_plot_path}")
        
        # 返回结果
        return {
            'start_frame': self.start_frame,
            'end_frame': self.end_frame,
            'trajectory': self.trajectory
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='物体检测边界框轨迹分析工具')
    parser.add_argument('--result_path', type=str, required=False, 
                      default="/home/haichao/workspace/real2sim/simulation/data/20250716-2018/video/detect_results.pkl",   
                      help='检测结果pkl文件路径')
    parser.add_argument('--show', action='store_true', default=False, 
                      help='是否显示图形')
    parser.add_argument('--fps', type=float, default=30.0, 
                      help='视频帧率，用于计算速度')
    parser.add_argument('--window_size', type=int, default=6,
                      help='速度计算的窗口大小，默认6帧')
    parser.add_argument('--threshold', type=float, default=0.05,
                      help='设定阈值，用于检测运动开始和结束')
    parser.add_argument('--no_normalize', action='store_true', default=False,
                      help='不进行归一化处理，使用原始像素值')
    parser.add_argument('-v', '--vis', action='store_true', default=True,
                      help='是否可视化')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.window_size < 1:
        print(f"警告: 窗口大小必须至少为1，已自动调整为1")
        args.window_size = 1
    if args.threshold < 0:
        print(f"警告: 阈值不能为负数，已自动调整为0")
        args.threshold = 0
    
    # 准备保存路径
    if args.result_path and os.path.exists(args.result_path):
        save_plot_path = os.path.join(os.path.dirname(args.result_path), 'process_obj_traj.png')
    else:
        save_plot_path = 'process_obj_traj.png'
    

    # 加载检测结果
    with open(args.result_path, 'rb') as f:
        data = pickle.load(f)
    
    bboxes = data[0]['bbox']
    # 假设数据结构中第一个元素包含边界框信息
    # 根据实际数据结构调整这里的索引
    # if isinstance(data, list) and len(data) > 0:
    #     # 假设data[0]是包含'bbox'键的字典
    #     if 'bbox' in data[0]:
    #         bboxes = data[0]['bbox']
    #         print(f"成功加载边界框数据，共 {len(bboxes)} 帧")
    #     else:
    #         raise ValueError("数据中未找到'bbox'键")
    # else:
    #     raise ValueError("数据格式不符合预期")
    
    # 初始化处理器并处理数据
    processor = TrajProcessor(
        fps=args.fps,
        window_size=args.window_size,
        threshold=args.threshold,
        no_normalize=args.no_normalize
    )
    result = processor.process(
        bboxes=bboxes,
        show_plot=args.show or args.vis,
        save_plot_path=save_plot_path
    )
        

