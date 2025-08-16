import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os
from datetime import datetime
import apriltag  # 使用apriltag库

# --- 配置参数 ---
TAG_SIZE = 0.06  # AprilTag的边长（米），根据实际标签尺寸修改
IMAGE_SAVE_DIR_BASE = "apriltag_images"  # 图像保存根目录

def setup_arg_parser():
    """设置命令行参数解析"""
    parser = argparse.ArgumentParser(description="RealSense AprilTag Detection and Visualization")
    
    parser.add_argument(
        '--tag_family',
        type=str,
        default='tag36h10',
        help="AprilTag家族类型 (例如: tag36h11, tag36h10, tag25h9等)"
    )
    
    return parser.parse_args()


def setup_realsense():
    """初始化并配置RealSense相机，返回相机及内参"""
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 启用彩色流和深度流
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        print(f"启动RealSense相机时出错: {e}")
        print("请确认相机已连接且未被其他程序占用。")
        return None, None, None
        
    # 获取相机内参
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()
    
    # 打印相机内参
    print("\n--- RealSense相机内参 ---")
    print(f"图像尺寸: {intrinsics.width}x{intrinsics.height}")
    print(f"内参矩阵:")
    print(f"  fx: {intrinsics.fx:.2f} (x方向焦距)")
    print(f"  fy: {intrinsics.fy:.2f} (y方向焦距)")
    print(f"  cx: {intrinsics.ppx:.2f} (主点x坐标)")
    print(f"  cy: {intrinsics.ppy:.2f} (主点y坐标)")
    print(f"畸变系数: {intrinsics.coeffs}")
    print("----------------------------------------\n")
    
    # 创建内参矩阵（用于姿态估计）
    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float64)
    
    distortion_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)
    
    return pipeline, camera_matrix, distortion_coeffs


def setup_apriltag_detector(tag_family):
    """初始化AprilTag检测器（兼容更多版本apriltag库）"""
    try:
        # 尝试创建检测器，使用最基础的参数集以提高兼容性
        options = {
            'families': tag_family,
            'nthreads': 2,
            'refine_edges': 1,
            'debug': 0
        }
        
        # 尝试不同的参数组合来适配不同版本
        try:
            # 版本1: 尝试使用完整参数
            detector_options = apriltag.DetectorOptions(
                families=options['families'],
                nthreads=options['nthreads'],
                quad_decimate=1.0,
                quad_sigma=0.0,
                refine_edges=options['refine_edges'],
                decode_sharpening=0.25,
                debug=options['debug']
            )
            return apriltag.Detector(detector_options)
        except TypeError:
            try:
                # 版本2: 移除可能不支持的quad_sigma参数
                detector_options = apriltag.DetectorOptions(
                    families=options['families'],
                    nthreads=options['nthreads'],
                    quad_decimate=1.0,
                    refine_edges=options['refine_edges'],
                    decode_sharpening=0.25,
                    debug=options['debug']
                )
                return apriltag.Detector(detector_options)
            except TypeError:
                # 版本3: 只使用最基础的参数
                detector_options = apriltag.DetectorOptions(
                    families=options['families'],
                    nthreads=options['nthreads'],
                    refine_edges=options['refine_edges']
                )
                return apriltag.Detector(detector_options)
                
    except Exception as e:
        print(f"初始化AprilTag检测器失败: {e}")
        print("可能的原因是标签家族不被支持或库版本不兼容，请尝试:")
        print("1. 更新apriltag库: pip install --upgrade apriltag")
        print("2. 尝试不同的标签家族，如: tag36h11")
        exit(1)


def draw_tag_info(image, tag):
    """在图像上绘制检测到的AprilTag边界框和中心坐标"""
    # 绘制标签边界框
    for i in range(4):
        cv2.line(image, 
                 (int(tag.corners[i][0]), int(tag.corners[i][1])), 
                 (int(tag.corners[(i+1)%4][0]), int(tag.corners[(i+1)%4][1])), 
                 (0, 255, 0), 2)
    
    # 计算标签中心坐标
    center_x = int(np.mean(tag.corners[:, 0]))
    center_y = int(np.mean(tag.corners[:, 1]))
    
    # 显示标签ID和中心坐标
    tag_id = tag.tag_id
    text = f"ID: {tag_id} | 中心: ({center_x}, {center_y})"
    cv2.putText(image, text, 
                (int(tag.corners[0][0]), int(tag.corners[0][1]) - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 绘制中心标记
    cv2.circle(image, (center_x, center_y), 5, (0, 0, 255), -1)  # 红色实心圆标记中心
    
    return image, (center_x, center_y)


def main():
    """主函数，执行AprilTag检测和可视化流程"""
    args = setup_arg_parser()
    pipeline, camera_matrix, distortion_coeffs = setup_realsense()
    if pipeline is None: return

    # 初始化AprilTag检测器
    at_detector = setup_apriltag_detector(args.tag_family)
    
    # 创建图片保存文件夹
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_save_dir = os.path.join(IMAGE_SAVE_DIR_BASE, timestamp)
    os.makedirs(image_save_dir, exist_ok=True)
    print(f"图像将保存至: {image_save_dir}")
    print("操作提示:")
    print("  - 按 'c' 保存当前帧并输出坐标")
    print("  - 按 'q' 退出程序")
    print("  - 检测到的AprilTag会显示ID和中心坐标")

    # 创建对齐对象（将深度帧对齐到彩色帧）
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    saved_count = 0
    
    try:
        while True:
            # 获取帧并对齐
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                continue

            # 转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            
            # 转换为灰度图用于检测
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            
            # 检测AprilTags (apriltag库的接口)
            tags = at_detector.detect(gray_image)
            
            # 处理每个检测到的标签
            display_image = color_image.copy()
            tag_coordinates = {}  # 存储标签ID和对应的中心坐标
            
            for tag in tags:
                # 绘制标签信息并获取中心坐标
                display_image, center = draw_tag_info(display_image, tag)
                tag_coordinates[tag.tag_id] = center
            
            # 显示检测到的标签数量
            info_text = f"检测到 {len(tags)} 个AprilTag | 已保存 {saved_count} 张图像"
            cv2.putText(display_image, info_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            # 显示图像
            cv2.imshow('AprilTag Detection', display_image)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("退出程序...")
                break
            elif key == ord('c'):
                saved_count += 1
                img_path = os.path.join(image_save_dir, f"tag_detect_{saved_count}.png")
                cv2.imwrite(img_path, display_image)
                
                # 输出每个标签的二维坐标
                print(f"\n已保存图像: {img_path}")
                print("标签ID及中心坐标 (x, y):")
                for tag_id, (x, y) in tag_coordinates.items():
                    print(f"  ID: {tag_id} - 坐标: ({x}, {y})")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
