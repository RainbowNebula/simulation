import cv2
import os
import argparse

def save_first_frame(video_path, output_filename=None):
    """
    读取视频的第一帧并保存到视频所在文件夹
    
    参数:
        video_path: 视频文件路径
        output_filename: 输出文件名（默认使用'视频名_第一帧.jpg'）
    """
    # 确保视频文件存在
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
    
    # 获取视频所在目录
    video_dir = os.path.dirname(video_path)
    video_name = os.path.basename(video_path)
    video_base, _ = os.path.splitext(video_name)
    
    # 设置默认输出文件名
    if output_filename is None:
        output_filename = f"{video_base}_第一帧.jpg"
    
    output_path = os.path.join(video_dir,"../images",output_filename)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    # 读取第一帧
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"无法读取视频的第一帧: {video_path}")
    
    # 保存第一帧
    cv2.imwrite(output_path, frame)
    print(f"成功保存第一帧到: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='读取视频第一帧并保存')
    parser.add_argument('--video_path', default="/home/haichao/workspace/real2sim/simulation/data/20250716-2018/video/rgb_video_338122303378_1752668333.mp4", required=False,help='视频文件路径')
    parser.add_argument('-o', '--output',default="0.png",required=False, help='输出文件名（可选）')
    args = parser.parse_args()
    
    try:
        save_first_frame(args.video_path, args.output)
    except Exception as e:
        print(f"错误: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()