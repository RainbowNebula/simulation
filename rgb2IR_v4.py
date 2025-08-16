import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser(description='基于特征点匹配的IR与RGB对齐')
   parser.add_argument('--rgb_path', required=False, default="/home/haichao/workspace/real2sim/simulation/data/20250716-2018/video/rgb_video_338122303378_1752668333.mp4",help='RGB视频文件路径')
    parser.add_argument('--left_ir_path', required=False,default="/home/haichao/workspace/real2sim/simulation/data/20250716-2018/video/ir_left_video_338122303378_1752668333.mp4",help='左IR视频文件路径')
    parser.add_argument('--right_ir_path', required=False, default="/home/haichao/workspace/real2sim/simulation/data/20250716-2018/video/ir_right_video_338122303378_1752668333.mp4",help='右IR视频文件路径')
    parser.add_argument('--output_dir', default='output', help='输出目录')
    parser.add_argument('--is_image', action='store_true', help='输入为图像而非视频')
    parser.add_argument('--save_matches', action='store_true', help='保存特征匹配可视化结果')
    return parser.parse_args()

def preprocess_images(rgb_img, ir_img):
    """预处理：统一灰度空间并增强特征"""
    # RGB转灰度
    rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    
    # 确保IR为单通道
    if len(ir_img.shape) == 3:
        ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
    else:
        ir_gray = ir_img
    
    # 对比度增强
    rgb_gray = cv2.equalizeHist(rgb_gray)
    ir_gray = cv2.equalizeHist(ir_gray)
    
    # 去噪
    rgb_gray = cv2.GaussianBlur(rgb_gray, (3, 3), 0)
    ir_gray = cv2.GaussianBlur(ir_gray, (3, 3), 0)
    
    return rgb_gray, ir_gray, rgb_img

def feature_matching(rgb_gray, ir_gray, rgb_img, ir_img, save_path=None):
    """特征点匹配并计算单应矩阵"""
    # 初始化ORB特征检测器
    orb = cv2.ORB_create(
        nfeatures=1500,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=31,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20
    )
    
    # 检测特征点并计算描述符
    kp_rgb, des_rgb = orb.detectAndCompute(rgb_gray, None)
    kp_ir, des_ir = orb.detectAndCompute(ir_gray, None)
    
    # 双向匹配策略
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches1 = matcher.knnMatch(des_rgb, des_ir, k=2)
    matches2 = matcher.knnMatch(des_ir, des_rgb, k=2)
    
    # 筛选优质匹配（Lowe's ratio test）
    good1 = []
    for m, n in matches1:
        if m.distance < 0.75 * n.distance:
            good1.append(m)
    
    good2 = []
    for m, n in matches2:
        if m.distance < 0.75 * n.distance:
            good2.append(m)
    
    # 双向验证匹配点
    good_matches = []
    for m in good1:
        for n in good2:
            if m.queryIdx == n.trainIdx and m.trainIdx == n.queryIdx:
                good_matches.append(m)
                break
    
    if len(good_matches) < 10:
        return None  # 匹配点不足，无法计算单应矩阵
    
    # 计算单应矩阵（RANSAC抗噪）
    src_pts = np.float32([kp_rgb[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ir[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # 保存匹配可视化结果
    if save_path:
        matched_img = cv2.drawMatches(
            rgb_img, kp_rgb, ir_img, kp_ir,
            good_matches, None,
            matchColor=(0, 255, 0),
            singlePointColor=None,
            matchesMask=mask.ravel().tolist(),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(save_path, matched_img)
    
    return M

def align_rgb_to_ir(rgb_img, ir_img, M):
    """使用单应矩阵将RGB对齐到IR视角"""
    h, w = ir_img.shape[:2]
    # 应用透视变换
    rgb_aligned = cv2.warpPerspective(rgb_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # 生成IR掩码（保留有效区域）
    if len(ir_img.shape) == 3:
        ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
    else:
        ir_gray = ir_img
    mask = ir_gray > 5  # 过滤暗区
    rgb_aligned[~mask] = 0  # 无效区域置黑
    
    return rgb_aligned

def process_single_image(rgb_path, ir_path, output_path, save_matches=False):
    """处理单张图像"""
    rgb_img = cv2.imread(rgb_path)
    ir_img = cv2.imread(ir_path)
    
    if rgb_img is None or ir_img is None:
        raise FileNotFoundError("无法读取图像文件")
    
    # 预处理
    rgb_gray, ir_gray, rgb_orig = preprocess_images(rgb_img, ir_img)
    
    # 特征匹配
    match_save_path = os.path.splitext(output_path)[0] + "_matches.jpg" if save_matches else None
    M = feature_matching(rgb_gray, ir_gray, rgb_orig, ir_img, match_save_path)
    
    if M is None:
        raise ValueError("特征匹配失败，无法计算对齐矩阵")
    
    # 对齐并保存结果
    aligned_rgb = align_rgb_to_ir(rgb_orig, ir_img, M)
    cv2.imwrite(output_path, aligned_rgb)
    print(f"已保存对齐结果: {output_path}")

def process_video(rgb_path, ir_path, output_path, save_matches=False):
    """处理视频序列"""
    rgb_cap = cv2.VideoCapture(rgb_path)
    ir_cap = cv2.VideoCapture(ir_path)
    
    fps = int(rgb_cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(rgb_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h = int(ir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(ir_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    # 初始化视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # 第一帧计算初始单应矩阵
    ret_rgb, rgb_first = rgb_cap.read()
    ret_ir, ir_first = ir_cap.read()
    if not ret_rgb or not ret_ir:
        raise IOError("无法读取视频帧")
    
    rgb_gray, ir_gray, _ = preprocess_images(rgb_first, ir_first)
    match_save_path = os.path.splitext(output_path)[0] + "_first_matches.jpg" if save_matches else None
    M = feature_matching(rgb_gray, ir_gray, rgb_first, ir_first, match_save_path)
    if M is None:
        raise ValueError("第一帧特征匹配失败，无法继续处理")
    
    # 处理第一帧
    aligned_first = align_rgb_to_ir(rgb_first, ir_first, M)
    out_writer.write(aligned_first)
    
    # 处理后续帧（复用初始单应矩阵，适合静态场景）
    progress = tqdm(total=frame_count, desc="处理视频")
    progress.update(1)
    
    while True:
        ret_rgb, rgb_frame = rgb_cap.read()
        ret_ir, ir_frame = ir_cap.read()
        
        if not ret_rgb or not ret_ir:
            break
        
        # 复用单应矩阵对齐
        aligned_frame = align_rgb_to_ir(rgb_frame, ir_frame, M)
        out_writer.write(aligned_frame)
        progress.update(1)
    
    progress.close()
    rgb_cap.release()
    ir_cap.release()
    out_writer.release()
    print(f"已保存对齐视频: {output_path}")

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理左IR
    left_output = os.path.join(args.output_dir, "left_ir_colorized.jpg" if args.is_image else "left_ir_colorized.mp4")
    # 处理右IR
    right_output = os.path.join(args.output_dir, "right_ir_colorized.jpg" if args.is_image else "right_ir_colorized.mp4")
    
    try:
        if args.is_image:
            process_single_image(args.rgb_path, args.left_ir_path, left_output, args.save_matches)
            process_single_image(args.rgb_path, args.right_ir_path, right_output, args.save_matches)
        else:
            process_video(args.rgb_path, args.left_ir_path, left_output, args.save_matches)
            process_video(args.rgb_path, args.right_ir_path, right_output, args.save_matches)
    except Exception as e:
        print(f"处理错误: {str(e)}")

if __name__ == "__main__":
    main()