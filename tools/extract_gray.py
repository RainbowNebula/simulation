from decord import VideoReader
import cv2

rgb_reader = VideoReader("/home/haichao/workspace/real2sim/simulation/data/20250716-2018/video/rgb_video_338122303378_1752668333.mp4")
ir_reader = VideoReader("/home/haichao/workspace/real2sim/simulation/data/20250716-2018/video/ir_left_video_338122303378_1752668333.mp4")

ir_frame0 = ir_reader[0].asnumpy()
rgb_frame0 = rgb_reader[0].asnumpy()

rgb_frame0 = cv2.cvtColor(rgb_frame0, cv2.COLOR_BGR2GRAY)
cv2.imwrite("frame0_gray.png", rgb_frame0)
cv2.imwrite("frame0_ir.png", ir_frame0)