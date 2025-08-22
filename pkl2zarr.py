import pickle, os
import numpy as np
import pdb
from copy import deepcopy
import zarr
import shutil
import argparse
import einops
import cv2


def main():
    parser = argparse.ArgumentParser(description='Process some episodes.')
    parser.add_argument('-task_name', required=False, type=str, default='grasp_cube',
                        help='The name of the task (e.g., block_hammer_beat)')
    parser.add_argument('-head_camera_type', type=str, required=False,default="D455")
    parser.add_argument('-expert_data_num', type=int, required=False,default=0,
                        help='Number of episodes to process (e.g., 50)')
    parser.add_argument('-data_dir', type=str, required=False,default='/mnt/nas/liuqipeng/workspace/simulation/sim_data/grasp_cube_h100/grasp_cube')
    args = parser.parse_args()

    task_name = args.task_name
    num = args.expert_data_num
    head_camera_type = args.head_camera_type
    # load_dir = f'{args.data_dir}/{task_name}_{head_camera_type}_pkl'
    
    if num == 0:
        num = len(os.listdir(args.data_dir))

    load_dir = args.data_dir
    
    total_count = 0

    save_dir = f'./policy/Diffusion-Policy/data/{task_name}_{head_camera_type}_{num}_2.zarr'

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    current_ep = 0

    zarr_root = zarr.group(save_dir)
    zarr_data = zarr_root.create_group('data')
    zarr_meta = zarr_root.create_group('meta')

    head_camera_arrays, front_camera_arrays, left_camera_arrays, right_camera_arrays = [], [], [], []
    episode_ends_arrays, action_arrays, state_arrays, joint_action_arrays = [], [], [], []
    front_camera_arrays = []
    
    cnt = 0
    while current_ep < num:
        if not os.path.isdir(load_dir+f'/episode{current_ep}'):
            current_ep += 1
            continue
        print(f'processing episode: {current_ep + 1} / {num}', end='\r')
        file_num = 0
        
        while os.path.exists(load_dir+f'/episode{current_ep}'+f'/pkl/{file_num}.pkl'):
            
            # 包含plan2start的情况 
            # if cnt != 0 or file_num < 100:
            #     cnt += 1
            #     continue
            # if file_num % 2 != 0:
            #     continue

            with open(load_dir+f'/episode{current_ep}'+f'/pkl/{file_num}.pkl', 'rb') as file:
                data = pickle.load(file)
            
            front_img = data['observation']['front_camera']['rgb']
            head_img = data['observation']['head_camera']['rgb']
            action = data['endpose']['endpose']
            joint_action = data['joint_action']
            
            front_camera_arrays.append(front_img)
            head_camera_arrays.append(head_img)
            action_arrays.append(action)
            state_arrays.append(joint_action)
            joint_action_arrays.append(joint_action)

            file_num += 1
            total_count += 1
            
        current_ep += 1

        episode_ends_arrays.append(total_count)

    # print()
    episode_ends_arrays = np.array(episode_ends_arrays)
    action_arrays = np.array(action_arrays)
    state_arrays = np.array(state_arrays)
    head_camera_arrays = np.array(head_camera_arrays)
    joint_action_arrays = np.array(joint_action_arrays)
    front_camera_arrays = np.array(front_camera_arrays)

    # import ipdb
    # ipdb.set_trace()
    head_camera_arrays = np.moveaxis(head_camera_arrays, -1, 1)  # NHWC -> NCHW

    compressor = zarr.Blosc(cname='zstd', clevel=3, shuffle=1)
    action_chunk_size = (100, action_arrays.shape[1])
    state_chunk_size = (100, state_arrays.shape[1])
    joint_chunk_size = (100, joint_action_arrays.shape[1])
    head_camera_chunk_size = (100, *head_camera_arrays.shape[1:])

    zarr_data.create_dataset('front_camera', data=front_camera_arrays, chunks=head_camera_chunk_size, overwrite=True, compressor=compressor)
    zarr_data.create_dataset('head_camera', data=head_camera_arrays, chunks=head_camera_chunk_size, overwrite=True, compressor=compressor)
    zarr_data.create_dataset('tcp_action', data=action_arrays, chunks=action_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('state', data=state_arrays, chunks=state_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_data.create_dataset('action', data=joint_action_arrays, chunks=joint_chunk_size, dtype='float32', overwrite=True, compressor=compressor)
    zarr_meta.create_dataset('episode_ends', data=episode_ends_arrays, dtype='int64', overwrite=True, compressor=compressor)

if __name__ == '__main__':
    main()
