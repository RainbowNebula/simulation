import sys
sys.path.append('./') 
sys.path.append('/mnt/nas/liuqipeng/workspace/RoboTwinOld/policy/Diffusion-Policy') 

import torch  
import os
import numpy as np
import hydra
from pathlib import Path
from collections import deque
import traceback

import yaml
from datetime import datetime
import importlib
import dill
from argparse import ArgumentParser
from diffusion_policy.workspace.robotworkspace import RobotWorkspace
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.env_runner.dp_runner import DPRunner, DPWrapper

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)
poilcy_dir = '/mnt/nas/liuqipeng/workspace/RoboTwinOld/policy/Diffusion-Policy'

def get_policy(checkpoint, output_dir, device):
    
    # load checkpoint
    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)
    cfg = payload['cfg']
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: RobotWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model
    
    device = torch.device(device)
    policy.to(device)
    policy.eval()

    return policy

class DP:
    def __init__(self, ckpt_path: str, seed: int):
         self.policy = get_policy(ckpt_path, None, 'cuda:0')
         self.runner = DPWrapper(output_dir=None)
         print("DP init success")

    # def __init__(self, task_name, head_camera_type: str, checkpoint_num: int, expert_data_num: int, seed: int):
    #     self.policy = get_policy(f'checkpoints/{task_name}_{head_camera_type}_{expert_data_num}_{seed}/{checkpoint_num}.ckpt', None, 'cuda:0')

    def update_obs(self, observation):
        self.runner.update_obs(observation)
    
    def get_action(self, observation=None):
        action = self.runner.get_action(self.policy, observation)
        return action

    def get_last_obs(self):
        return self.runner.obs[-1]

def load_env_class(task_name):
    envs_module = importlib.import_module(f'{task_name}') # envs.{task_name}
    # try:
    #  检查模块中是否存在对应的类
    if hasattr(envs_module, task_name):
        # 获取类并返回
        env_class = getattr(envs_module, task_name)
        return env_class
    else:
        raise AttributeError(f"在模块 {task_name}.py 中未找到 {task_name} 类")
    env_class = getattr(envs_module, task_name)
    env_instance = env_class()
    # except:
    #     raise SystemExit("No Task")
    return env_instance

def test_policy(EnvClass, args, dp: DP, test_num=20):
    expert_check = False
    print("Task name: ", args["task_name"])

    # import ipdb
    # ipdb.set_trace()

    Env = EnvClass(**args)
    Env.suc = 0
    Env.test_num =0

    print("开始测试")
    for i in range(test_num):
        dp.runner.reset_obs()
        args['seed'] += 1
        Env.apply_dp(dp, args)

    # now_id = 0
    # succ_seed = 0
    # suc_test_seed_list = []

    # 保存验证视频
    # args['save_eval_video'] = True
    
    # now_seed = st_seed
    # while succ_seed < test_num:
    #     render_freq = args['render_freq']
    #     args['render_freq'] = 0
        
        # if expert_check:
        #     try:
        #         EnvClass.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** args)
        #         EnvClass.play_once()
        #         EnvClass.close()
        #     except Exception as e:
        #         stack_trace = traceback.format_exc()
        #         print(' -------------')
        #         print('Error: ', stack_trace)
        #         print(' -------------')
        #         EnvClass.close()
        #         now_seed += 1
        #         args['render_freq'] = render_freq
        #         print('error occurs !')
        #         continue

        # if (not expert_check) or EnvClass.check_success():
        #     succ_seed += 1
        #     suc_test_seed_list.append(now_seed)
        # else:
        #     now_seed += 1
        #     args['render_freq'] = render_freq
        #     continue


        # args['render_freq'] = render_freq

        # EnvClass.setup_demo(now_ep_num=now_id, seed = now_seed, is_test = True, ** args)
        # EnvClass.apply_dp(dp, args)

        # now_id += 1
        # EnvClass.close()
        # if EnvClass.render_freq:
        #     EnvClass.viewer.close()
        # dp.runner.reset_obs()
        # print(f"{task_name} success rate: {EnvClass.suc}/{EnvClass.test_num}, current seed: {now_seed}\n")

        # EnvClass._take_picture()
        # now_seed += 1

    # return EnvClass.suc

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, '../task_config/camera_config.yaml')

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, 'r', encoding='utf-8') as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f'camera {camera_type} is not defined'
    return args[camera_type]

def main(usr_args):
    task_name = usr_args.task_name
    with open(f'{usr_args.base_dir}/task_config/{task_name}.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    
    seed = cfg['seed']
    # args['head_camera_type'] = head_camera_type 
    # head_camera_config = get_camera_config(args['head_camera_type'])
    # args['head_camera_fovy'] = head_camera_config['fovy']
    # args['head_camera_w'] = head_camera_config['w']
    # args['head_camera_h'] = head_camera_config['h']
    # head_camera_config = 'fovy' + str(args['head_camera_fovy']) + '_w' + str(args['head_camera_w']) + '_h' + str(args['head_camera_h'])
    
    # wrist_camera_config = get_camera_config(args['wrist_camera_type'])
    # args['wrist_camera_fovy'] = wrist_camera_config['fovy']
    # args['wrist_camera_w'] = wrist_camera_config['w']
    # args['wrist_camera_h'] = wrist_camera_config['h']
    # wrist_camera_config = 'fovy' + str(args['wrist_camera_fovy']) + '_w' + str(args['wrist_camera_w']) + '_h' + str(args['wrist_camera_h'])

    # front_camera_config = get_camera_config(args['front_camera_type'])
    # args['front_camera_fovy'] = front_camera_config['fovy']
    # args['front_camera_w'] = front_camera_config['w']
    # args['front_camera_h'] = front_camera_config['h']
    # front_camera_config = 'fovy' + str(args['front_camera_fovy']) + '_w' + str(args['front_camera_w']) + '_h' + str(args['front_camera_h'])

    # # output camera config
    # print('============= Camera Config =============\n')
    # print('Head Camera Config:\n    type: '+ str(args['head_camera_type']) + '\n    fovy: ' + str(args['head_camera_fovy']) + '\n    camera_w: ' + str(args['head_camera_w']) + '\n    camera_h: ' + str(args['head_camera_h']))
    # print('Wrist Camera Config:\n    type: '+ str(args['wrist_camera_type']) + '\n    fovy: ' + str(args['wrist_camera_fovy']) + '\n    camera_w: ' + str(args['wrist_camera_w']) + '\n    camera_h: ' + str(args['wrist_camera_h']))
    # print('Front Camera Config:\n    type: '+ str(args['front_camera_type']) + '\n    fovy: ' + str(args['front_camera_fovy']) + '\n    camera_w: ' + str(args['front_camera_w']) + '\n    camera_h: ' + str(args['front_camera_h']))
    # print('\n=======================================')

    cfg['expert_seed'] = cfg['seed']
    cfg['expert_data_num'] = 19

    task_env = load_env_class(cfg['task_name'])

    st_seed = 100000 * (1+cfg["seed"])
    suc_nums = []
    test_num = 100 
    topk = 1

    # dp = DP(task_name,  usr_args.expert_data_num, seed)

    dp = DP(usr_args.ckpt_path, seed)

    # import ipdb
    # ipdb.set_trace()

    print(task_name)

    suc_num = test_policy(task_env, cfg, dp, test_num=test_num)
    suc_nums.append(suc_num)

    print(f"success rate: {task_env.suc}/{test_num}")

    # topk_success_rate = sorted(suc_nums, reverse=True)[:topk]
    # # save_dir = Path(f'eval_result/dp/{task_name}_{usr_args.head_camera_type}/{usr_args.expert_data_num}')
    # save_dir = Path(f'eval_result/dp/{task_name}/')
    # save_dir.mkdir(parents=True, exist_ok=True)
    # file_path = save_dir / f'ckpt_{checkpoint_num}_seed_{seed}.txt'
    # current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # with open(file_path, 'w') as file:
    #     file.write(f'Timestamp: {current_time}\n\n')

    #     file.write(f'Checkpoint Num: {checkpoint_num}\n')
        
    #     file.write('Successful Rate of Diffenent checkpoints:\n')
    #     file.write('\n'.join(map(str, np.array(suc_nums) / test_num)))
    #     file.write('\n\n')
    #     file.write(f'TopK {topk} Success Rate (every):\n')
    #     file.write('\n'.join(map(str, np.array(topk_success_rate) / test_num)))
    #     file.write('\n\n')
    #     file.write(f'TopK {topk} Success Rate:\n')
    #     file.write(f'\n'.join(map(str, np.array(topk_success_rate) / (topk * test_num))))
    #     file.write('\n\n')

    # print(f'Data has been saved to {file_path}')



if __name__ == "__main__":

    # print("start test")
    
    parser = ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='/mnt/nas/liuqipeng/workspace/simulation/')
    parser.add_argument('--task_name', type=str, default='grasp_cube2')
    parser.add_argument("--ckpt_path", type=str, default="/mnt/nas/liuqipeng/workspace/RoboTwinOld/policy/Diffusion-Policy/checkpoints/grasp_cube_D455_104_action_0/300.ckpt")
    # parser.add_argument('head_camera_type', type=str)
    # parser.add_argument('expert_data_num', type=int, default=20)
    # parser.add_argument('checkpoint_num', type=int, default=1000)
    # parser.add_argument('--seed', type=int, default=0)
    usr_args = parser.parse_args()
    
    main(usr_args)
