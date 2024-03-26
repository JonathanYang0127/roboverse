import matplotlib.pyplot as plt
import roboverse
from perspectives import perspectives

import argparse
import os
import pickle
import torch
import numpy as np
from PIL import Image
from datetime import datetime
import yaml
import pickle as pkl

from omnimimic.policies import *
from omnimimic.data.data_utils import unnormalize_action
import omnimimic.torch.pytorch_util as ptu
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import torch.nn as nn


def process_image(image, downsample=False):
    ''' ObsDictReplayBuffer wants flattened (channel, height, width) images float32'''
    if image.dtype == np.uint8:
        image =  image.astype(np.float32) / 255.0
    if len(image.shape) == 3 and image.shape[0] != 3 and image.shape[2] == 3:
        image = np.transpose(image, (2, 0, 1))
    if downsample:
        image = image[:,::2, ::2]
    return image.flatten()

def process_obs(obs, task, use_robot_state, prev_obs=None, downsample=False, image_idx=0):
    if use_robot_state:
        observation_keys = ['image', 'desired_pose', 'current_pose', 'task_embedding']
    else:
        observation_keys = ['image', 'task_embedding']

    if prev_obs:
        observation_keys = ['previous_image'] + observation_keys

    if task is None:
        observation_keys = observation_keys[:-1]

    obs['image'] = process_image(obs['images'][image_idx]['array'], downsample=downsample)
    if prev_obs is not None:
        obs['previous_image'] = process_image(prev_obs['images'][image_idx]['array'], downsample=downsample)
    obs['task_embedding'] = task
    return ptu.from_numpy(np.concatenate([obs[k] for k in observation_keys]))

def stack_obs(context):
    return np.concatenate(context, axis=-1)

def process_action(action):
    return np.clip(action, -1, 1)

def rollout_policy(args, env, action_metadata, policy_config, eval_policy, goal_image, num_trajs=100):
    distance_preds = []
    num_success = 0
    for i in range(num_trajs):
        env.reset()
        obs = env.get_observation()
        images = []
        context_size = policy_config['context_size']
        if context_size != 0: 
            context_state = [obs['state'] for _ in range(context_size + 1)]
            context = [obs['image'].reshape(3, 128, 128).transpose(1, 2, 0) for _ in range(context_size + 1)]

        if args.data_save_dir is not None:
            trajectory = []

        success = False
        for j in range(args.num_timesteps):
            obs = env.get_observation()
            state = obs['state']
            img = obs['image'].reshape(3, 128, 128).transpose(1, 2, 0)
            if goal_image is None:
                goal_image = img
            if context_size != 0:
                context.pop(0) 
                context =  context + [img]
                img = stack_obs(context)
                context_state.pop(0)
                context_state =  context_state + [state]
                state = np.array(context_state)
            action = eval_policy.get_action(img[None], goal_image[None], state=state[None], task_embedding=None)[0]
            action = unnormalize_action(action, action_metadata, discrete=policy_config['discrete'],
                num_bins=policy_config['num_bins']) 
            action = process_action(action)[0]
            next_observation, reward, done, info = env.step(action)

            if args.video_save_dir:
                image = obs['image'].reshape(3, 128, 128).transpose(1, 2, 0)
                image = (image * 255.0).astype(np.uint8)
                images.append(Image.fromarray(image))
       
            success = success or info['place_success']

        print(success, num_success)
        num_success += int(success)
 
        #Save Trajectory
        if args.data_save_dir is not None:
            now = datetime.now()
            dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")

            with open(os.path.join(args.data_save_dir, 'traj_{}.pkl'.format(dt_string)), 'wb+') as f:
                pickle.dump(trajectory, f)

        # Save Video
        if args.video_save_dir:
            print("Saving Video")
            images[0].save('{}/eval_{}.gif'.format(args.video_save_dir, i),
                            format='GIF', append_images=images[1:],
                            save_all=True, duration=200, loop=0)
    return num_success, num_trajs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, required=True)
    parser.add_argument("-v", "--video-save-dir", type=str, default="")
    parser.add_argument("-d", "--data-save-dir", type=str, default=None)
    parser.add_argument("-n", "--num-timesteps", type=int, default=15)
    parser.add_argument("--q-value-eval", default=False, action='store_true')
    parser.add_argument("--num-tasks", type=int, default=0)
    parser.add_argument("--task-embedding", default=False, action="store_true")
    parser.add_argument("--task-encoder", default=None)
    parser.add_argument("--sample-trajectory", type=str, default=None)
    parser.add_argument("--use-checkpoint-encoder", action='store_true', default=False)
    parser.add_argument("--use-robot-state", action='store_true', default=False)
    parser.add_argument("--action-relabelling", type=str, choices=("achieved, cdp"))
    parser.add_argument("--normalize-relabelling", action="store_true", default=False)
    parser.add_argument("--robot-model", type=str, choices=('wx250s', 'franka'), default='wx250s')
    parser.add_argument("--downsample-image", action='store_true', default=False)
    parser.add_argument("--multi-head-idx", type=int, default=-1)
    parser.add_argument("--blocking", action="store_true", default=False)
    parser.add_argument("--image-idx", type=int, default=2)
    parser.add_argument("--policy-class", type=str, required=True)
    parser.add_argument("--policy-config", type=str, required=True)
    parser.add_argument('--goal_image', type=str, default=None)
    parser.add_argument('--normalization-path', type=str, 
        default='/iris/u/jyang27/rlds_data/roboverse_dataset/1.0.1/obs_action_stats_roboverse_dataset.pkl')
    args = parser.parse_args()

    ptu.set_gpu_mode(True)
    torch.cuda.empty_cache()

    if not os.path.exists(args.video_save_dir) and args.video_save_dir:
        os.mkdir(args.video_save_dir)

    camera_params = perspectives[0]
    camera_params['object_names'] = perspectives[1]['object_names']
    camera_params['object_scales'] = perspectives[1]['object_scales']
    camera_params['target_object'] = perspectives[1]['target_object']
    env = roboverse.make('Widow250PickPlace-v0', gui=False, 
        observation_img_dim=128, **camera_params)
    env.reset()


    with open(args.policy_config, 'rb') as f:
        policy_config = yaml.safe_load(f) 
    policy_config['num_bins'] = policy_config.get('num_bins', 0)
    policy_config['discrete'] = policy_config.get('discrete', False)
    # policy = make_policy(args.policy_class, policy_config)

    config = policy_config
    model, noise_scheduler = make_policy(config["model_type"], config) 
    model = load_model(model, args.checkpoint_path, config, 'cuda')
    model = model.cuda()
    policy = model.eval()
    num_diffusion_iters = config["num_diffusion_iters"]
    wrapper = get_numpy_wrapper('nomad')
    policy = wrapper(policy, noise_scheduler, policy_config)

    if args.goal_image is None:
        goal_image = None
    else:
        goal_image = Image.open(args.goal_image)
        goal_image = np.asarray(goal_image, dtype=np.uint8)
        import matplotlib.pyplot as plt
        print("Showing Goal Image")
        fig = plt.figure()
        plt.imshow(goal_image)
        plt.savefig('out.png')

    #Get Action Metadata
    with open(args.normalization_path, 'rb') as f:
        action_metadata = pickle.load(f)

    eval_policy = policy
    num_success, num_trajs  = rollout_policy(args, env, action_metadata, policy_config, eval_policy, goal_image, num_trajs=10)
    print(num_success)

