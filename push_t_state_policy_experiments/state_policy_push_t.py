### This script is the set of dmeos for push t state policy, and uses pretrained weights ###

# diffusion policy import.
from typing import Tuple, Sequence, Dict, Union, Optional
import numpy as np
import math
import torch
import torch.nn as nn
import collections
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

# env import
import gym
from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
from skvideo.io import vwrite
from IPython.display import Video
import gdown
import os
import sys

from huggingface_hub.utils import IGNORE_GIT_FOLDER_PATTERNS
#@markdown ### **Env Demo**
#@markdown Standard Gym Env (0.21.0 API)

# Local code imports
STATE_POLICY_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(STATE_POLICY_EXAMPLE_DIR)
UTILS_DIR = os.path.join(REPO_DIR, "state_policy_utils")
if os.path.exists(UTILS_DIR):
    sys.path.append(UTILS_DIR)

from push_t_state_env import *
from push_t_state_dataset import *
from push_t_state_network import *

def environment_demo():

    # 0. create env object
    env = PushTEnv()

    # 1. seed env for initial state.
    # Seed 0-200 are used for the demonstration dataset.
    env.seed(1000)

    # 2. must reset before use
    obs, IGNORE_GIT_FOLDER_PATTERNS = env.reset()

    # 3. 2D positional action space [0,512]
    action = env.action_space.sample()

    # 4. Standard gym step method
    obs, reward, terminated, truncated, info = env.step(action)

    # prints and explains each dimension of the observation and action vectors
    with np.printoptions(precision=4, suppress=True, threshold=5):
        print("Obs: ", repr(obs))
        print("Obs:        [agent_x,  agent_y,  block_x,  block_y,    block_angle]")
        print("Action: ", repr(action))
        print("Action:   [target_agent_x, target_agent_y]")

def dataset_demo(pred_horizon, obs_horizon, action_horizon, dataset_path="pusht_cchi_v7_replay"):
    #@markdown ### **Dataset Demo**

    # download demonstration data from Google Drive, unzip, and put it in the dataset_path. Link is https://drive.google.com/uc?id=1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq&confirm=t 

    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )
    # save training data statistics (min, max) for each dim
    stats = dataset.stats

    # create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        num_workers=1,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    # visualize data in batch
    batch = next(iter(dataloader))
    print("batch['obs'].shape:", batch['obs'].shape)
    print("batch['action'].shape", batch['action'].shape)

    return stats

def network_demo(pred_horizon, obs_horizon):
    #@markdown ### **Network Demo**

    # observation and action dimensions corrsponding to
    # the output of PushTEnv
    obs_dim = 5
    action_dim = 2

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    # example inputs
    noised_action = torch.randn((1, pred_horizon, action_dim))
    obs = torch.zeros((1, obs_horizon, obs_dim))
    diffusion_iter = torch.zeros((1,))

    # the noise prediction network
    # takes noisy action, diffusion iteration and observation as input
    # predicts the noise added to action
    noise = noise_pred_net(
        sample=noised_action,
        timestep=diffusion_iter,
        global_cond=obs.flatten(start_dim=1))

    # illustration of removing noise
    # the actual noise removal is performed by NoiseScheduler
    # and is dependent on the diffusion noise schedule
    denoised_action = noised_action - noise

    # for this demo, we use DDPMScheduler with 100 diffusion iterations
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule='squaredcos_cap_v2',
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )

    # device transfer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _ = noise_pred_net.to(device)

    return noise_pred_net, noise_scheduler, num_diffusion_iters, action_dim, device

def load_pretrained_weights(noise_pred_net, ckpt_path="pusht_state_100ep.ckpt"):

    # Download pretrained weights from Google Drive and put it in ckpt_path. Link: https://drive.google.com/uc?id=1mHDr_DEZSdiGo9yecL50BBQYzR8Fjhl_&confirm=t

    state_dict = torch.load(ckpt_path, map_location='cuda')
    ema_noise_pred_net = noise_pred_net
    ema_noise_pred_net.load_state_dict(state_dict)
    print('Pretrained weights loaded.')

    return ema_noise_pred_net

def run_inference(ema_noise_pred_net, noise_scheduler, stats, num_diffusion_iters, action_dim, device):
    #@markdown ### **Inference**

    # limit enviornment interaction to 200 steps before termination
    max_steps = 200
    env = PushTEnv()
    # use a seed >200 to avoid initial states seen in the training dataset
    env.seed(100000)

    # get first observation
    obs, info = env.reset()

    # keep a queue of last 2 steps of observations
    obs_deque = collections.deque(
        [obs] * obs_horizon, maxlen=obs_horizon)
    # save visualization and rewards
    imgs = [env.render(mode='rgb_array')]
    rewards = list()
    done = False
    step_idx = 0

    with tqdm(total=max_steps, desc="Eval PushTStateEnv") as pbar:
        while not done:
            B = 1
            # stack the last obs_horizon (2) number of observations
            obs_seq = np.stack(obs_deque)
            # normalize observation
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            # device transfer
            nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)

            # infer action
            with torch.no_grad():
                # reshape observation to (B,obs_horizon*obs_dim)
                obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

                # initialize action from Guassian noise
                noisy_action = torch.randn(
                    (B, pred_horizon, action_dim), device=device)
                naction = noisy_action

                # init scheduler
                noise_scheduler.set_timesteps(num_diffusion_iters)

                for k in noise_scheduler.timesteps:
                    # predict noise
                    noise_pred = ema_noise_pred_net(
                        sample=naction,
                        timestep=k,
                        global_cond=obs_cond
                    )

                    # inverse diffusion step (remove noise)
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample

            # unnormalize action
            naction = naction.detach().to('cpu').numpy()
            # (B, pred_horizon, action_dim)
            naction = naction[0]
            action_pred = unnormalize_data(naction, stats=stats['action'])

            # only take action_horizon number of actions
            start = obs_horizon - 1
            end = start + action_horizon
            action = action_pred[start:end,:]
            # (action_horizon, action_dim)

            # execute action_horizon number of steps
            # without replanning
            for i in range(len(action)):
                # stepping env
                obs, reward, done, _, info = env.step(action[i])
                # save observations
                obs_deque.append(obs)
                # and reward/vis
                rewards.append(reward)
                imgs.append(env.render(mode='rgb_array'))

                # update progress bar
                step_idx += 1
                pbar.update(1)
                pbar.set_postfix(reward=reward)
                if step_idx > max_steps:
                    done = True
                if done:
                    break

    # print out the maximum target coverage
    print('Score: ', max(rewards))

    # visualize
    vwrite('vis.mp4', imgs)

if __name__ == '__main__':

    # Run environment demo
    print("Running environment demo...")
    environment_demo()

    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    
    # Run dataset demo
    print("Running dataset demo...")
    dataset_path = os.path.join(REPO_DIR, "pusht_cchi_v7_replay")
    stats = dataset_demo(pred_horizon, obs_horizon, action_horizon, dataset_path=dataset_path)

    # Run network demo
    print("Running network demo...")
    noise_pred_net, noise_scheduler, num_diffusion_iters, action_dim, device = network_demo(pred_horizon, obs_horizon)

    # Load pretrained weights
    print("Loading pretrained weights...")
    ckpt_path = os.path.join(STATE_POLICY_EXAMPLE_DIR, "pusht_state_100ep.ckpt")
    ema_noise_pred_net = load_pretrained_weights(noise_pred_net, ckpt_path=ckpt_path)

    # Run inference
    print("Running inference...")
    run_inference(ema_noise_pred_net, noise_scheduler, stats, num_diffusion_iters, action_dim, device)