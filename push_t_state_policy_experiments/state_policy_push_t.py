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

from huggingface_hub.utils import IGNORE_GIT_FOLDER_PATTERNS
#@markdown ### **Env Demo**
#@markdown Standard Gym Env (0.21.0 API)

# Local code imports
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

    return noise_pred_net

def load_pretrained_weights(noise_pred_net, ckpt_path="pusht_state_100ep.ckpt"):

    # Download pretrained weights from Google Drive and put it in ckpt_path. Link: https://drive.google.com/uc?id=1mHDr_DEZSdiGo9yecL50BBQYzR8Fjhl_&confirm=t

    if not os.path.isfile(ckpt_path):
        id = "1mHDr_DEZSdiGo9yecL50BBQYzR8Fjhl_&confirm=t"
        gdown.download(id=id, output=ckpt_path, quiet=False)

    state_dict = torch.load(ckpt_path, map_location='cuda')
    ema_noise_pred_net = noise_pred_net
    ema_noise_pred_net.load_state_dict(state_dict)
    print('Pretrained weights loaded.')

    return ema_noise_pred_net

if __name__ == '__main__':

    # Run environment demo
    print("Running environment demo...")
    environment_demo()

    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    
    # Run dataset demo
    print("Running dataset demo...")
    dataset_demo(pred_horizon, obs_horizon, action_horizon)

    # Run network demo
    print("Running network demo...")
    noise_pred_net = network_demo(pred_horizon, obs_horizon)

    # Load pretrained weights
    print("Loading pretrained weights...")
    ema_noise_pred_net = load_pretrained_weights(noise_pred_net)