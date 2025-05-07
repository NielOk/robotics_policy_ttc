### This script is the set of dmeos for push t vision policy, and uses pretrained weights ###

# diffusion policy import
from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
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

# Local code imports
from push_t_vision_env import *
from push_t_vision_dataset import *
from push_t_vision_network import *
from push_t_vision_encoder import *

VISION_POLICY_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(VISION_POLICY_EXAMPLE_DIR)

def environment_demo():
    # 0. create env object
    env = PushTImageEnv()

    # 1. seed env for initial state.
    # Seed 0-200 are used for the demonstration dataset.
    env.seed(1000)

    # 2. must reset before use
    obs, info = env.reset()

    # 3. 2D positional action space [0,512]
    action = env.action_space.sample()

    # 4. Standard gym step method
    obs, reward, terminated, truncated, info = env.step(action)

    # prints and explains each dimension of the observation and action vectors
    with np.printoptions(precision=4, suppress=True, threshold=5):
        print("obs['image'].shape:", obs['image'].shape, "float32, [0,1]")
        print("obs['agent_pos'].shape:", obs['agent_pos'].shape, "float32, [0,512]")
        print("action.shape: ", action.shape, "float32, [0,512]")

def dataset_demo(pred_horizon, obs_horizon, action_horizon, dataset_path):
    
    # parameters
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    #|o|o|                             observations: 2
    #| |a|a|a|a|a|a|a|a|               actions executed: 8
    #|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

    # create dataset from file
    dataset = PushTImageDataset(
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
        batch_size=64,
        num_workers=4,
        shuffle=True,
        # accelerate cpu-gpu transfer
        pin_memory=True,
        # don't kill worker process afte each epoch
        persistent_workers=True
    )

    # visualize data in batch
    batch = next(iter(dataloader))
    print("batch['image'].shape:", batch['image'].shape)
    print("batch['agent_pos'].shape:", batch['agent_pos'].shape)
    print("batch['action'].shape", batch['action'].shape)

    return stats

def network_demo(pred_horizon, obs_horizon):
    # construct ResNet18 encoder
    # if you have multiple camera views, use seperate encoder weights for each view.
    vision_encoder = get_resnet('resnet18')

    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    vision_encoder = replace_bn_with_gn(vision_encoder)

    # ResNet18 has output dim of 512
    vision_feature_dim = 512
    # agent_pos is 2 dimensional
    lowdim_obs_dim = 2
    # observation feature has 514 dims in total per step
    obs_dim = vision_feature_dim + lowdim_obs_dim
    action_dim = 2

    # create network object
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim*obs_horizon
    )

    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net
    })

    # demo
    with torch.no_grad():
        # example inputs
        image = torch.zeros((1, obs_horizon,3,96,96))
        agent_pos = torch.zeros((1, obs_horizon, 2))
        # vision encoder
        image_features = nets['vision_encoder'](
            image.flatten(end_dim=1))
        # (2,512)
        image_features = image_features.reshape(*image.shape[:2],-1)
        # (1,2,512)
        obs = torch.cat([image_features, agent_pos],dim=-1)
        # (1,2,514)

        noised_action = torch.randn((1, pred_horizon, action_dim))
        diffusion_iter = torch.zeros((1,))

        # the noise prediction network
        # takes noisy action, diffusion iteration and observation as input
        # predicts the noise added to action
        noise = nets['noise_pred_net'](
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
    device = torch.device('cuda')
    _ = nets.to(device)

    return noise_pred_net, noise_scheduler, num_diffusion_iters, action_dim, device

if __name__ == '__main__':

    # Run environment demo
    print("Running environment demo...")
    environment_demo()

    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8

    # Run dataset demo
    print("Running dataset demo...")
    # download demonstration data from Google Drive
    dataset_path = os.path.join(REPO_DIR, "pusht_cchi_v7_replay")
    stats = dataset_demo(pred_horizon, obs_horizon, action_horizon, dataset_path=dataset_path)

    # Run network demo
    print("Running network demo...")
    noise_pred_net, noise_scheduler, num_diffusion_iters, action_dim, device = network_demo(pred_horizon, obs_horizon)