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

def dataset_demo():
    # download demonstration data from Google Drive
    dataset_path = os.path.join(REPO_DIR, "pusht_cchi_v7_replay")
    
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

if __name__ == '__main__':

    # Run environment demo
    print("Running environment demo...")
    environment_demo()

    # Run dataset demo
    print("Running dataset demo...")
    dataset_demo()