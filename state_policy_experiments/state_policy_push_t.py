# diffusion policy import
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
from push_t_env import *

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

if __name__ == '__main__':

    # Run environment demo
    print("Running environment demo...")
    environment_demo()