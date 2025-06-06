import os
import json
import numpy as np
import torch
from tqdm import tqdm
import collections
import multiprocessing as mp
import sys

VAR_AWARE_SCALING_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.dirname(VAR_AWARE_SCALING_UTILS_DIR)
STATE_POLICY_UTILS_DIR = os.path.join(HOME_DIR, "state_policy_utils")

sys.path.append(STATE_POLICY_UTILS_DIR)

from push_t_state_env import PushTEnv
from push_t_state_dataset import PushTStateDataset, normalize_data, unnormalize_data
from push_t_state_network import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

def find_nonzero_eval_seeds(single_model_eval_results_json_path):
    """
    Find the non-zero eval seeds in single_model_eval_results_path
    """
    with open(single_model_eval_results_json_path, "r") as f:
        single_model_eval_results_dict = json.load(f)

    solved_seeds = single_model_eval_results_dict['solved_seeds']    

    return solved_seeds