import os
import json
import numpy as np
import torch
from tqdm import tqdm
import collections
import multiprocessing as mp
import sys

from variance_aware_scaling_utils import *

VAR_AWARE_SCALING_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
HOME_DIR = os.path.dirname(VAR_AWARE_SCALING_UTILS_DIR)
STATE_POLICY_UTILS_DIR = os.path.join(HOME_DIR, "state_policy_utils")

sys.path.append(STATE_POLICY_UTILS_DIR)

from push_t_state_env import PushTEnv
from push_t_state_dataset import PushTStateDataset, normalize_data, unnormalize_data
from push_t_state_network import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

def main():
    
    # Find the proper evaluation seeds
    single_model_eval_results_path = os.path.join(VAR_AWARE_SCALING_UTILS_DIR, "single_model_eval_results.json")
    nonzero_eval_seeds = find_nonzero_eval_seeds(single_model_eval_results_path)
    print(len(nonzero_eval_seeds),nonzero_eval_seeds)

if __name__ == '__main__':
    main()