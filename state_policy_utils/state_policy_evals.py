import os
import json
import numpy as np
import torch
from tqdm import tqdm
import collections
import multiprocessing as mp

from push_t_state_env import PushTEnv
from push_t_state_dataset import PushTStateDataset, normalize_data, unnormalize_data
from push_t_state_network import ConditionalUnet1D
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# This function is picklable for multiprocessing
def evaluate_single_trial(args):
    model_state_dict, stats, pred_horizon, obs_horizon, action_horizon, action_dim, num_diffusion_iters, device_id = args

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    model = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_horizon * 5)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    max_steps = 200
    env = PushTEnv()
    obs, _ = env.reset()

    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    rewards = []
    step_idx = 0
    done = False

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    noise_scheduler.set_timesteps(num_diffusion_iters)

    with torch.no_grad():
        while not done and step_idx <= max_steps:
            obs_seq = np.stack(obs_deque)
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            nobs = torch.from_numpy(nobs).float().to(device)
            obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

            noisy_action = torch.randn((1, pred_horizon, action_dim), device=device)
            naction = noisy_action

            for k in noise_scheduler.timesteps:
                noise_pred = model(naction, k, global_cond=obs_cond)
                naction = noise_scheduler.step(noise_pred, k, naction).prev_sample

            naction = naction[0].cpu().numpy()
            action_pred = unnormalize_data(naction, stats['action'])

            actions = action_pred[obs_horizon - 1:obs_horizon - 1 + action_horizon]
            for act in actions:
                obs, reward, done, _, _ = env.step(act)
                obs_deque.append(obs)
                rewards.append(reward)
                step_idx += 1
                if done or step_idx > max_steps:
                    done = True
                    break

    return float(max(rewards))


def evaluate_all(checkpoint_dir, stats, device, num_trials=10):
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    action_dim = 2
    obs_dim = 5
    num_diffusion_iters = 100

    results = {}
    num_workers = min(os.cpu_count(), num_trials)

    for epoch in tqdm(range(100), desc="Evaluating checkpoints"):
        ckpt_path = os.path.join(checkpoint_dir, f"epoch_{epoch}.pt")
        if not os.path.exists(ckpt_path):
            print(f"[skip] Missing: {ckpt_path}")
            continue

        print(f"[load] {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model_state_dict = ckpt["ema_state_dict"]

        args = [
            (
                model_state_dict,
                stats,
                pred_horizon,
                obs_horizon,
                action_horizon,
                action_dim,
                num_diffusion_iters,
                0  # device_id 0; can be adapted if using multi-GPU
            )
            for _ in range(num_trials)
        ]

        with mp.Pool(processes=num_workers) as pool:
            scores = pool.map(evaluate_single_trial, args)

        results[epoch] = {
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "all_scores": scores
        }

        print(f"  Mean: {results[epoch]['mean_score']:.3f} | Std: {results[epoch]['std_score']:.3f}")

    with open("state_policy_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    stats_path = "pusht_cchi_v7_replay_stats.pt"
    if not os.path.exists(stats_path):
        print(f"[generate] Saving stats to {stats_path} ...")
        dataset = PushTStateDataset(
            dataset_path="pusht_cchi_v7_replay",
            pred_horizon=16,
            obs_horizon=2,
            action_horizon=8
        )
        torch.save(dataset.stats, stats_path)

    stats = torch.load(stats_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = evaluate_all("checkpoints", stats, device=device, num_trials=5)