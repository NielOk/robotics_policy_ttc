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

# === Single trial evaluation ===
def evaluate_single_trial(args):
    checkpoint_path, stats, pred_horizon, obs_horizon, action_horizon, action_dim, num_diffusion_iters, device_id, seed = args

    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # Load model checkpoint inside the process
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_state_dict = ckpt["ema_state_dict"]

    model = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_horizon * 5)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()

    max_steps = 200
    env = PushTEnv()
    env.seed(seed)
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

# === Multi-trial evaluator ===
def evaluate_all_single_model(checkpoint_path, stats, device, num_trials=10):
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    action_dim = 2
    obs_dim = 5
    num_diffusion_iters = 100

    print(f"[load] {checkpoint_path}")

    # === Generate evaluation seeds > 200 ===
    seeds = np.random.randint(201, 10_000_000, size=num_trials).tolist()

    args = [
        (
            checkpoint_path,
            stats,
            pred_horizon,
            obs_horizon,
            action_horizon,
            action_dim,
            num_diffusion_iters,
            0,         # device_id
            seed
        )
        for seed in seeds
    ]

    num_workers = min(os.cpu_count(), num_trials)
    scores = []
    batch_size = 5  # Adjust as needed

    for i in range(0, num_trials, batch_size):
        batch_args = args[i:i + batch_size]
        with mp.Pool(processes=min(batch_size, len(batch_args))) as pool:
            batch_scores = pool.map(evaluate_single_trial, batch_args)
            scores.extend(batch_scores)

    # === Track performance ===
    solved_seeds = [seeds[i] for i, score in enumerate(scores) if score > 0.0]
    failed_seeds = [seeds[i] for i, score in enumerate(scores) if score == 0.0]
    trial_results = [{"seed": seeds[i], "score": scores[i]} for i in range(num_trials)]

    results = {
        "description": "Evaluation of a single checkpoint on PushT dataset. Each trial uses a unique seed >200.",
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "all_scores": scores,
        "seeds": seeds,
        "solved_seeds": solved_seeds,
        "failed_seeds": failed_seeds,
        "trial_results": trial_results
    }

    print(f"Mean: {results['mean_score']:.3f} | Std: {results['std_score']:.3f}")
    print(f"Solved: {len(solved_seeds)} / {num_trials} trials")

    with open("single_model_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

# === Entry point ===
if __name__ == '__main__':
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

    checkpoint_model_path = "epoch_20.pt"
    results = evaluate_all_single_model(checkpoint_model_path, stats, device=device, num_trials=20)

    with open("single_model_eval_results.json", "w") as f:
        json.dump(results, f, indent=2)