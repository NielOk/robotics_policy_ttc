import os
import numpy as np
import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm

from push_t_state_env import *
from push_t_state_dataset import *
from push_t_state_network import *

def prepare_network(obs_horizon, obs_dim, action_dim, num_diffusion_iters, dataset_path):
    noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon
    )

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    dataset = PushTStateDataset(
        dataset_path=dataset_path,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=256,
        num_workers=1,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

    return noise_pred_net, noise_scheduler, dataloader

def train(noise_pred_net, noise_scheduler, dataloader, device="cuda", num_epochs=100, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    noise_pred_net.to(device)
    ema = EMAModel(parameters=noise_pred_net.parameters(), power=0.75)

    optimizer = torch.optim.AdamW(noise_pred_net.parameters(), lr=1e-4, weight_decay=1e-6)

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []

            with tqdm(dataloader, desc=f'Batch (Epoch {epoch_idx})', leave=False) as tepoch:
                for nbatch in tepoch:
                    nobs = nbatch['obs'].to(device)
                    naction = nbatch['action'].to(device)
                    B = nobs.shape[0]

                    obs_cond = nobs[:, :obs_horizon, :].flatten(start_dim=1).to(device)
                    noise = torch.randn_like(naction).to(device)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (B,), device=device).long()
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                    noise_pred = noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    ema.step(noise_pred_net.parameters())

                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # Save minimal inference-only checkpoint
            checkpoint = {
                "epoch": epoch_idx,
                "model_state_dict": noise_pred_net.state_dict(),
                "ema_state_dict": {
                    k: v.clone().detach().cpu()
                    for k, v in zip(noise_pred_net.state_dict().keys(), ema.shadow_params)
                }
            }
            torch.save(checkpoint, os.path.join(checkpoint_dir, f"epoch_{epoch_idx}.pt"))

    # optional: apply EMA weights to model after training
    ema.copy_to(noise_pred_net.parameters())

if __name__ == '__main__':
    pred_horizon = 16
    obs_horizon = 2
    action_horizon = 8
    obs_dim = 5
    action_dim = 2
    num_diffusion_iters = 100
    dataset_path = "pusht_cchi_v7_replay"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    noise_pred_net, noise_scheduler, dataloader = prepare_network(
        obs_horizon, obs_dim, action_dim, num_diffusion_iters, dataset_path
    )

    num_epochs = 100
    train(noise_pred_net, noise_scheduler, dataloader, device=device, num_epochs=num_epochs)