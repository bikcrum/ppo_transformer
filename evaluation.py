import argparse
import logging
import os.path

import cv2
import gym
import numpy as np
import pandas as pd
import torch
import tqdm
from torch import nn

import wandb
from network import Actor

logging.basicConfig(level=logging.INFO)


def get_action(actor, s, mask):
    assert s.dim() == 4, f"s must be 4D, [seq_len, *state_dim]. Actual: {s.dim()}"

    # Add batch dimension
    s = s.unsqueeze(0)
    # s: [1, seq_len, *state_dim]

    a, _, attn_maps = actor.forward(s, mask, need_weights=True)

    # Get output from last observation
    a = a.squeeze(0)[-1]
    # mean: [action_dim]

    attn_maps = torch.stack(attn_maps)

    return a, attn_maps


def scale_action(y1, y2, x1, x2, x):
    return (x - x1) * (y2 - y1) / (x2 - x1) + y1


def evaluate_policy(env_name, run_name, replace=True, best=True, render=True):
    parser = argparse.ArgumentParser("Hyperparameters")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Output dimension of CNN and input to transformer")
    parser.add_argument("--transformer_window", type=int, default=64, help="Maximum sequence length in transformer")
    parser.add_argument("--time_horizon", type=int, default=1000, help="The maximum length of the episode")
    parser.add_argument('--transformer_num_layers', type=int, default=4, help='Number of layers in transformer encoder')
    parser.add_argument('--transformer_nhead', type=int, default=4, help='Number of attention heads in transformer')
    parser.add_argument('--transformer_dim_feedforward', type=int, default=64, help='FF dimension in transformer')
    parser.add_argument('--transformer_dropout', type=int, default=0.0,
                        help='Dropout positional encoder and transformer encoder')
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Orthogonal initialization for FF")

    args = parser.parse_args()

    env = gym.make(env_name, domain_randomize=True)

    args.state_dim = env.observation_space.shape
    args.action_dim = env.action_space.shape[0]
    action_low = env.action_space.low
    action_high = env.action_space.high

    print(f"env={env_name}")
    print(f"state_dim={args.state_dim}")
    print(f"action_dim={args.action_dim}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    actor = Actor(args)
    actor.to(device)

    wandb.login()

    run = wandb.Api().run(os.path.join(env_name, run_name.replace(':', '_')))

    os.makedirs('saved_models', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    logging.info(f'Checkpoint loaded from: {run_name}')
    if best:
        if replace or not os.path.exists(f'saved_models/agent-{run_name}.pth'):
            run.file(name=f'saved_models/agent-{run_name}.pth').download(replace=replace)

        with open(f'saved_models/agent-{run_name}.pth', 'rb') as r:
            checkpoint = torch.load(r, map_location=device)

        actor.load_state_dict(checkpoint)
    else:
        if replace or not os.path.exists(f'checkpoints/checkpoint-{run_name}.pt'):
            run.file(name=f'checkpoints/checkpoint-{run_name}.pt').download(replace=replace)

        with open(f'checkpoints/checkpoint-{run_name}.pt', 'rb') as r:
            checkpoint = torch.load(r, map_location=device)

        actor.load_state_dict(checkpoint['actor_state_dict'])

    n_epoch = 50
    reward = 0
    length = 0

    state_buffer = torch.zeros(args.time_horizon, *args.state_dim, dtype=torch.float32, device=device)

    causal_mask = nn.Transformer.generate_square_subsequent_mask(args.transformer_window).to(device)

    for _ in tqdm.tqdm(range(n_epoch)):

        s = env.reset(options={"randomize": True})

        episode_length = 0
        episode_reward = 0

        for step in range(args.time_horizon):
            state_buffer[step] = torch.tensor(s / 255.0, dtype=torch.float32, device=device)

            start_idx, end_idx = max(0, step - args.transformer_window + 1), step + 1

            a, attn_maps = get_action(actor,
                                      state_buffer[start_idx:end_idx],
                                      causal_mask[:end_idx - start_idx, :end_idx - start_idx])

            # logging.info(f'Action:{a}', pd.DataFrame(state_buffer[step].flatten()).describe())

            action = scale_action(y1=action_low, y2=action_high,
                                  x1=-1, x2=1, x=a.cpu().numpy())

            s_, r, done, info = env.step(action)

            if render:
                env.render()

                attn_maps_resized = []
                for attn_map in attn_maps.cpu().numpy():
                    attn_maps_resized.append(cv2.resize(attn_map.squeeze(0), (128, 128),
                                                        interpolation=cv2.INTER_NEAREST))

                cv2.imshow(f'attn_map', np.concatenate(attn_maps_resized, axis=-1))

                cv2.waitKey(1)

            episode_reward += r
            episode_length += 1

            s = s_

            if done:
                break

        logging.info(f'Reward:{episode_reward}')

        reward += episode_reward
        length += episode_length

    return reward / n_epoch, length / n_epoch


if __name__ == '__main__':
    with torch.inference_mode():
        evaluate_policy(env_name='CarRacing-v1',
                        run_name='2023-06-07 20:31:08.669398',
                        replace=False,
                        best=False,
                        render=True)
