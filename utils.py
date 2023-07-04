import glob
import logging
import os
from copy import deepcopy

import gym
import torch
import wandb
from torch import nn
import ray
from normalization import RewardScaling
from replaybuffer import ReplayBuffer

logging.getLogger().setLevel(logging.DEBUG)


@ray.remote
class Worker:
    def __init__(self, env_name, dispatcher, actor, args, device, worker_id):
        self.env = gym.make(env_name)

        self.action_low = self.env.action_space.low
        self.action_high = self.env.action_space.high

        self.dispatcher = dispatcher
        if args.use_reward_scaling:
            self.reward_scaling = RewardScaling(shape=1, gamma=args.gamma)
        self.args = args
        self.device = device
        self.actor = deepcopy(actor).to(device)
        self.worker_id = worker_id
        self.causal_mask = nn.Transformer.generate_square_subsequent_mask(args.transformer_window).to(self.device)

    @staticmethod
    def scale_action(y1, y2, x1, x2, x):
        return (x - x1) * (y2 - y1) / (x2 - x1) + y1

    def update_model(self, new_actor_params):
        for p, new_p in zip(self.actor.parameters(), new_actor_params):
            p.data.copy_(new_p)

    def get_action(self, s, mask, deterministic=False):
        assert s.dim() == 2, f"s must be 2D, [seq_len, state_dim]. Actual: {s.dim()}"

        # Add batch dimension
        s = s.unsqueeze(0)
        # s: [1, seq_len, state_dim]

        if deterministic:
            a, _ = self.actor(s, mask)

            # Get output from last observation
            a = a.squeeze(0)[-1]
            # a: [action_dim]
            return a
        else:
            dist = self.actor.pdf(s, mask)
            a = dist.sample()
            # a: [1, seq_len, action_dim]

            # a_logprob = dist.log_prob(a)
            # a_logprob: [1, seq_len, action_dim]

            # a, a_logprob = a.squeeze(0)[-1], a_logprob.squeeze(0)[-1]
            # a: [action_dim], a_logprob: [action_dim]

            a = a.squeeze(0)[-1]
            # a: [action_dim]

            return a

    def collect(self, max_ep_len, render=False):
        with torch.inference_mode():
            replay_buffer = ReplayBuffer(self.args, buffer_size=max_ep_len)

            episode_reward = 0

            s = self.env.reset(options={"randomize": True})

            if self.args.use_reward_scaling:
                self.reward_scaling.reset()

            for step in range(max_ep_len):
                replay_buffer.store_state(torch.tensor(s, dtype=torch.float32, device=self.device))

                start_idx, end_idx = max(0, step - self.args.transformer_window + 1), step + 1

                a = self.get_action(replay_buffer.buffer['s'][start_idx:end_idx],
                                    self.causal_mask[:end_idx - start_idx, :end_idx - start_idx],
                                    deterministic=False)

                action = self.scale_action(y1=self.action_low, y2=self.action_high,
                                           x1=-1, x2=1, x=a.cpu().numpy())

                s, r, done, _ = self.env.step(action)

                if render and not done:
                    self.env.render()

                episode_reward += r

                if done and step != self.args.time_horizon - 1:
                    dw = True
                else:
                    dw = False

                if self.args.use_reward_scaling:
                    r = self.reward_scaling(r)

                r = torch.tensor(r, dtype=torch.float32, device=self.device)
                replay_buffer.store_transition(a, r, dw)

                if done:
                    break

                if not ray.get(self.dispatcher.is_collecting.remote()):
                    del replay_buffer
                    return

            replay_buffer.store_last_state(torch.tensor(s, dtype=torch.float32, device=self.device))

            return replay_buffer, episode_reward, step + 1, self.worker_id

    def evaluate(self, max_ep_len, render=False):
        with torch.inference_mode():
            assert max_ep_len <= self.args.time_horizon, f"max_ep_len must be less than or equal time_horizon."

            state_buffer = torch.zeros(self.args.transformer_window, self.args.state_dim, dtype=torch.float32)

            s = self.env.reset(options={"randomize": True})

            episode_reward = 0

            for step in range(max_ep_len):
                seq_len = min(step + 1, self.args.transformer_window)

                # state_buffer[step] = torch.tensor(s, dtype=torch.float32, device=self.device)

                state_buffer[seq_len - 1] = torch.tensor(s, dtype=torch.float32, device=self.device)

                a = self.get_action(state_buffer[:seq_len], self.causal_mask[:seq_len, :seq_len], deterministic=True)

                if seq_len == self.args.transformer_window:
                    state_buffer = state_buffer.roll(-1, dims=1)

                action = self.scale_action(y1=self.action_low, y2=self.action_high,
                                           x1=-1, x2=1, x=a.cpu().numpy())

                s, r, done, _ = self.env.step(action)

                if render and not done:
                    self.env.render()

                episode_reward += r

                if done:
                    break

                if not ray.get(self.dispatcher.is_evaluating.remote()):
                    return

            del state_buffer

            return None, episode_reward, step + 1, self.worker_id


@ray.remote
class Dispatcher:
    def __init__(self):
        self.collecting = False
        self.evaluating = False

    def is_collecting(self):
        return self.collecting

    def is_evaluating(self):
        return self.evaluating

    def set_collecting(self, val):
        self.collecting = val

    def set_evaluating(self, val):
        self.evaluating = val


def get_device():
    if torch.cuda.is_available():
        return torch.device("cpu"), torch.device("cuda")
    else:
        try:
            # For apple silicon
            if torch.backends.mps.is_available():
                # May not require in future pytorch after fix
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1"
                return torch.device("cpu"), torch.device("mps")
            else:
                return torch.device("cpu"), torch.device("cpu")
        except Exception as e:
            logging.error(e)
            return torch.device("cpu"), torch.device("cpu")


def optimizer_to_device(optimizer, device):
    state_dict = optimizer.state_dict()

    if 'state' not in state_dict:
        logging.warning(f'No state in optimizer. Not converting to {device}')
        return

    states = state_dict['state']

    for k, state in states.items():
        for key, val in state.items():
            states[k][key] = val.to(device)


def update_model(model, new_model_params):
    for p, new_p in zip(model.parameters(), new_model_params):
        p.data.copy_(new_p)


def init_logger(args, agent, run_name, project_name, previous_run, parent_run):
    epochs = 0
    total_steps = 0
    trajectory_count = 0

    # Create new run from scratch if previous run is not provided
    if previous_run is None:
        # parent_run by default is equal to run name if not provided
        if parent_run is None:
            parent_run = run_name

        run = wandb.init(
            entity='team-osu',
            project=project_name,
            name=run_name,
            # mode='disabled',
            config={**args.__dict__, 'parent_run': parent_run},
            id=run_name.replace(':', '_'),
        )
    # Previous run is given, parent run not given -> resume training
    elif parent_run is None:
        run = wandb.init(
            entity='team-osu',
            project=project_name,
            resume='allow',
            id=previous_run.replace(':', '_'),
        )

        if run.resumed:
            checkpoint = torch.load(run.restore(f'checkpoints/checkpoint-{run.name}.pt'), map_location=agent.device)
            logging.info(f'Resuming from the run: {run.name} ({run.id})')
            total_steps = checkpoint['total_steps']
            trajectory_count = checkpoint['trajectory_count']
            epochs = checkpoint['epochs']
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
            agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])
        else:
            logging.error(f'Run: {previous_run} did not resume')
            raise Exception(f'Run: {previous_run} did not resume')
    # Previous run is given, parent run is given, resume training but create new run under same parent
    else:
        wandb.login()

        run = wandb.Api().run(os.path.join(project_name, previous_run.replace(':', '_')))

        logging.info(f'Checkpoint loaded from: {previous_run}')
        run.file(name=f'checkpoints/checkpoint-{previous_run}.pt').download(replace=True)

        with open(f'checkpoints/checkpoint-{previous_run}.pt', 'rb') as r:
            checkpoint = torch.load(r, map_location=agent.device)

        # Create new run
        run = wandb.init(
            entity='team-osu',
            project=project_name,
            name=run_name,
            config={**args.__dict__, 'parent_run': parent_run},
            id=run_name.replace(':', '_'),
        )

        total_steps = checkpoint['total_steps']
        trajectory_count = checkpoint['trajectory_count']
        epochs = checkpoint['epochs']
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.critic.load_state_dict(checkpoint['critic_state_dict'])
        agent.optimizer_actor.load_state_dict(checkpoint['optimizer_actor_state_dict'])
        agent.optimizer_critic.load_state_dict(checkpoint['optimizer_critic_state_dict'])

    parent = os.path.dirname(os.path.abspath(__file__))

    cwd = os.getcwd()
    base_path = os.path.join(cwd, parent, '*.py')

    for file in glob.glob(base_path):
        file_path = os.path.relpath(file, start=cwd)
        logging.debug('Saving file:{} to wandb'.format(file_path))
        run.save(file_path, policy='now')

    return run, epochs, total_steps, trajectory_count
