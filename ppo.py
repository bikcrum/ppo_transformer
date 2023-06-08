import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from network import Actor, Critic


class PPO:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        self.actor = Actor(args)
        self.critic = Critic(args)

        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        if self.args.set_adam_eps:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_a, eps=self.args.eps)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_c, eps=self.args.eps)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_c)

    def update(self, buffer, total_steps):
        losses = []
        entropies = []

        dev_buffer = buffer['s'].device

        causal_mask = nn.Transformer.generate_square_subsequent_mask(self.args.transformer_window).to(dev_buffer)

        ep_lens = buffer['ep_lens']

        # Get buffer size
        buffer_size = buffer['s'].size(0)

        # Create an episode lookup tensor
        episode_lookup = torch.arange(ep_lens.size(0), device=dev_buffer).repeat_interleave(ep_lens)

        # Calculate episode start indices
        ep_start_indices = torch.cat((torch.tensor([0], dtype=torch.int32, device=dev_buffer), ep_lens.cumsum(0)[:-1]))

        # Create sampling indices for sequences
        sampling_indices = torch.cat([torch.arange(s, max(s, s + l) + 1, device=dev_buffer) for s, l in
                                      zip(ep_start_indices, ep_lens - self.args.transformer_window)], dim=0)

        # Determine the sequence length
        seq_len = min(ep_lens.max(), self.args.transformer_window)

        # Create a range of indices for selecting sequences
        select_range = torch.arange(seq_len, device=dev_buffer)

        # Get the sequence lengths
        seq_lens = ep_lens[episode_lookup[sampling_indices]].clamp_max(seq_len)

        # Create an active mask for valid sequence indices
        active = select_range < seq_lens.unsqueeze(-1)

        # Create the indices for selecting sequences from the buffer
        select_indices = (sampling_indices.unsqueeze(-1) + select_range).clamp_max(buffer_size - 1)

        # Create a mask for start sequences
        start_sequence = ep_start_indices[episode_lookup[sampling_indices]] == sampling_indices

        if self.args.loss_only_at_end:
            # Set active mask to False for non-start sequences
            active[~start_sequence] = False
            # Set active mask to True for last time step of each sequence
            active[torch.arange(active.size(0)), seq_lens - 1] = True

        # Drop sequences that are short
        if self.args.drop_short_sequence:
            valid_mask = seq_lens >= self.args.transformer_window
            active = active[valid_mask]
            select_indices = select_indices[valid_mask]

        batch_size = active.size(0)

        for i in range(self.args.num_epoch):
            sampler = tqdm.tqdm(BatchSampler(SubsetRandomSampler(range(batch_size)), self.args.mini_batch_size, False))
            for index in sampler:
                # Get active mask for selected indices
                _active = active[index].to(self.device)

                # Get transitions for selected indices
                s = buffer['s'][select_indices[index]].to(self.device)
                # s: [batch_size, seq_len, args.state_dim]

                a = buffer['a'][select_indices[index]].to(self.device)
                # a: [batch_size, seq_len, args.action_dim]

                a_logprob = buffer['a_logprob'][select_indices[index]].to(self.device)
                # a_logprob: [batch_size, seq_len]

                adv = buffer['adv'][select_indices[index]].to(self.device)
                # adv: [batch_size, seq_len]

                v_target = buffer['v_target'][select_indices[index]].to(self.device)
                # v_target: [batch_size, seq_len]

                # Causal mask for transformer
                c_mask = causal_mask[:s.size(1), :s.size(1)].to(self.device)

                # Forward pass
                dist_now = self.actor.pdf(s, c_mask)
                values_now = self.critic(s, c_mask).squeeze(-1)

                del s, c_mask

                ratios = torch.exp(dist_now.log_prob(a)[_active].sum(-1) - a_logprob[_active].sum(-1))

                del a_logprob

                # actor loss
                adv = adv[_active]
                surr1 = ratios * adv
                surr2 = torch.clamp(ratios, 1 - self.args.epsilon, 1 + self.args.epsilon) * adv
                dist_entropy = dist_now.entropy()[_active].sum(-1)
                actor_loss = -torch.min(surr1, surr2) - self.args.entropy_coef * dist_entropy

                del ratios, surr1, surr2, dist_now

                actor_loss = actor_loss.mean()
                critic_loss = F.mse_loss(values_now[_active], v_target[_active]) * 0.5

                log = {'epochs': i, 'actor_loss': actor_loss.item(),
                       'critic_loss': critic_loss.item(), 'batch_size': batch_size,
                       'active_count': len(adv), 'active_shape': _active.shape}

                sampler.set_description(str(log))

                losses.append((log['actor_loss'], log['critic_loss']))

                # Update
                self.optimizer_actor.zero_grad()
                self.optimizer_critic.zero_grad()

                actor_loss.backward()
                critic_loss.backward()

                if self.args.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.optimizer_actor.step()
                self.optimizer_critic.step()

                mean_entropy = dist_entropy.mean().item()
                entropies.append((mean_entropy, - self.args.entropy_coef * mean_entropy))

                del adv, actor_loss, critic_loss, values_now, v_target, _active

                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

        if self.args.use_lr_decay:
            self.lr_decay(total_steps)

        a_loss, c_loss = zip(*losses)
        entropy, entropy_bonus = zip(*entropies)

        del causal_mask, losses, entropies, buffer, episode_lookup, ep_start_indices, ep_lens, sampling_indices, \
            select_range, seq_lens, active, select_indices, start_sequence

        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return np.mean(a_loss), np.mean(c_loss), np.mean(entropy), np.mean(entropy_bonus), batch_size

    def lr_decay(self, total_steps):
        lr_a_now = self.args.lr_a * (1 - total_steps / self.args.max_steps)
        lr_c_now = self.args.lr_c * (1 - total_steps / self.args.max_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
