import math
import random
import time
from collections import deque
from typing import Tuple, Dict, Deque, List, Any
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import clear_output
from datetime import datetime, timezone, timedelta
import logging
import sys
from colorlog import ColoredFormatter
from joint_environment import JointEnvironmentTSN
from segmenttree import SumSegmentTree, MinSegmentTree
from scheduling_environment import SchedulingEnvironmentTSN
from config import SAVE_PLOTS, PLOTS_PATH, BATCH_SIZE, TARGET_UPDATE, GAMMA, REPLAY_BUFFER_SIZE, \
    RAINBOW_LOG_LEVEL, RAINBOW_LOG_FILE_NAME, ALPHA, BETA, N_STEP, V_MIN, V_MAX, ATOM_SIZE, PRIOR_EPS, LEARNING_RATE, \
    MODEL_PATH, TAU, ALL_ROUTES, VNF_LISTS_DIRECTORY, CUSTOM_ROUTE


def _get_n_step_info(n_step_buffer: Deque, gamma: float) -> Tuple[np.int64, np.ndarray, bool]:
    """Return n step rew, next_obs, and done."""
    # info of the last transition
    rew, next_obs, done = n_step_buffer[-1][-3:]

    for transition in reversed(list(n_step_buffer)[:-1]):
        r, n_o, d = transition[-3:]

        rew = r + gamma * rew * (1 - d)
        next_obs, done = (n_o, d) if d else (next_obs, done)

    return rew, next_obs, done


class ReplayBuffer:
    """A simple numpy replay buffer."""

    def __init__(
            self,
            obs_dim: int,
            size: int,
            batch_size: int = 32,
            n_step: int = 1,
            gamma: float = 0.99
    ):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
            self,
            obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ) -> tuple | Any:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # make an n-step transition
        rew, next_obs, done = _get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
            # for N-step Learning
            indices=idxs,
        )

    def sample_batch_from_idxs(
            self, idxs: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(
            self,
            obs_dim: int,
            size: int,
            batch_size: int = 32,
            alpha: float = 0.6,
            n_step: int = 1,
            gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(
            obs_dim, size, batch_size, n_step, gamma
        )
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(
            self,
            obs: np.ndarray,
            act: int,
            rew: float,
            next_obs: np.ndarray,
            done: bool,
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        """Store experience and priority."""
        transition = super().store(obs, act, rew, next_obs, done)

        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size

        return transition

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.



    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            std_init: float = 0.5,
    ):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)

        return x.sign().mul(x.abs().sqrt())


class Network(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            atom_size: int,
            support: torch.Tensor
    ):
        """Initialization."""
        super(Network, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 60),
            nn.ReLU(),
        )

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(60, 60)
        self.advantage_layer = NoisyLinear(60, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(60, 60)
        self.value_layer = NoisyLinear(60, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()


class SchedulingRainbowAgent:
    """DQN Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
            self,
            env: SchedulingEnvironmentTSN | JointEnvironmentTSN,
            log_file_id: str
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            log_file_id (str): string to add to the end of the log file name
        """

        # Logging settings
        self.logger = logging.getLogger('scheduling_rainbow')
        self.logger.setLevel(RAINBOW_LOG_LEVEL)
        self.logger.addHandler(logging.FileHandler(f'{RAINBOW_LOG_FILE_NAME}'
                                                   f'{log_file_id}.log', mode='w', encoding='utf-8'))
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(ColoredFormatter('%(log_color)s%(message)s'))
        self.logger.addHandler(stream_handler)

        # device: cpu / gpu
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if str(self.device) == 'cpu':
            self.logger.warning('[!] WARNING! Processing device: ' + str(self.device))
        else:
            self.logger.info('[I] A Processing device: ' + str(self.device))

        # Spaces
        obs_dim = env.scheduling_state_space.shape[0]
        action_dim = env.scheduling_action_space.n

        self.env = env
        self.batch_size = BATCH_SIZE
        self.target_update = TARGET_UPDATE
        self.gamma = GAMMA
        self.learning_rate = LEARNING_RATE
        self.tau = TAU

        # PER
        # memory for 1-step Learning
        self.beta = BETA
        self.prior_eps = PRIOR_EPS
        self.memory = PrioritizedReplayBuffer(
            obs_dim, REPLAY_BUFFER_SIZE, BATCH_SIZE, alpha=ALPHA, gamma=GAMMA
        )

        # memory for N-step Learning
        self.use_n_step = True if N_STEP > 1 else False
        if self.use_n_step:
            self.n_step = N_STEP
            self.memory_n = ReplayBuffer(
                obs_dim, REPLAY_BUFFER_SIZE, BATCH_SIZE, n_step=N_STEP, gamma=GAMMA
            )

        # Categorical DQN parameters
        self.v_min = V_MIN
        self.v_max = V_MAX
        self.atom_size = ATOM_SIZE
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device)

        # networks: dqn, dqn_target
        self.online_net = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.target_net = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        # optimizer
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)

        # transition to store in memory
        self.transition = list()

        # mode: train / test
        self.is_test = False

    @staticmethod
    def _plot(step: int, max_steps: int, scores: List[float], mean_scores: List[float], losses: List[float],
              mean_losses: List[float]):
        """
        Plot the training progress.

        Parameters:
        - episode (int): The current episode number.
        - scores (List[float]): List of scores over episodes.
        - mean_scores (List[float]): List of mean scores over episodes.
        - losses (List[float]): List of losses over episodes.
        - mean_losses (List[float]): List of mean losses over episodes.
        - epsilons (List[float]): List of epsilon values over episodes.

        This static method plots the training progress, including scores, losses, and epsilon values.

        """
        clear_output(True)
        plt.figure(figsize=(20, 5))

        # Plot scores
        plt.subplot(121)
        plt.title(f'Step {step}. Average Score: {np.mean(scores[-100:]):.2f}')
        plt.plot(scores, label='Real')
        plt.plot(mean_scores, label='Average')

        # Plot losses
        plt.subplot(122)
        plt.title('Loss Over Steps')
        plt.plot(losses, label='Real')
        plt.plot(mean_losses, label='Average')

        if SAVE_PLOTS and step == max_steps:
            date = datetime.now(tz=timezone(timedelta(hours=2))).strftime('%Y%m%d%H%M')
            plt.savefig(PLOTS_PATH + 'plot_' + date + '_scheduling_' + str(int(np.mean(scores[-100:]))) + '.png')

        plt.show()

    def select_action(self, state: np.ndarray, info: Dict) -> int:
        """Select an action from the input state."""

        possible_actions = info['valid_actions']
        self.logger.debug('[D] VALID ACTIONS (' + str(len(possible_actions)) + ' options): ' + str(possible_actions))
        state_tensor = torch.FloatTensor(state).to(self.device)
        q_values = self.online_net(state_tensor)
        possible_q_values = [(idx, q_values[0, idx]) for idx in possible_actions]
        selected_action, max_q_value = max(possible_q_values, key=lambda x: x[1])

        if not self.is_test:
            self.transition = [state, selected_action]

        return selected_action

    def execute_action(self, action: int) -> tuple[Any, Any, Any, Any, Any]:
        """Take an action and return the response of the env."""
        next_state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        if not self.is_test:
            self.transition += [reward, next_state, done]

            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)

        return next_state, reward, terminated, truncated, info

    def update_model(self) -> int | int | float | float | bool | bool:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(
            samples["weights"].reshape(-1, 1)
        ).to(self.device)
        indices = samples["indices"]

        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)

        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are going to combine 1-step loss and n-step loss to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        # NoisyNet: reset noise
        self.online_net.reset_noise()
        self.target_net.reset_noise()

        return loss.item()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.online_net(next_state).argmax(1)
            next_dist = self.target_net.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.online_net.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def _target_soft_update(self, tau: float = 0.005):
        """
        Perform a soft update of the target network.

        Parameters:
        - tau (float, optional): The interpolation parameter for the soft update.
          A smaller value simulation_results in a slower update. Default is 0.01.

        This method updates the target network parameters as a weighted average
        of the online network parameters.

        """

        # Get the state dictionaries of the online and target networks
        online_params = dict(self.online_net.named_parameters())
        target_params = dict(self.target_net.named_parameters())

        # Update the target network parameters using a weighted average
        for name in target_params:
            target_params[name].data.copy_(tau * online_params[name].data + (1.0 - tau) * target_params[name].data)

    def load_custom_model(self, model_name):
        """
        Load a pre-trained model from a file and update the online and target networks.

        This method attempts to load a pre-trained model from the specified file path. If successful,
        it updates the weights of both the online and target networks with the loaded state dictionary.
        Additionally, the target network is updated with the state dictionary of the online network.

        Raises:
            FileNotFoundError: If the model file is not found at the specified path.
            Exception: If an unexpected error occurs during the model loading process.

        Note:
            Make sure to handle the returned exceptions appropriately when calling this method.

        """
        try:
            # Try to open the model file and load its state dictionary
            with open(model_name, 'rb') as model_file:
                state_dict = torch.load(model_file)
                # Update the online network with the loaded state dictionary
                self.online_net.load_state_dict(state_dict)
                # Update the target network with the state dictionary of the online network
                self.target_net.load_state_dict(self.online_net.state_dict())
        except FileNotFoundError:
            # Handle the case where the model file is not found
            self.logger.error('[!] Model file not found at ' + MODEL_PATH)
        except Exception as e:
            # Handle unexpected errors during the model loading process
            self.logger.error('An unexpected error occurred when loading the saved model: ' + str(e))

    def train(self, max_steps: int, monitor_training: int = 5000, plotting_interval: int = 20000):
        """Train the agent."""
        self.is_test = False

        # *******************************************  Reset env  ******************************************************
        if ALL_ROUTES:
            self.env.vnf_list = self.env.import_vnf_list(f'{VNF_LISTS_DIRECTORY}all_routes_'
                                                         f'{min(self.env.graph.nodes)}_{max(self.env.graph.nodes)}.pkl')
        else:
            self.env.vnf_list = self.env.import_vnf_list(f'{VNF_LISTS_DIRECTORY}'
                                                         f'{CUSTOM_ROUTE[0]}_{CUSTOM_ROUTE[1]}.pkl')
        self.env.vnf_list_index = 0

        state, info = self.env.reset()

        update_cnt = 0
        losses = []
        mean_losses = []
        scores = []
        mean_scores = []
        score = 0
        episodes_count = 0
        episode_data = {
            'Optimal_scheduling': 0,
            'Non-optimal_scheduling': 0
        }

        # Record start time for monitoring
        monitoring_start_time = time.time()
        step = 0

        # *******************************************  Start  **********************************************************
        try:
            for step in range(1, max_steps + 1):
                action = self.select_action(state, info)

                next_state, reward, terminated, _, info = self.execute_action(action)

                state = next_state
                score += reward

                # stats
                episode_data['Optimal_scheduling'] += info['Optimal_scheduling']
                episode_data['Non-optimal_scheduling'] += info['Non-optimal_scheduling']

                # PER: increase beta
                fraction = min(step / max_steps, 1.0)
                self.beta = self.beta + fraction * (1.0 - self.beta)

                # *****************************************  Evaluate episode  *****************************************
                if terminated:
                    state, info = self.env.reset()
                    scores.append(score)
                    if len(scores) < 100:
                        mean_scores.append(np.mean(scores[0:]))
                    else:
                        mean_scores.append(np.mean(scores[-100:]))
                    score = 0
                    episodes_count += 1
                # *****************************************  Update model  *********************************************
                if len(self.memory) >= self.batch_size:
                    loss = self.update_model()
                    losses.append(loss)
                    if len(losses) < 100:
                        mean_losses.append(np.mean(losses[0:]))
                    else:
                        mean_losses.append(np.mean(losses[-100:]))
                    update_cnt += 1

                    # if hard update is needed
                    if update_cnt % self.target_update == 0:
                        self._target_soft_update()

                # *****************************************  Monitor training  *****************************************

                # Print training progress at specified intervals
                if step % plotting_interval == 0:
                    self._plot(step, max_steps, scores, mean_scores, losses, mean_losses)
                    plt.close()

                if step % monitor_training == 0:
                    monitoring_elapsed_time = time.time() - monitoring_start_time
                    elapsed_time_formatted = str(timedelta(seconds=int(monitoring_elapsed_time)))
                    self.logger.info(f"[I] Elapsed time: {elapsed_time_formatted} | "
                                     f"Step: {step} | "
                                     f"Rewards: {round(np.mean(scores[-100:]), 3)} | "
                                     f"Loss: {round(np.mean(losses[-100:]), 6)} | "
                                     f"Episodes: {episodes_count} | "
                                     f"Optimal schedules: {episode_data['Optimal_scheduling']} | "
                                     f"Non-optimal schedules: {episode_data['Non-optimal_scheduling']}")
        except KeyboardInterrupt:
            monitoring_elapsed_time = time.time() - monitoring_start_time
            elapsed_time_formatted = str(timedelta(seconds=int(monitoring_elapsed_time)))
            self.logger.info(f"[I] Elapsed time: {elapsed_time_formatted} | "
                             f"Step: {step} | "
                             f"Rewards: {round(np.mean(scores[-100:]), 3)} | "
                             f"Loss: {round(np.mean(losses[-100:]), 6)} | "
                             f"Episodes: {episodes_count} | "
                             f"Optimal schedules: {episode_data['Optimal_scheduling']} | "
                             f"Non-optimal schedules: {episode_data['Non-optimal_scheduling']}")
            if episodes_count > 1000:
                self._plot(step, max_steps, scores, mean_scores, losses, mean_losses)
                plt.close()

        # *******************************************  End training ****************************************************

        monitoring_elapsed_time = time.time() - monitoring_start_time
        elapsed_time_formatted = str(timedelta(seconds=int(monitoring_elapsed_time)))
        self.logger.info('[I] Total run time steps: ' + str(step))
        self.logger.info('[I] Total run episodes: ' + str(episodes_count))
        self.logger.info('[I] Total elapsed time: ' + str(elapsed_time_formatted))
        self.logger.info('[I] Final average reward: ' + str(round(np.mean(scores[-100:]), 3)))

        # Save model
        if episodes_count > 1000:
            date = datetime.now(tz=timezone(timedelta(hours=2))).strftime('%Y%m%d%H%M')
            model_name = 'model_' + date + '_scheduling_' + str(int(np.mean(scores[-100:]))) + '.pt'
            torch.save(self.online_net.state_dict(), MODEL_PATH + model_name)

        self.env.close_env()

    def evaluate(self, max_episodes):
        self.is_test = True
        episode_data = {
            'Optimal_scheduling': 0,
            'Non-optimal_scheduling': 0
        }

        # Episode reward list
        episode_rewards = []
        if ALL_ROUTES:
            self.env.vnf_list = self.env.import_vnf_list(f'{VNF_LISTS_DIRECTORY}all_routes_'
                                                         f'{min(self.env.graph.nodes)}_'
                                                         f'{max(self.env.graph.nodes)}_evaluate.pkl')
        else:
            self.env.vnf_list = self.env.import_vnf_list(f'{VNF_LISTS_DIRECTORY}'
                                                         f'{CUSTOM_ROUTE[0]}_{CUSTOM_ROUTE[1]}_evaluate.pkl')
        self.env.vnf_list_index = 0

        for episode in range(max_episodes):
            state, info = self.env.reset()
            terminated = False
            reward = 0
            step = 0

            while not terminated:
                action = self.select_action(state, info)
                next_state, reward, terminated, _, info = self.execute_action(action)
                state = next_state
                self.logger.debug(f"[D] Step: {step} | Action: {action} | State: {[int(x) for x in state]} | "
                                  f"Reward: {reward} | Terminated: {terminated} | Info: {info}")
                step += 1

                # stats
                episode_data['Optimal_scheduling'] += info['Optimal_scheduling']
                episode_data['Non-optimal_scheduling'] += info['Non-optimal_scheduling']

            # Episode reward
            episode_rewards.append(reward)

        mean_score = np.mean(episode_rewards)
        self.logger.info(f"\n[I] Evaluation mean score over {max_episodes} episodes: {mean_score:.2f}\n")
        self.logger.info(f"[I] Optimal schedules: {episode_data['Optimal_scheduling']} | "
                         f"Non-optimal schedules: {episode_data['Non-optimal_scheduling']}")

    def evaluate_routes(self, max_episodes):
        self.is_test = True

        for i in self.env.graph.nodes:
            for j in self.env.graph.nodes:
                if i != j:
                    self.logger.info('\n[I] Route ' + str(i) + ' -> ' + str(j))

                    episode_data = {
                        'Optimal_scheduling': 0,
                        'Non-optimal_scheduling': 0
                    }

                    # Episode reward list
                    episode_rewards = []

                    self.env.vnf_list = self.env.import_vnf_list(f'{VNF_LISTS_DIRECTORY}{i}_{j}_evaluate.pkl')
                    self.env.vnf_list_index = 0

                    for episode in range(max_episodes):
                        state, info = self.env.reset()
                        terminated = False
                        reward = 0
                        step = 0

                        while not terminated:
                            action = self.select_action(state, info)
                            next_state, reward, terminated, _, info = self.execute_action(action)
                            state = next_state
                            self.logger.debug(f"[D] Step: {step} | Action: {action} | State: {[int(x) for x in state]} "
                                              f"| Reward: {reward} | Terminated: {terminated} | Info: {info}")
                            step += 1

                            # stats
                            episode_data['Optimal_scheduling'] += info['Optimal_scheduling']
                            episode_data['Non-optimal_scheduling'] += info['Non-optimal_scheduling']

                        # Episode reward
                        episode_rewards.append(reward)

                    mean_score = np.mean(episode_rewards)
                    self.logger.info(f"[I] Evaluation mean score over {max_episodes} episodes: {mean_score:.2f}")
                    self.logger.info(f"[I] Optimal schedules: {episode_data['Optimal_scheduling']} | "
                                     f"Non-optimal schedules: {episode_data['Non-optimal_scheduling']}")
