import numpy as np
import logging
import sys
from colorlog import ColoredFormatter
from routing_rainbow import RoutingRainbowAgent
from scheduling_rainbow import SchedulingRainbowAgent
from joint_environment import JointEnvironmentTSN
from config import RAINBOW_LOG_LEVEL, RAINBOW_LOG_FILE_NAME, ALL_ROUTES, VNF_LISTS_DIRECTORY, CUSTOM_ROUTE


class JointRainbowAgent:
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
            env: JointEnvironmentTSN,
            log_file_id: str
    ):
        """Initialization.

        Args:
            env (gym.Env): openAI Gym environment
            log_file_id (str): string to add to the end of the log file name
        """

        self.env = env

        # Logging settings
        self.logger = logging.getLogger('joint_rainbow')
        self.logger.setLevel(RAINBOW_LOG_LEVEL)
        self.logger.addHandler(logging.FileHandler(f'{RAINBOW_LOG_FILE_NAME}'
                                                   f'{log_file_id}.log', mode='w', encoding='utf-8'))
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(ColoredFormatter('%(log_color)s%(message)s'))
        self.logger.addHandler(stream_handler)

        self.routing_agent = RoutingRainbowAgent(self.env, 'routing_cent1')
        self.routing_agent.is_test = True

        self.scheduling_agent = SchedulingRainbowAgent(self.env, 'scheduling_cent1')
        self.scheduling_agent.is_test = True

    def evaluate(self, max_episodes):
        routing_episode_data = {
            'Reached_target': 0,
            'Optimal_route': 0,
            'Delay_exceeded': 0,
            'Terminal_state': 0
        }
        scheduling_episode_data = {
            'Optimal_scheduling': 0,
            'Non-optimal_scheduling': 0
        }

        # Episode reward list
        episode_routing_rewards = []
        episode_scheduling_rewards = []
        if ALL_ROUTES:
            self.env.vnf_list = self.env.import_vnf_list(f'{VNF_LISTS_DIRECTORY}all_routes_'
                                                         f'{min(self.env.graph.nodes)}_'
                                                         f'{max(self.env.graph.nodes)}_evaluate.pkl')
        else:
            self.env.vnf_list = self.env.import_vnf_list(f'{VNF_LISTS_DIRECTORY}'
                                                         f'{CUSTOM_ROUTE[0]}_{CUSTOM_ROUTE[1]}_evaluate.pkl')
        self.env.vnf_list_index = 0

        for episode in range(max_episodes):
            step = 0
            scheduling_reward = 0

            routing_state, routing_info = self.env.routing_reset()
            current_node = self.env.current_node
            routing_action = self.routing_agent.select_action(routing_state, routing_info)
            scheduling_state, scheduling_info = self.env.scheduling_reset((current_node, routing_action))
            scheduling_action = self.scheduling_agent.select_action(scheduling_state, scheduling_info)
            routing_state, routing_reward, routing_terminated, _, routing_info = self.env.routing_step(routing_action)

            self.logger.debug(f"[D] Routing step: {step} | Action: {routing_action} |"
                              f"State: {[int(x) for x in routing_state]} | "
                              f"Reward: {routing_reward} | Terminated: {routing_terminated} | Info: {routing_info}")
            # stats
            routing_episode_data['Reached_target'] += routing_info['Reached_target']
            routing_episode_data['Optimal_route'] += routing_info['Optimal_route']
            routing_episode_data['Delay_exceeded'] += routing_info['Delay_exceeded']
            routing_episode_data['Terminal_state'] += routing_info['Terminal_state']

            while not routing_terminated:
                previous_node = current_node
                current_node = routing_action
                routing_action = self.routing_agent.select_action(routing_state, routing_info)
                scheduling_state, scheduling_reward, _, _, \
                    scheduling_info = self.env.scheduling_step((previous_node, current_node),
                                                               (current_node, routing_action),
                                                               scheduling_action)

                self.logger.debug(f"[D] Scheduling step: {step} | Action: {scheduling_action} |"
                                  f"State: {[int(x) for x in scheduling_state]} | "
                                  f"Reward: {scheduling_reward} | Info: {scheduling_info}")

                # stats
                scheduling_episode_data['Optimal_scheduling'] += scheduling_info['Optimal_scheduling']
                scheduling_episode_data['Non-optimal_scheduling'] += scheduling_info['Non-optimal_scheduling']

                scheduling_action = self.scheduling_agent.select_action(scheduling_state, scheduling_info)
                routing_state, routing_reward, routing_terminated, _, \
                    routing_info = self.env.routing_step(routing_action)

                self.logger.debug(f"[D] Routing step: {step} | Action: {routing_action} |"
                                  f"State: {[int(x) for x in routing_state]} | "
                                  f"Reward: {routing_reward} | Terminated: {routing_terminated} | Info: {routing_info}")
                step += 1

                # stats
                routing_episode_data['Reached_target'] += routing_info['Reached_target']
                routing_episode_data['Optimal_route'] += routing_info['Optimal_route']
                routing_episode_data['Delay_exceeded'] += routing_info['Delay_exceeded']
                routing_episode_data['Terminal_state'] += routing_info['Terminal_state']

            # Episode reward
            episode_routing_rewards.append(routing_reward)
            episode_scheduling_rewards.append(scheduling_reward)

        mean_routing_score = np.mean(episode_routing_rewards)
        self.logger.info(f"\n[I] Evaluation mean routing score for {max_episodes} episodes: {mean_routing_score:.2f}\n")
        self.logger.info(f"[I] Reached target: {routing_episode_data['Reached_target']} | "
                         f"Optimal routes: {routing_episode_data['Optimal_route']} | "
                         f"Delayed service: {routing_episode_data['Delay_exceeded']} | "
                         f"Terminal states: {routing_episode_data['Terminal_state']}")

        mean_scheduling_score = np.mean(episode_scheduling_rewards)
        self.logger.info(f"\n[I] Evaluation mean scheduling score for {max_episodes} episodes: {mean_scheduling_score:.2f}\n")
        self.logger.info(f"[I] Optimal schedules: {scheduling_episode_data['Optimal_scheduling']} | "
                         f"Non-optimal schedules: {scheduling_episode_data['Non-optimal_scheduling']}")
