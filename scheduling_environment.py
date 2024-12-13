import gc
import pickle
import networkx as nx
import numpy as np
from gymnasium.spaces import Discrete, Box
import logging
import sys
from colorlog import ColoredFormatter
import random
from typing import Dict, Tuple
from config import EDGES, ENV_LOG_LEVEL, ENV_LOG_FILE_NAME, VNF_PERIOD, SLOT_CAPACITY, DIVISION_FACTOR, \
    BACKGROUND_STREAMS


def get_graph(edges: Dict[Tuple[int, int], Dict[str, int]]) -> nx.DiGraph:
    """
    Construct a directed graph from a dictionary of edges.

    Parameters:
        edges (Dict[Tuple[int, int], Dict[str, int]]): A dictionary where keys are tuples
            representing edges (source, target), and values are dictionaries containing
            edge attributes (e.g., weight).

    Returns:
        nx.DiGraph: A directed graph constructed from the given edges.
    """
    # Create a directed graph
    graph: nx.DiGraph = nx.DiGraph()

    # Extract all nodes from the edges dictionary
    all_nodes: set[int] = set(node for edge in edges.keys() for node in edge)

    # Add nodes to the graph
    graph.add_nodes_from(range(min(all_nodes), max(all_nodes) + 1))

    # Add edges with labels and weights
    for edge, data in edges.items():
        source, target = edge
        delay: int = data.get('delay', 1)  # Set default weight to 1 if not provided
        graph.add_edge(source, target, delay=delay)

    return graph


# CLASS
class SchedulingEnvironmentTSN:
    def __init__(self, log_file_id):
        self.vnf = None
        self.background_traffic = {}
        self.current_position = None
        self.reward = None
        self.info = {}
        self.vnf_list = []
        self.vnf_list_index = 0

        # Logging settings
        self.logger = logging.getLogger('env')
        self.logger.setLevel(ENV_LOG_LEVEL)
        self.logger.addHandler(
            logging.FileHandler(ENV_LOG_FILE_NAME + log_file_id + '.log', mode='w', encoding='utf-8'))
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(ColoredFormatter('%(log_color)s%(message)s'))
        self.logger.addHandler(stream_handler)

        # Generate graph adding given topology (see config.py)
        self.logger.info('[I] Reading topology from config...')
        self.graph = get_graph(EDGES)
        self.logger.info('[I] Received network topology: ' + str(len(self.graph.nodes)) + ' nodes and '
                         + str(len(self.graph.edges)) + ' edges')

        # Hyperperiod setting. Use maximum period of a given set (see config.py)
        self.hyperperiod = max(VNF_PERIOD)

        self.edge_schedule = [SLOT_CAPACITY] * self.hyperperiod * DIVISION_FACTOR

        # action space
        self.scheduling_action_space = Discrete(self.hyperperiod * DIVISION_FACTOR + 1)
        self.scheduling_state_space = self.get_state_space()

        self.logger.info('[I] Environment ready to operate')

    def import_vnf_list(self, filename: str):
        """
        Import the VNF list from a file using Pickle.

        Parameters:
            filename (str): The name of the file to import (without extension).

        Returns:
            list: The imported list of VNFs.
        """
        with open(f"{filename}", 'rb') as file:
            vnfs_list = pickle.load(file)
        self.logger.info(f"[I] Imported VNF list from {filename}")

        return vnfs_list

    def get_state_space(self):
        # state space
        state_features = {
            'current_position': None
        }
        for n in range(self.hyperperiod * DIVISION_FACTOR):
            state_features['position_status_' + str(n)] = None

        # Define state space using Box
        lo = np.array([0], dtype=np.float32)
        lo = np.append(lo, np.array([-1] * self.hyperperiod * DIVISION_FACTOR, dtype=np.float32))

        hi = np.array([(self.hyperperiod * DIVISION_FACTOR) - 1], dtype=np.float32)
        hi = np.append(hi, np.array([1] * self.hyperperiod * DIVISION_FACTOR, dtype=np.float32))

        return Box(
            low=lo,
            high=hi,
            shape=(len(state_features),),
            dtype=np.float32  # Using float32 for compatibility
        )

    def get_vnf(self):
        vnf = self.vnf_list[self.vnf_list_index]  # vnfs list
        self.vnf_list_index += 1
        return vnf

    def generate_background_traffic(self):
        # Iterate until having created all desired background streams (see config.py)
        num_abortions = 0
        for i in range(BACKGROUND_STREAMS):
            # Create random VNF (see vnf_generator.py) and get the route that will follow (shortest path)
            # VNF = {source, destination, length, period, max_delay, actions}
            self.background_traffic[i] = self.get_vnf()

            # action = (e, t)  e = edge ID  t = position ID
            position = random.randrange(self.background_traffic[i][3])
            first_position = position

            # Schedule action
            # If action is done (return True), pass to next hop
            # If action is impossible, create cent VNF and try to schedule it. If looping too much, stop
            while True:
                if self.schedule_stream(position, self.background_traffic[i][2], self.background_traffic[i][3]):
                    break
                else:
                    position = (position + 1) % self.background_traffic[i][3]
                    if position == first_position:
                        num_abortions += 1
                        i -= 1
            if num_abortions >= 100:
                # If looping too much, network has collapsed. Stop execution
                self.logger.warning(
                    '[!] Background traffic could not be allocated! Ask for less '
                    'background streams (see config.py --> BACKGROUND_STREAMS)')
                break

    # Try to allocate resources for a real stream. If not possible, terminate episode. Called during step
    def schedule_stream(self, position, length, period):
        # Check if scheduling is possible. All time slots of the given position must have enough space
        # If scheduling is possible, assign resources. Otherwise, terminate episode
        if self.get_position_availability(position, period) >= length:
            time_slot = position[1]
            # Loop along all time slots of the position subtracting the requested resources
            while time_slot < self.hyperperiod * DIVISION_FACTOR:
                self.edge_schedule[time_slot] -= length
                time_slot += period
            return True
        else:
            self.logger.debug('[D] Could not schedule the action!')
            return False

    def get_position_availability(self, position, period):
        min_availability = SLOT_CAPACITY
        slot = position
        while slot < self.hyperperiod * DIVISION_FACTOR:
            if self.edge_schedule[slot] < min_availability:
                min_availability = self.edge_schedule[slot]
            slot += period
        return min_availability

    def get_valid_actions(self):
        valid_actions = []
        for n in range(self.vnf[3]):
            if self.get_position_availability(n, self.vnf[3]) >= self.vnf[2]:
                valid_actions.append(n)
        if valid_actions == []:
            valid_actions.append(self.hyperperiod)
        return valid_actions

    def reward_function(self, action):
        # Compute availabilities of all positions
        position_availabilities = []
        for i in range(self.vnf[3]):
            position_availabilities.append(self.get_position_availability(i, self.vnf[3]))

        # Not take into account the load of the current VNF
        position_availabilities[action] += self.vnf[2]
        # print('Current position: ' + str(self.current_position))

        for i in range(self.vnf[3]):
            eval_position = (self.current_position + i) % self.vnf[3]
            # print('Evaluation of position: ' + str(eval_position))
            if self.vnf[2] <= position_availabilities[eval_position]:
                # print('Optimal position: ' + str(eval_position))
                # print('Selected position: ' + str(action[1]))
                if eval_position == action:
                    self.reward += 10  # Positive reward for reaching the target
                    self.info['Optimal_scheduling'] = 1
                else:
                    # print('Different positions found!')
                    if action < eval_position:
                        position = action + self.vnf[3]
                    else:
                        position = action
                    # print('Reward to subtract: ' + str((position - eval_position)))
                    self.reward -= (position - eval_position)
                    self.info['Non-optimal_scheduling'] = 1

                break

    def get_observation(self, valid_actions) -> np.ndarray:
        """
        Retrieve the current state, which includes source, target, VNF, current node, accumulated delay, and edge status

        Returns:
            np.ndarray: The state array containing [src, tgt, vnf, current_node, accumulated_delay, *edge_status]
        """

        edge_status = []
        for n in range(self.vnf[3]):
            if n in valid_actions:
                edge_status.append(1)
            else:
                edge_status.append(-1)

        while len(edge_status) < (self.hyperperiod * DIVISION_FACTOR):
            edge_status.append(-1)

        state = np.array([self.current_position, *edge_status], dtype=np.int16)

        return state

    def reset(self):
        """
        Reset the environment for the next episode.

        Returns:
            Tuple[np.ndarray, Dict]: The initial state and additional information after reset.
        """
        self.vnf = self.get_vnf()
        self.logger.debug('[D] VNF: ' + str(self.vnf))

        self.reward = 0

        self.edge_schedule = [SLOT_CAPACITY] * self.hyperperiod * DIVISION_FACTOR

        # Generate background traffic
        self.generate_background_traffic()
        self.logger.info('[I] Generated ' + str(len(self.background_traffic)) + ' background streams')

        self.current_position = random.randrange(self.vnf[3])

        # Get valid actions
        valid_actions = self.get_valid_actions()

        # Get the initial state
        state = self.get_observation(valid_actions)

        self.info = {'valid_actions': valid_actions,
                     'Optimal_scheduling': 0,
                     'Non-optimal_scheduling': 0
                     }

        self.logger.debug('[D] RESET. obs = ' + str(state) + ', info = ' + str(self.info))
        return state, self.info

    def step(self, action: int):
        """
        Perform a step in the environment by applying the given action (move to a new node).

        Args:
            action (int): The node to move to.

        Returns:
            Tuple containing:
                - state (np.ndarray): The updated state of the environment.
                - reward (float): The reward for the current step.
                - terminated (bool): Placeholder (True).
                - truncated (bool): Placeholder (False).
                - info (Dict): Additional information, such as valid actions.
        """

        self.logger.debug('[I] STEP. Action: ' + str(action))

        if action < self.hyperperiod:
            self.schedule_stream(action, self.vnf[2], self.vnf[3])

            # Reward function
            self.reward_function(action)

        # Get neighbors of the action and filter out the previous node when applying the action
        valid_actions = self.get_valid_actions()

        # Update the state
        state = self.get_observation(valid_actions)

        # Prepare additional info (e.g., valid actions from the new node)
        self.info['valid_actions'] = valid_actions

        self.logger.debug('[D] STEP. obs = ' + str(state) + ', reward = ' + str(self.reward) +
                          ', info = ' + str(self.info))

        # Return the updated state, reward, termination status, truncated flag (always False), and info
        return state, self.reward, True, False, self.info

    def close_env(self):
        """
        Clean up resources and free memory used by the environment.
        """
        # Example: Reset large data structures
        self.graph.clear()  # Clear the graph to free memory

        # Optional: Call garbage collection to free up unused memory immediately
        gc.collect()

        print("\nEnvironment resources cleaned up and memory freed.")
