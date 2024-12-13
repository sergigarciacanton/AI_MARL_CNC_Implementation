import gc
import pickle
import random

import networkx as nx
import itertools
import numpy as np
from gymnasium.spaces import Discrete, Box
import logging
import sys
from colorlog import ColoredFormatter
from typing import Dict, Tuple, List
from config import EDGES, VNF_DELAY, ENV_LOG_LEVEL, ENV_LOG_FILE_NAME, VNF_PERIOD, SLOT_CAPACITY, \
    DIVISION_FACTOR, BACKGROUND_STREAMS


def display_all_routes(graph):
    """ Helper method to see all routes"""
    i = 0  # Counter for all routes
    weights = []
    for source, target in itertools.permutations(graph.nodes,
                                                 2):  # Permutations to cover all source-target pairs
        for route in nx.all_simple_paths(graph, source, target):
            weights.append(nx.path_weight(graph, route, weight='weight'))
            print(f"{i}: {route} - {nx.path_weight(graph, route, weight='weight')}")
            i += 1  # Increment for each route found
    print(f"Total routes: {i} - max route delay: {max(weights)}")


def all_short_paths_between_nodes(graph, src, tgt):
    for i, route in enumerate(nx.all_shortest_paths(graph, source=src, target=tgt, weight='weight')):
        print(f"{i}: {route}")


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
class RoutingEnvironmentTSN:
    def __init__(self, log_file_id):
        self.vnf = None
        self.route = None
        self.background_traffic = {}
        self.edges_info = {}
        self.current_node = None
        self.accumulated_delay = None
        self.reward = None
        self.terminated = False
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

        # Create edges info. Contains available bytes of each edge
        for edge, delay in EDGES.items():
            self.edges_info[(edge[0], edge[1])] = SLOT_CAPACITY * self.hyperperiod * DIVISION_FACTOR

        # action space
        self.routing_action_space = Discrete(len(self.graph.nodes) + 1)
        self.routing_state_space = self.get_state_space()

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
            'source': None,
            'target': None,
            'current_n': None,
            'vnf': None,
            'accumulate_delay': None,
        }
        for n in self.graph.nodes:
            state_features['edge_status_' + str(n)] = None

        # Get max edge delays robustly
        max_edge_delay = max(data.get('delay', float('-inf')) for _, _, data in self.graph.edges(data=True))

        # Get max node value from the action space
        min_node_value = min(self.graph.nodes)
        max_node_value = max(self.graph.nodes)

        # Handle VNFS list
        min_vnf_delay = min(VNF_DELAY) if VNF_DELAY else 0  # Assuming VNFS is non-empty
        max_vnf_delay = max(VNF_DELAY) if VNF_DELAY else 0

        # Define state space using Box
        lo = np.array([min_node_value, min_node_value, min_node_value, min_vnf_delay, 0], dtype=np.float32)
        lo = np.append(lo, np.array([-1] * len(self.graph.nodes), dtype=np.float32))

        hi = np.array([max_node_value, max_node_value, max_node_value, max_vnf_delay, max_vnf_delay], dtype=np.float32)
        hi = np.append(hi, np.array([max_edge_delay] * len(self.graph.nodes), dtype=np.float32))

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

    # Background traffic generator. Called during reset
    def generate_background_traffic(self):
        # Iterate until having created all desired background streams (see config.py)
        num_abortions = 0
        for i in range(BACKGROUND_STREAMS):
            # Create random VNF (see vnf_generator.py) and get the route that will follow (shortest path)
            # VNF = {source, destination, length, period, max_delay, actions}
            self.background_traffic[i] = self.get_vnf()
            path = random.choice(list(nx.all_shortest_paths(self.graph,
                                                            source=self.background_traffic[i][0],
                                                            target=self.background_traffic[i][1])))

            # Iterate along the path nodes until having assigned resources at all intermediate edges
            for j in range(len(path) - 1):
                # Schedule action
                # If action is done (return True), pass to next hop
                # If action is impossible, create cent VNF and try to schedule it. If looping too much, stop
                if not self.schedule_stream((path[j], path[j + 1]),
                                            self.background_traffic[i][2],
                                            self.background_traffic[i][3]):
                    num_abortions += 1
                    i -= 1
                    if num_abortions >= 1000:
                        # If looping too much, network has collapsed. Stop execution
                        self.logger.warning(
                            '[!] Background traffic could not be allocated! Ask for less '
                            'background streams (see config.py --> BACKGROUND_STREAMS)')
                        return
                    break

    # Try to allocate resources for a real stream. If not possible, terminate episode. Called during step
    def schedule_stream(self, edge, length, period):
        # Check if scheduling is possible. All time slots of the given position must have enough space
        # If scheduling is possible, assign resources. Otherwise, terminate episode
        if self.edges_info[edge] >= (length * (self.hyperperiod / period)):
            self.edges_info[edge] -= (length * (self.hyperperiod / period))
            return True
        else:
            self.terminated = True
            self.logger.debug('[D] Could not schedule the action!')
            return False

    def get_edges_status(self, valid_actions) -> List[int]:
        """
        Retrieve the weight values of edges connected to the current node, sorted by neighboring node values.

        Returns:
            List[int]: A list of edge weight values for the edges connected to the current node.
                       The list is sorted by the neighboring node values.
        """

        edge_status = []
        for n in self.graph.nodes:
            if n in valid_actions:
                edge_delay = nx.path_weight(self.graph, [self.current_node, n], weight='delay')
                edge_status.append(edge_delay)
            else:
                edge_status.append(-1)
        return edge_status

    def get_valid_actions(self):
        valid_actions = []
        neighbors = list(nx.neighbors(self.graph, self.current_node))
        request = (self.vnf[2] * (self.hyperperiod / self.vnf[3]))
        if len(self.route) < 2:
            for n in self.graph.nodes:
                if n in neighbors and self.edges_info[(self.current_node, n)] >= request:
                    valid_actions.append(n)
        else:
            for n in self.graph.nodes:
                if n in neighbors and n != self.route[-2] and self.edges_info[(self.current_node, n)] >= request:
                    valid_actions.append(n)
        if valid_actions == []:
            valid_actions.append(max(self.graph.nodes) + 1)
        return valid_actions

    def reward_function(self, action, action_weight):
        # Validate the action based on the accumulated delay
        if self.accumulated_delay > self.vnf[4]:
            self.reward += -50  # Penalize for exceeding allowed delay
            self.terminated = True  # End episode if delay exceeds limit
            self.info['Delay_exceeded'] = 1

        else:
            # Negative reward based on path weight (delay)
            self.reward += -action_weight

            # Check if the agent has reached the target node
            if action == self.vnf[1]:  # If reached the target (tgt)
                self.reward += 50  # Positive reward for reaching the target
                self.info['Reached_target'] = 1
                self.logger.info('[I] Reached destination!')
                self.terminated = True

                # Bonus if the route matches one of the shortest paths
                if self.accumulated_delay == nx.dijkstra_path_length(self.graph, self.vnf[0], self.vnf[1], 'delay'):
                    self.reward += 100
                    self.info['Optimal_route'] = 1

    def get_observation(self, valid_actions) -> np.ndarray:
        """
        Retrieve the current state, which includes source, target, VNF, current node, accumulated delay, and edge status

        Returns:
            np.ndarray: The state array containing [src, tgt, vnf, current_node, accumulated_delay, *edge_status]
        """
        src, tgt, length, prd, delay = self.vnf

        # Get the edge status
        edge_status = self.get_edges_status(valid_actions)

        state = np.array([src, tgt, self.current_node, delay, self.accumulated_delay, *edge_status], dtype=np.int16)

        return state

    def reset(self):
        """
        Reset the environment for the next episode.

        Returns:
            Tuple[np.ndarray, Dict]: The initial state and additional information after reset.
        """
        self.vnf = self.get_vnf()
        self.logger.debug('[D] VNF: ' + str(self.vnf))

        self.accumulated_delay = 0
        self.reward = 0
        self.route = [self.vnf[0]]
        self.current_node = self.vnf[0]
        self.terminated = False

        # Create edges info. Contains available bytes of each edge
        for edge, delay in EDGES.items():
            self.edges_info[(edge[0], edge[1])] = SLOT_CAPACITY * self.hyperperiod * DIVISION_FACTOR

        # Generate background traffic
        self.generate_background_traffic()
        self.logger.info('[I] Generated ' + str(len(self.background_traffic)) + ' background streams')

        # Get valid actions
        valid_actions = self.get_valid_actions()

        # Get the initial state
        state = self.get_observation(valid_actions)

        self.info = {'valid_actions': valid_actions,
                     'Delay_exceeded': 0,
                     'Optimal_route': 0,
                     'Reached_target': 0,
                     'Terminal_state': 0
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
                - terminated (bool): Whether the episode has terminated.
                - truncated (bool): Placeholder (False).
                - info (Dict): Additional information, such as valid actions.
        """

        self.logger.debug('[I] STEP. Action: ' + str(action))

        if action > max(self.graph.nodes):
            self.terminated = True
            self.info['Terminal_state'] = 1
        else:
            # Compute the weight of the action (moving from current_node to action)
            action_weight = nx.path_weight(self.graph, [self.current_node, action], weight='delay')

            self.schedule_stream((self.current_node, action), self.vnf[2], self.vnf[3])

            # Update accumulated delay and route
            self.accumulated_delay += action_weight
            self.route.append(action)
            self.current_node = self.route[-1]

            # Reward function
            self.reward_function(action, action_weight)

        # Get neighbors of the action and filter out the previous node when applying the action
        valid_actions = self.get_valid_actions()

        # Update the state
        state = self.get_observation(valid_actions)

        # Prepare additional info (e.g., valid actions from the new node)
        self.info['valid_actions'] = valid_actions

        self.logger.debug('[D] STEP. obs = ' + str(state) + ', terminated = ' + str(self.terminated) +
                          ', reward = ' + str(self.reward) +
                          ', info = ' + str(self.info))
        if self.terminated:
            self.logger.info('[I] Ending episode...\n')

        # Return the updated state, reward, termination status, truncated flag (always False), and info
        return state, self.reward, self.terminated, False, self.info

    def close_env(self):
        """
                Clean up resources and free memory used by the environment.
                """
        # Example: Reset large data structures
        self.graph.clear()  # Clear the graph to free memory
        self.route = None  # Reset the route

        # Optional: Call garbage collection to free up unused memory immediately
        gc.collect()

        print("\nEnvironment resources cleaned up and memory freed.")
