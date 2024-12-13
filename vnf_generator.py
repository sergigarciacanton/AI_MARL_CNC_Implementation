import pickle
import random
from typing import List, Dict, Tuple
from config import EDGES, VNF_LENGTH, VNF_PERIOD, VNF_DELAY, VNF_LISTS_DIRECTORY, VNF_LIST_LENGTH, \
    ROUTE_EVALUATION_EPISODES, CUSTOM_EVALUATION_EPISODES
import networkx as nx


def get_graph(edges: Dict[Tuple[int, int], Dict[str, int]]) -> nx.DiGraph:
    """
    Construct a directed graph from a dictionary of edges.

    Parameters:
        edges (Dict[Tuple[int, int], Dict[str, int]]): A dictionary where keys are tuples representing edges
            (source, target), and values are dictionaries containing edge attributes (e.g., weight).

    Returns:
        nx.DiGraph: A directed graph constructed from the given edges.
    """
    # Create a directed graph
    env_graph: nx.DiGraph = nx.DiGraph()

    # Extract all nodes from the edges dictionary
    all_nodes: set[int] = set(node for edge in edges.keys() for node in edge)

    # Add nodes to the graph
    env_graph.add_nodes_from(all_nodes)  # Directly add existing nodes without assuming consecutive range

    # Add edges with labels and weights
    for edge, data in edges.items():
        source, target = edge
        delay: int = data.get('delay', 1)  # Set default weight to 1 if not provided
        env_graph.add_edge(source, target, delay=delay)

    return env_graph


def get_route_nodes(graph: nx.Graph) -> Tuple[int, int]:
    """
    Randomly select source and target nodes from the graph and ensure there is a valid path between them.

    Parameters:
        graph (nx.Graph): A NetworkX graph object.

    Returns:
        Tuple[int, int]: A tuple representing the source and target nodes.
    """
    while True:
        source_node, target_node = random.sample(list(graph.nodes), 2)
        if nx.has_path(graph, source_node, target_node):
            return source_node, target_node


def generate_vnfs_with_routes(num_samples: int, graph: nx.Graph, all_routes: bool, source: int,
                              destination: int) -> List[Tuple[int, int, int, int, int]]:
    """
    Generate a list of VNFs with consistent randomness for VNFs and variable source/target nodes.

    Parameters:
        num_samples (int): The number of VNF samples to generate.
        graph (nx.Graph): A NetworkX graph object to generate source/target nodes dynamically.
        all_routes (bool): All routes
        source (int): Source node
        destination (int): Destination node

    Returns:
        List[Tuple[int, int, int, int, int]]: A list of VNFs, where each entry is a tuple (source, target, delay).
    """
    vnfs_list: List[Tuple[int, int, int, int, int]] = []

    for _ in range(num_samples):
        # VNF selection is consistent across runs
        length = random.choice(VNF_LENGTH)
        period = random.choice(VNF_PERIOD)
        delay = random.choice(VNF_DELAY)

        # If source and target are provided, use them; otherwise, generate dynamically
        if not all_routes:
            src, tgt = source, destination
        else:
            src, tgt = get_route_nodes(graph=graph)

        # Append the selected VNF with the variable source/target
        vnfs_list.append((src, tgt, length, period, delay))

    print(f"\n{num_samples} VNFs sequence generated\n")
    return vnfs_list


def export_vnfs_pickle(vnfs_list: List[Tuple[int, int, int, int, int]], filename: str) -> None:
    """
    Export the VNF list to a file using Pickle.

    Parameters:
        vnfs_list (List[Tuple[int, int, int, int, int]]): The list of VNFs to export, where each entry is a tuple
            (src, tgt, len, prd, delay).
        filename (str): The name of the file to export to (without extension).
    """
    with open(f"{filename}.pkl", 'wb') as file:
        pickle.dump(vnfs_list, file)
    print(f"Exported VNF list to {filename}.pkl")


def create_vnf_list(evaluate, all_routes, src, dst):
    g = get_graph(EDGES)

    # Export the generated VNF list
    if not evaluate and all_routes:
        vnf_list = generate_vnfs_with_routes(VNF_LIST_LENGTH, g, True, 0, 0)
        export_vnfs_pickle(vnf_list, f'{VNF_LISTS_DIRECTORY}all_routes_{min(g.nodes)}_{max(g.nodes)}')
    elif not evaluate and not all_routes:
        vnf_list = generate_vnfs_with_routes(VNF_LIST_LENGTH, g, False, src, dst)
        export_vnfs_pickle(vnf_list, f'{VNF_LISTS_DIRECTORY}{src}_{dst}')
    elif evaluate and all_routes:
        vnf_list = generate_vnfs_with_routes(CUSTOM_EVALUATION_EPISODES, g, True, 0, 0)
        export_vnfs_pickle(vnf_list, f'{VNF_LISTS_DIRECTORY}all_routes_{min(g.nodes)}_{max(g.nodes)}_evaluate')
    elif evaluate and not all_routes:
        vnf_list = generate_vnfs_with_routes(CUSTOM_EVALUATION_EPISODES, g, False, src, dst)
        export_vnfs_pickle(vnf_list, f'{VNF_LISTS_DIRECTORY}{src}_{dst}_evaluate')


def create_all_routes_vnf_lists(evaluate):
    g = get_graph(EDGES)

    for i in g.nodes:
        for j in g.nodes:
            if i != j:
                if not evaluate:
                    vnf_list = generate_vnfs_with_routes(VNF_LIST_LENGTH, g, False, i, j)
                    export_vnfs_pickle(vnf_list, f'{VNF_LISTS_DIRECTORY}{i}_{j}')
                else:
                    vnf_list = generate_vnfs_with_routes(ROUTE_EVALUATION_EPISODES, g, False, i, j)
                    export_vnfs_pickle(vnf_list, f'{VNF_LISTS_DIRECTORY}{i}_{j}_evaluate')
