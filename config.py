# Logging settings
ENV_LOG_LEVEL = 40  # Levels: DEBUG 10 | INFO 20 | WARN 30 | ERROR 40 | CRITICAL 50
ENV_LOG_FILE_NAME = 'logs/env_'  # Name of file where to store all env generated logs. Make sure the directory exists
RAINBOW_LOG_LEVEL = 20  # Levels: DEBUG 10 | INFO 20 | WARN 30 | ERROR 40 | CRITICAL 50
RAINBOW_LOG_FILE_NAME = 'logs/dqn_'  # Name of file where to store all agent generated logs. Make sure directory exists

# Environment settings
TRAINING_STEPS = 500000  # Number of time-steps to run for training an agent
CUSTOM_EVALUATION_EPISODES = 1000  # Number of episodes to run for a custom evaluation.
ROUTE_EVALUATION_EPISODES = 100  # Number of episodes to run for each route of a multi-route evaluation
MONITOR_TRAINING = 10000  # Number of time-steps to count for logging training results
SLOT_CAPACITY = 12500  # Number of bytes that can be transmitted at each time slot
DIVISION_FACTOR = 1  # Number of slots to define per millisecond
BACKGROUND_STREAMS = 300  # Number of streams to create as background traffic. Their scheduling is prefixed
CUSTOM_ROUTE = (2, 5)  # Custom route used for evaluating models
ALL_ROUTES = True  # Flag that is set in case of wanting to train or evaluate for all routes of the graph.
#                    If set to False, use custom route defined at CUSTOM_ROUTE variables

# VNF generator settings
VNF_LISTS_DIRECTORY = 'vnf_lists/'
VNF_LIST_LENGTH = 50000000
VNF_LENGTH = [128, 256, 512, 1024, 1500]  # List of the possible lengths of packets to generate in random VNFs
VNF_DELAY = [20, 23, 26, 29]  # List of possible delay bounds to generate in random VNFs
VNF_PERIOD = [2, 4, 8, 16]  # List of possible periods to generate in random VNFs
#                             Must ALWAYS be set (maximum value is used as hyperperiod)

# Agent settings
MODEL_PATH = "models/rainbow/cent/"  # Path where models will be stored. Make sure that the directory exists!
SEED = None  # 1976  # Seed used for randomization purposes
REPLAY_BUFFER_SIZE = 1000000  # Hyperparameter for DQN agent
BATCH_SIZE = 32  # Hyperparameter for DQN agent
TARGET_UPDATE = 1000  # Hyperparameter for DQN agent
GAMMA = 0.99  # Hyperparameter for DQN agent
LEARNING_RATE = 0.0001  # Hyperparameter for DQN agent
TAU = 0.005  # Hyperparameter for DQN agent
ALPHA = 0.6
BETA = 0.4
PRIOR_EPS = 1e-6
N_STEP = 3
V_MIN = -60
V_MAX = 150
ATOM_SIZE = 51

# Plotting settings
SAVE_PLOTS = True
PLOTS_PATH = 'plots/rainbow/cent/'

# Topology settings
# EDGES = {
#     (0, 1): {'delay': 1},
#     (1, 0): {'delay': 1},
#     (1, 3): {'delay': 2},
#     (3, 1): {'delay': 2},
#     (3, 2): {'delay': 3},
#     (2, 3): {'delay': 3},
#     (2, 0): {'delay': 4},
#     (0, 2): {'delay': 4},
#     (0, 4): {'delay': 3},
#     (4, 0): {'delay': 3},
#     (1, 5): {'delay': 5},
#     (5, 1): {'delay': 5},
#     (3, 7): {'delay': 7},
#     (7, 3): {'delay': 7},
#     (2, 6): {'delay': 1},
#     (6, 2): {'delay': 1},
#     (4, 5): {'delay': 4},
#     (5, 4): {'delay': 4},
#     (5, 7): {'delay': 1},
#     (7, 5): {'delay': 1},
#     (7, 6): {'delay': 2},
#     (6, 7): {'delay': 2},
#     (6, 4): {'delay': 3},
#     (4, 6): {'delay': 3}
# }  # List of edges in the TSN network

EDGES = {
    (0, 1): {'delay': 1},
    (1, 0): {'delay': 1},
    (0, 2): {'delay': 2},
    (2, 0): {'delay': 2},
    (0, 5): {'delay': 3},
    (5, 0): {'delay': 3},
    (1, 4): {'delay': 4},
    (4, 1): {'delay': 4},
    (1, 8): {'delay': 1},
    (8, 1): {'delay': 1},
    (2, 3): {'delay': 2},
    (3, 2): {'delay': 2},
    (2, 5): {'delay': 3},
    (5, 2): {'delay': 3},
    (3, 4): {'delay': 4},
    (4, 3): {'delay': 4},
    (3, 6): {'delay': 1},
    (6, 3): {'delay': 1},
    (4, 8): {'delay': 2},
    (8, 4): {'delay': 2},
    (5, 9): {'delay': 3},
    (9, 5): {'delay': 3},
    (6, 7): {'delay': 4},
    (7, 6): {'delay': 4},
    (6, 9): {'delay': 1},
    (9, 6): {'delay': 1},
    (7, 10): {'delay': 2},
    (10, 7): {'delay': 2},
    (7, 11): {'delay': 3},
    (11, 7): {'delay': 3},
    (8, 12): {'delay': 4},
    (12, 8): {'delay': 4},
    (11, 12): {'delay': 1},
    (12, 11): {'delay': 1},
    (9, 10): {'delay': 2},
    (10, 9): {'delay': 2},
    (10, 11): {'delay': 3},
    (11, 10): {'delay': 3},
}
