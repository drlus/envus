# Core Library
import logging

# Third party
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(id="RL-v0", entry_point="RLcircuit.envs:RLcircuitEnv", max_episode_steps = 100)