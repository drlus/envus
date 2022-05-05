# Core Library
import logging

# Third party
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(id="SmibDAE-v0", entry_point="smib.envs:SmibDAE")