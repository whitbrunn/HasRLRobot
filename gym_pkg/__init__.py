import os
import gym
from gym.envs.registration import register

__version__ = "0.1.1"


def envpath():
    resdir = os.path.join(os.path.dirname(__file__))
    return resdir


print("cassie-bipedal-robot-RL: ")
print("|    gym version and path:", gym.__version__, gym.__path__)

print("|    REGISTERING me5418-Cassie-v0 from", envpath())
register(
    id="me5418-Cassie-v0",
    entry_point="gym_pkg.envs:CassieEnv",
    max_episode_steps=1000,
)