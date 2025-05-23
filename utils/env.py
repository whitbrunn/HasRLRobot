import os
import time
import torch
import numpy as np
import gym
from gym.envs.registration import register
import gym

if 'me5418-Cassie-v0' not in gym.envs.registry.env_specs:
    register(
        id='me5418-Cassie-v0',
        entry_point='gym_pkg.envs:CassieEnv'
    )

def env_factory(path, **kwargs):
    from functools import partial

    """
    Returns an *uninstantiated* environment constructor.

    Since environments containing cpointers (e.g. Mujoco envs) can't be serialized,
    this allows us to pass their constructors to Ray remote functions instead
    (since the gym registry isn't shared across ray subprocesses we can't simply
    pass gym.make() either)

    Note: env.unwrapped.spec is never set, if that matters for some reason.
    """

    # Custom Cassie Environment


    spec = gym.envs.registry.spec(path)
    _kwargs = spec._kwargs.copy()
    _kwargs.update(kwargs)

    try:
        if callable(spec._entry_point):
            cls = spec._entry_point(**_kwargs)
        else:
            cls = gym.envs.registration.load(spec._entry_point)
    except AttributeError:
        if callable(spec.entry_point):
            cls = spec.entry_point(**_kwargs)
        else:
            cls = gym.envs.registration.load(spec.entry_point)

    return partial(cls, **_kwargs)
