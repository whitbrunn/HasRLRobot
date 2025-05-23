import numpy as np

# Gives a vectorized interface to a single environment
class WrapEnv:
    def __init__(self, env_fn):
        self.env = env_fn()

    def __getattr__(self, attr):
        return getattr(self.env, attr)

    def step(self, action, term_thresh=0):
        # print(f"\nHere is wrapper, {action.shape}.\n")
        state, reward, done, info = self.env.step(action, f_term=term_thresh)
        return np.array([state]), np.array([reward]), np.array([done]), np.array([info])

    def render(self):
        self.env.render()

    def reset(self):
        return np.array([self.env.reset()])