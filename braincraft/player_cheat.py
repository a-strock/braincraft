# Braincraft challenge — 1000 neurons, 100 seconds, 10 runs, 2 choices, no reward
# Copyright (C) 2025 Nicolas P. Rougier
# Released under the GNU General Public License 3
"""
Example and evaluation of the performances of a random player.
"""
from bot import Bot
from environment_1 import Environment

old_env_init = Environment.__post_init__
def cheat_env_init(self):
    old_env_init(self)
    Environment.last_instance = self
Environment.__post_init__ = cheat_env_init

old_bot_init = Bot.__post_init__
def cheat_bot_init(self):
    old_bot_init(self)
    env = Environment.last_instance
    idx = np.where(env.world == env.source.identity)
    cell_size = 1/max(env.world.shape)
    x,y = (np.mean(idx[1])+0.5)*cell_size, (np.mean(idx[0])+0.5)*cell_size # TODO: Fix for not square
    self.position = (x,y)
    self.rotation_max = 360
Bot.__post_init__ = cheat_bot_init

def cheat_player():
    """Random players building"""

    env = Environment()
    bot = Bot()

    # Fixed parameters
    n = 1000
    p = bot.camera.resolution
    warmup = 0
    f = np.tanh
    g = np.tanh
    leak = 1.0

    Win  = np.random.uniform(-1,1, (n,p+3))
    W = np.random.uniform(-1,1, (n,n))*(np.random.uniform(0,1, (n,n)) < 0.1)
    Wout = 0.1*np.random.uniform(-1, 1, (1,n))
    yield Win, W, Wout, warmup, leak, f, g


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    import numpy as np    
    from challenge import train, evaluate

    seed = 12345
    
    # Training (100 seconds)
    np.random.seed(seed)
    print(f"Starting training for 100 seconds (user time)")
    model = train(cheat_player, timeout=100)

    # Evaluation
    start_time = time.time()
    score, std = evaluate(model, Bot, Environment, debug=False, seed=seed)
    #score, std = evaluate(model, Bot, Environment, debug=True, seed=seed)
    elapsed = time.time() - start_time
    print(f"Evaluation completed after {elapsed:.2f} seconds")
    print(f"Final score: {score:.2f} ± {std:.2f}")

