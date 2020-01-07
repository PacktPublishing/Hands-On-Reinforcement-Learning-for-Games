import numpy as np
from minipacman import MiniPacman

import matplotlib.pyplot as plt

def displayImage(image, step, reward):
    s = "step" + str(step) + " reward " + str(reward)
    plt.title(s)
    plt.imshow(image)
    plt.show()

keys = {
    'w': 2,
    'd': 1,
    'a': 3,
    's': 4,
    ' ': 0
}

MODES = ('regular', 'avoid', 'hunt', 'ambush', 'rush')
frame_cap = 1000

mode = 'rush'

env = MiniPacman(mode, 1000)

state = env.reset()
done = False

total_reward = 0
step = 1

displayImage(state.transpose(1, 2, 0), step, total_reward)

while not done:
    #x = raw_input()
    #clear_output()
    try:
        keys[x]
    except:
        print("Only 'w' 'a' 'd' 's'")
        continue
    action = keys[x]
    
    next_state, reward, done, _ = env.step(action)
    total_reward += reward
    displayImage(next_state.transpose(1, 2, 0), step, total_reward)
    step += 1