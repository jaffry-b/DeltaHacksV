## Base model to run the game, using random movements
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
from aux import *
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)

done = True
oldi = {'coins': 0, 'flag_get': False, 'life': 2, 'score': 0, 'stage': 1, 'status': 'small', 'time': 400, 'world': 1, 'x_pos': 40}
for step in range(100):
    if done:
        state = env.reset()
    state, rwd, done, info = env.step(1)#env.action_space.sample())
    print(reward(info,oldi), "vs", rwd)
    print(env.observation_space.shape)
    oldi = info
    env.render()

env.close()
