import numpy as np
import gym
import gym_foosball

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents import SARSAAgent
from rl.policy import BoltzmannQPolicy

import matplotlib.pyplot as plt


ENV_NAME = 'DiscreteFoosballVREP_sp-v0'
WINDOW_LENGTH = 1


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# SARSA does not require a memory.
policy = BoltzmannQPolicy()
sarsa = SARSAAgent(model=model, nb_actions=nb_actions, nb_steps_warmup=100, policy=policy)
sarsa.compile(Adam(lr=1e-3), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
history = sarsa.fit(env, nb_steps=50000, visualize=False, verbose=1)

# Plot results
plt.plot(history.history['episode_reward'])
plt.title('Model reward')
plt.ylabel('Reward')
plt.xlabel('Episode')
# plt.show()
plt.savefig('sarsa_{}_rewards.png'.format(ENV_NAME))

# After training is done, we save the final weights.
sarsa.save_weights('sarsa_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
directory = 'videos/foosball/sarsa'
env = gym.wrappers.Monitor(env, directory, force=True, video_callable=lambda episode_id: True)
sarsa.test(env, nb_episodes=5, visualize=True)
