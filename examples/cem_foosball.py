import numpy as np
import gym
import gym_foosball

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory

import matplotlib.pyplot as plt

ENV_NAME = 'DiscreteFoosballVREP_sp-v0'
WINDOW_LENGTH = 4


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

# Option 1 : Simple model
# model = Sequential()
# model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
# model.add(Dense(nb_actions))
# model.add(Activation('softmax'))
# print(model.summary())

# Option 2: deep network
model = Sequential()
model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))
print(model.summary())


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = EpisodeParameterMemory(limit=1000, window_length=WINDOW_LENGTH)

cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory,
               batch_size=50, nb_steps_warmup=2000, train_interval=50, elite_frac=0.05)
cem.compile()
cem.load_weights('cem_{}_params.h5f'.format(ENV_NAME))

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
history = cem.fit(env, nb_steps=50000, visualize=False, verbose=1)

# Plot results
plt.plot(history.history['episode_reward'])
plt.title('Model reward')
plt.ylabel('Reward')
plt.xlabel('Episode')
# plt.show()
plt.savefig('cem_{}_rewards.png'.format(ENV_NAME))

# After training is done, we save the best weights.
cem.save_weights('cem_{}_params.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
directory = 'videos/foosball/cem'
env = gym.wrappers.Monitor(env, directory, force=True, video_callable=lambda episode_id: True)
cem.test(env, nb_episodes=5, visualize=True)
