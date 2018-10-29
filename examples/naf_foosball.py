import numpy as np
import gym
import gym_foosball

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor

import matplotlib.pyplot as plt

class PendulumProcessor(Processor):
    def process_reward(self, reward):
        # The magnitude of the reward can be important. Since each step yields a relatively
        # high reward, we reduce the magnitude by two orders.
        return reward / 100.


ENV_NAME = 'FoosballVREP_sp-v0'
WINDOW_LENGTH = 4
gym.undo_logger_setup()


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Build all necessary models: V, mu, and L networks.
V_model = Sequential()
V_model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
V_model.add(Dense(32))
V_model.add(Activation('relu'))
V_model.add(Dense(32))
V_model.add(Activation('relu'))
V_model.add(Dense(32))
V_model.add(Activation('relu'))
V_model.add(Dense(1))
V_model.add(Activation('linear'))
print(V_model.summary())

mu_model = Sequential()
mu_model.add(Flatten(input_shape=(WINDOW_LENGTH,) + env.observation_space.shape))
mu_model.add(Dense(32))
mu_model.add(Activation('relu'))
mu_model.add(Dense(32))
mu_model.add(Activation('relu'))
mu_model.add(Dense(32))
mu_model.add(Activation('relu'))
mu_model.add(Dense(nb_actions))
mu_model.add(Activation('linear'))
print(mu_model.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(WINDOW_LENGTH,) + env.observation_space.shape, name='observation_input')
x = Concatenate()([action_input, Flatten()(observation_input)])
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
x = Activation('linear')(x)
L_model = Model(inputs=[action_input, observation_input], outputs=x)
print(L_model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
processor = PendulumProcessor()
memory = SequentialMemory(limit=100000, window_length=WINDOW_LENGTH)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
agent = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                 memory=memory, nb_steps_warmup=100, random_process=random_process,
                 gamma=.99, target_model_update=1e-3, processor=processor)
agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
history = agent.fit(env, nb_steps=50000, visualize=True, verbose=2, nb_max_episode_steps=200)

# Plot results
plt.plot(history.history['episode_reward'])
plt.title('Model reward')
plt.ylabel('Reward')
plt.xlabel('Episode')
# plt.show()
plt.savefig('naf_{}_rewards.png'.format(ENV_NAME))

# After training is done, we save the final weights.
agent.save_weights('naf_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
directory = 'videos/foosball/naf'
env = gym.wrappers.Monitor(env, directory, force=True, video_callable=lambda episode_id: True)
agent.test(env, nb_episodes=10, visualize=True, nb_max_episode_steps=200)