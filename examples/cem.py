import numpy as np
import gym
import argparse

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory

parser = argparse.ArgumentParser(description='Cross-entropy method (CEM) with gym environments')
parser.add_argument('--env', type=str, default='CartPole-v0',help="OpenAI Gym Environment")
parser.add_argument('--window_length', type=int, default=4, help="Window lenght")
parser.add_argument('--memory_limit', type=int, default=1000, help="Memory limit")
parser.add_argument('--batch_size', type=int, default=50, help="Batch size of the agent")
parser.add_argument('--nb_steps_warmup', type=int, default=2000, help="Number of warmup steps")
parser.add_argument('--train_interval', type=int, default=50, help="Number of warmup steps")
parser.add_argument('--elite_frac', type=float, default=0.05, help="Elite frac")
parser.add_argument('--nb_steps', type=int, default=100000, help="Number of training steps")
parser.add_argument('--render_train', dest='render_train', action='store_true', help="Either to visualize render or not during training")
parser.add_argument('--verbose', type=int, default=2, help="Level of verbosity of training")
parser.add_argument('--nb_episodes', type=int, default=5, help="Number of episodes to test")
parser.add_argument('--render_test', dest='render_test', action='store_true', help="Either to visualize render or not during testing")
parser.set_defaults(render_train=False)
parser.set_defaults(render_test=False)
args = parser.parse_args()

ENV_NAME = args.env
WINDOW_LENGHT = args.window_length
MEMORY_LIMIT = args.memory_limit
BATCH_SIZE = args.batch_size
NB_STEPS_WARMUP = args.nb_steps_warmup
TRAIN_INTERVAL = args.train_interval
ELITE_FRAC =  args.elite_frac
NB_STEPS = args.nb_steps
VISUALIZE_TRAIN = args.render_train
VERBOSE = args.verbose
NB_EPISODES = args.nb_episodes
VISUALIZE_TEST = args.render_test


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)

nb_actions = env.action_space.n
obs_dim = env.observation_space.shape[0]

# Option 1 : Simple model
model = Sequential()
model.add(Flatten(input_shape=(WINDOW_LENGHT,) + env.observation_space.shape))
model.add(Dense(nb_actions))
model.add(Activation('softmax'))

# Option 2: deep network
# model = Sequential()
# model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(16))
# model.add(Activation('relu'))
# model.add(Dense(nb_actions))
# model.add(Activation('softmax'))


print(model.summary())


# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = EpisodeParameterMemory(limit=MEMORY_LIMIT, window_length=WINDOW_LENGHT)

cem = CEMAgent(model=model, nb_actions=nb_actions, memory=memory,
               batch_size=BATCH_SIZE, nb_steps_warmup=NB_STEPS_WARMUP, train_interval=TRAIN_INTERVAL, elite_frac=ELITE_FRAC)
cem.compile()

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
cem.fit(env, nb_steps=NB_STEPS, visualize=VISUALIZE_TRAIN, verbose=VERBOSE)

# After training is done, we save the best weights.
cem.save_weights('cem_{}_params.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
cem.test(env, nb_episodes=NB_EPISODES, visualize=VISUALIZE_TEST)
