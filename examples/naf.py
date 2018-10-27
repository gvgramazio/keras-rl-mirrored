import numpy as np
import gym
import gym_foosball
import argparse

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import NAFAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.core import Processor
from rl.callbacks import FileLogger

parser = argparse.ArgumentParser(description='Continuous DQN (NAF) with gym environments')
parser.add_argument('--env', type=str, default='Pendulum-v0',help="OpenAI Gym Environment")
parser.add_argument('--window_length', type=int, default=1, help="Window lenght")
parser.add_argument('--memory_limit', type=int, default=100000, help="Memory limit")
parser.add_argument('--nb_steps_warmup', type=int, default=100, help="Number of warmup steps")
parser.add_argument('--gamma', type=float, default=.99, help="Discount factor")
parser.add_argument('--target_model_update', type=float, default=1e-3, help="Target model update")
parser.add_argument('--nb_max_episode_steps', type=int, default=200, help="Maximum number of steps for each episode")
parser.add_argument('--nb_steps', type=int, default=50000, help="Number of training steps")
parser.add_argument('--render_train', dest='render_train', action='store_true', help="Either to visualize render or not during training")
parser.add_argument('--verbose', type=int, default=2, help="Level of verbosity of training")
parser.add_argument('--nb_episodes', type=int, default=5, help="Number of episodes to test")
parser.add_argument('--render_test', dest='render_test', action='store_true', help="Either to visualize render or not during testing")
parser.add_argument('--load_weights', dest='load_weights', action='store_true', help="Either to load previous weights or not before training")
parser.set_defaults(render_train=False)
parser.set_defaults(render_test=False)
parser.set_defaults(load_weights=False)
args = parser.parse_args()

ENV_NAME = args.env
WINDOW_LENGHT = args.window_length
MEMORY_LIMIT = args.memory_limit
NB_STEPS_WARMUP = args.nb_steps_warmup
GAMMA = args.gamma
TARGET_MODEL_UPDATE = args.target_model_update
NB_MAX_EPISODE_STEPS = args.nb_max_episode_steps
NB_STEPS = args.nb_steps
VISUALIZE_TRAIN = args.render_train
VERBOSE = args.verbose
NB_EPISODES = args.nb_episodes
VISUALIZE_TEST = args.render_test
LOAD_WEIGHTS = args.load_weights


class FoosballProcessor(Processor):
    def process_observation(self, observation):
        return observation * 10.


# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Build all necessary models: V, mu, and L networks.
V_model = Sequential()
V_model.add(Flatten(input_shape=(WINDOW_LENGHT,) + env.observation_space.shape))
V_model.add(Dense(64))
V_model.add(Activation('relu'))
V_model.add(Dense(64))
V_model.add(Activation('relu'))
V_model.add(Dense(32))
V_model.add(Activation('relu'))
V_model.add(Dense(1))
V_model.add(Activation('linear'))
print(V_model.summary())

mu_model = Sequential()
mu_model.add(Flatten(input_shape=(WINDOW_LENGHT,) + env.observation_space.shape))
mu_model.add(Dense(64))
mu_model.add(Activation('relu'))
mu_model.add(Dense(64))
mu_model.add(Activation('relu'))
mu_model.add(Dense(32))
mu_model.add(Activation('relu'))
mu_model.add(Dense(nb_actions))
mu_model.add(Activation('linear'))
print(mu_model.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(WINDOW_LENGHT,) + env.observation_space.shape, name='observation_input')
x = Concatenate()([action_input, Flatten()(observation_input)])
x = Dense(128)(x)
x = Activation('relu')(x)
x = Dense(128)(x)
x = Activation('relu')(x)
x = Dense(64)(x)
x = Activation('relu')(x)
x = Dense(((nb_actions * nb_actions + nb_actions) // 2))(x)
x = Activation('linear')(x)
L_model = Model(inputs=[action_input, observation_input], outputs=x)
print(L_model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
if ENV_NAME == 'FoosballVREP_sp-v1':
    processor = FoosballProcessor()
else:
    processor = Processor()
memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=WINDOW_LENGHT)
random_process = OrnsteinUhlenbeckProcess(theta=.15, mu=0., sigma=.3, size=nb_actions)
agent = NAFAgent(nb_actions=nb_actions, V_model=V_model, L_model=L_model, mu_model=mu_model,
                 memory=memory, nb_steps_warmup=NB_STEPS_WARMUP, random_process=random_process,
                 gamma=GAMMA, target_model_update=TARGET_MODEL_UPDATE, processor=processor)
agent.compile(Adam(lr=.0001, clipnorm=1.), metrics=['mae'])

if LOAD_WEIGHTS:
    agent.load_weights('cdqn_{}_weights.h5f'.format(ENV_NAME))

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
log_filename = 'naf_{}_log.json'.format(ENV_NAME)
callbacks = [FileLogger(log_filename, interval=1)]
history = agent.fit(
    env,
    nb_steps=NB_STEPS,
    visualize=VISUALIZE_TRAIN,
    verbose=VERBOSE,
    nb_max_episode_steps=NB_MAX_EPISODE_STEPS,
    callbacks=callbacks)

# After training is done, we save the final weights.
agent.save_weights('cdqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=NB_EPISODES, visualize=VISUALIZE_TRAIN, nb_max_episode_steps=NB_MAX_EPISODE_STEPS)
