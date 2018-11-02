import argparse


parser = argparse.ArgumentParser(description='Deep Deterministic Policy Gradients (DDPG) with gym environments')
parser.add_argument('--env', type=str, default='Pendulum-v0', help="OpenAI Gym Environment")
parser.add_argument('--batch_size', type=int, default=32, help="Size of the batch of memory used to train")
parser.add_argument('--gamma', type=float, default=.99, help="Discount factor during train")
parser.add_argument('--learning_rate', type=float, default=.001, help="Learning rate")
parser.add_argument('--memory_limit', type=int, default=100000, help="Max number of steps stored in memory")
parser.add_argument('--nb_max_test_episode_steps', type=int, default=200, help="Max number of steps for each episode during test")
parser.add_argument('--nb_max_train_episode_steps', type=int, default=200, help="Max number of steps for each episode during train")
parser.add_argument('--nb_steps_warmup_actor', type=int, default=100, help="Number of warmup steps for actor")
parser.add_argument('--nb_steps_warmup_critic', type=int, default=100, help="Number of warmup steps for critic")
parser.add_argument('--nb_test_episodes', type=int, default=5, help="Number of episodes during test")
parser.add_argument('--nb_train_steps', type=int, default=50000, help="Number of training steps")
parser.add_argument('--render_test', dest='render_test', action='store_true', help="Either to render the environment during train or not")
parser.add_argument('--render_train', dest='render_train', action='store_true', help="Either to render the environment during test or not")
parser.add_argument('--target_model_update', type=float, default=1e-3, help="Target model update")
parser.add_argument('--verbose', type=int, default=1, help="Level of verbosity")
parser.add_argument('--window_length', type=int, default=1, help="How much consecutive frames should be passed to the agent")
parser.add_argument('--weights_checkpoint_interval', type=int, default=10000, help="Interval (expressed as number of timesteps) after which weights must be stored")
parser.add_argument('--actor_hidden_units', nargs='*', type=int, default=[16, 16, 16], help="Number of units for each hidden layer of actor NN")
parser.add_argument('--critic_hidden_units', nargs='*', type=int, default=[32, 32, 32], help="Number of units for each hidden layer of critic NN")
parser.set_defaults(render_test=False)
parser.set_defaults(render_train=False)

args = parser.parse_args()

ENV_NAME = args.env
BATCH_SIZE = args.batch_size
GAMMA = args.gamma
LEARNING_RATE = args.learning_rate
MEMORY_LIMIT = args.memory_limit
NB_MAX_TEST_EPISODE_STEPS = args.nb_max_test_episode_steps
NB_MAX_TRAIN_EPISODE_STEPS = args.nb_max_train_episode_steps
NB_STEPS_WARMUP_ACTOR = args.nb_steps_warmup_actor
NB_STEPS_WARMUP_CRITIC = args.nb_steps_warmup_critic
NB_TEST_EPISODES = args.nb_test_episodes
NB_TRAIN_STEPS = args.nb_train_steps
RENDER_TEST = args.render_test
RENDER_TRAIN = args.render_train
TARGET_MODEL_UPDATE = args.target_model_update
VERBOSE = args.verbose
WINDOW_LENGHT = args.window_length
WEIGHTS_CHECKPOINT_INTERVAL = args.weights_checkpoint_interval
ACTOR_HIDDEN_UNITS = args.actor_hidden_units
CRITIC_HIDDEN_UNITS = args.critic_hidden_units

LOG_FILEPATH = 'ddpg_{}_log.json'.format(ENV_NAME)
WEIGHTS_FILEPATH = 'ddpg_backup_weights_{step}.h5f'


import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
from rl.core import Processor


class LunarLanderContinuousProcessor(Processor):
    def __init__(self, env):
        self.env = env
        super().__init__()

    def process_action(self, action):
        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)
        return action

# Get the environment and extract the number of actions.
env = gym.make(ENV_NAME)
np.random.seed(123)
env.seed(123)
assert len(env.action_space.shape) == 1
nb_actions = env.action_space.shape[0]

# Some environments need to be processed
if ENV_NAME == 'LunarLanderContinuous-v2':
    processor = LunarLanderContinuousProcessor(env)
else:
    processor = Processor()

# Next, we build a very simple model.
actor = Sequential()
actor.add(Flatten(input_shape=(WINDOW_LENGHT,) + env.observation_space.shape))
for i in range(len(ACTOR_HIDDEN_UNITS)):
    actor.add(Dense(ACTOR_HIDDEN_UNITS[i]))
    actor.add(Activation('relu'))
actor.add(Dense(nb_actions))
actor.add(Activation('linear'))
print(actor.summary())

action_input = Input(shape=(nb_actions,), name='action_input')
observation_input = Input(shape=(WINDOW_LENGHT,) + env.observation_space.shape, name='observation_input')
flattened_observation = Flatten()(observation_input)
x = Concatenate()([action_input, flattened_observation])
for i in range(len(CRITIC_HIDDEN_UNITS)):
    x = Dense(CRITIC_HIDDEN_UNITS[i])(x)
    x = Activation('relu')(x)
x = Dense(1)(x)
x = Activation('linear')(x)
critic = Model(inputs=[action_input, observation_input], outputs=x)
print(critic.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=MEMORY_LIMIT, window_length=WINDOW_LENGHT)
random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
agent = DDPGAgent(nb_actions=nb_actions, actor=actor, critic=critic,
                  critic_action_input=action_input, memory=memory,
                  nb_steps_warmup_critic=NB_STEPS_WARMUP_CRITIC,
                  nb_steps_warmup_actor=NB_STEPS_WARMUP_ACTOR,
                  random_process=random_process, gamma=GAMMA,
                  target_model_update=TARGET_MODEL_UPDATE, processor=processor,
                  batch_size=BATCH_SIZE)
agent.compile(Adam(lr=LEARNING_RATE, clipnorm=1.), metrics=['mae'])

# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.
callbacks = [FileLogger(LOG_FILEPATH, interval=1)]
callbacks += [ModelIntervalCheckpoint(WEIGHTS_FILEPATH, interval=WEIGHTS_CHECKPOINT_INTERVAL, verbose=1)]
agent.fit(env, nb_steps=NB_TRAIN_STEPS, visualize=RENDER_TRAIN, verbose=VERBOSE, nb_max_episode_steps=NB_MAX_TRAIN_EPISODE_STEPS, callbacks=callbacks)

# After training is done, we save the final weights.
agent.save_weights('ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
agent.test(env, nb_episodes=NB_TEST_EPISODES, visualize=RENDER_TEST, nb_max_episode_steps=NB_MAX_TEST_EPISODE_STEPS)
