#!/bin/bash

# xvfb-run -s "-screen 0 1400x900x24" python3 ddpg.py \
python3 ddpg.py \
  --env FoosballVREP_sp-v1 \
  --window_length 4 \
  --memory_limit 500000 \
  --nb_steps_warmup_actor 10000 \
  --nb_steps_warmup_critic 10000 \
  --nb_train_steps 50000 \
  --verbose 1 \
  --batch_size 32 \
  --actor_hidden_units 16 16 16 \
  --critic_hidden_units 32 32 32 \
  --nb_test_episodes 20 \
  --verbose 2 \
  --render_train
