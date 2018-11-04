#!/bin/bash

# python3 ddpg.py \
xvfb-run -s "-screen 0 1400x900x24" python3 ddpg.py \
  --env FoosballVREP_sp-v1 \
  --window_length 4 \
  --memory_limit 1000000 \
  --nb_steps_warmup_actor 0 \
  --nb_steps_warmup_critic 0 \
  --nb_train_steps 100000 \
  --verbose 1 \
  --batch_size 32 \
  --nb_test_episodes 0 \
  --load_memory
