#!/bin/bash

python3 naf.py \
  --env FoosballVREP_sp-v1 \
  --window_length 1 \
  --memory_limit 100000 \
  --nb_steps_warmup 1000 \
  --nb_max_episode_steps 400 \
  --nb_steps 500000 \
  --verbose 2 \
  --nb_episodes 5 
