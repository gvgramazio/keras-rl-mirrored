#!/bin/bash

xvfb-run -s "-screen 0 1400x900x24" python3 naf.py \
  --env FoosballVREP_sp-v1 \
  --window_length 4 \
  --memory_limit 500000 \
  --nb_steps_warmup 10000 \
  --nb_max_episode_steps 400 \
  --nb_steps 500000 \
  --verbose 2 \
  --nb_episodes 5 \
  --batch_size 512
