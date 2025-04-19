## About
This is the Avt-v5 env but with rough terrain (height field). Edits should be done in your conda environment python gymnasium; full paths for my current working version are given below.

## How to use

in `/home/jblevins32/anaconda3/envs/RL/lib/python3.9/site-packages/gymnasium/envs/mujoco/assets`

add files: `ant_hfield.xml`, `hfield.png`

in `/home/jblevins32/anaconda3/envs/RL/lib/python3.9/site-packages/gymnasium/envs/mujoco`

add files: `ant_v5_hfield.py`

in `/home/jblevins32/anaconda3/envs/RL/lib/python3.9/site-packages/gymnasium/envs`

replace `__init__.py` with this folder''s `__init__.py`

Now you can use env `Ant-v5-hfield`
