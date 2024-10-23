import os
from typing import Generic, Optional, SupportsFloat, Tuple, TypeVar, Union,Any
import torch
# import couple of libs some will be useful
import gym
import numpy as np
from collections import deque
import random
import re
import os
import sys
import time
import json
import torch as th
import itertools
from datasets import Dataset
from _code.const import PATH_MODEL_SB,PATH_DATA_INTERACTIONS
from citylearn.agents.rbc import BasicRBC as BRBC
from agents.rbc import RBCAgent1 
# import stable_baselines3
from stable_baselines3 import PPO, A2C, DDPG, TD3,SAC
from stable_baselines3.common.utils import set_random_seed
import citylearn
from citylearn.citylearn import CityLearnEnv
from pathlib import Path
from rewards.combined_reward import CombinedReward
from stable_baselines3.common.utils import set_random_seed
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

import functools
from citylearn.wrappers import *
from utils.wrappers_custom import *


schema = "citylearn_challenge_2022_phase_2"
env = CityLearnEnv(schema=schema,reward_function = CombinedReward)
env = NormalizedObservationWrapperCustom(env)
        


agent = RBCAgent1()
obs,_ = env.reset()
        #print(env.central_agent)
obs_dict = env_reset(env)
actions = agent.register_reset(obs_dict)


while True:
    observations,reward,done,_ ,_= env.step(actions)

    


            
    actions= agent.compute_action(observations)
            
            
  

        


    if done:
        print("Here Done")
              
        obs,_ = env.reset()
        obs_dict = env_reset(env)
        actions = agent.register_reset(obs_dict)
        
            
        df_evaluate = env.evaluate()

        df_evaluate.to_csv("current_rbc.csv")
        
        