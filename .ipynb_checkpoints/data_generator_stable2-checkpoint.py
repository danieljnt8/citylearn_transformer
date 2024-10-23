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

def add_noise(actions,noise,probability_to_add_noise):
    for bi in range(len(actions)):
        if random.uniform(0,1) < probability_to_add_noise:
            actions[bi][0] += random.uniform(-noise, noise)
    return actions


def train_agent(schema, timesteps, seed, additional= None, saved_path = PATH_MODEL_SB, model_str = "PPO"):
    env = CityLearnEnv(schema=schema,reward_function = CombinedReward)
    #reward_func = CustomReward(env)
    #env.reward_function = reward_func
    reward_name = "CombinedReward"
    #env = EnvCityGym(env)
    set_random_seed(seed)
    

    if model_str == "PPO":
        
        env.central_agent = True
        env = NormalizedObservationWrapper(env)
        env = StableBaselines3WrapperCustom(env)
        policy_kwargs = dict(net_arch=dict(pi=[64, 32], vf=[64, 32]))
        model =  PPO("MlpPolicy", env,policy_kwargs = policy_kwargs)
        model.learn(total_timesteps=timesteps)
        data = env.dataset  #for stablebaselines

        type_ = "norm_wrapper"
       

        model_path = PATH_MODEL_SB + "/" + model_str + "/" + "model_{}_timesteps_{}_rf_{}_seed_{}_type_{}".format(model_str,timesteps,reward_name,seed,type_)
        model.save(model_path)

    elif model_str == "SAC":
        
        env.central_agent = True
        env = NormalizedObservationWrapper(env)
        env = StableBaselines3WrapperCustom(env)
        policy_kwargs = dict(net_arch=dict(pi=[64, 32], qf=[64, 32]))
        model =  SAC("MlpPolicy", env,policy_kwargs = policy_kwargs)
        model.learn(total_timesteps=timesteps)
        data = env.dataset  #for stablebaselines

        type_ = "norm_wrapper"
       

        model_path = PATH_MODEL_SB + "/" + model_str + "/" + "model_{}_timesteps_{}_rf_{}_seed_{}_type_{}".format(model_str,timesteps,reward_name,seed,type_)
        model.save(model_path)
    else:
        print("Here")
        env = NormalizedObservationWrapperCustom(env)
        model_str = "RBCAgent1"

        """
        model = RBCAgent1(env)

        model.learn(episodes=11)
        data = model.dataset #for RBC Agent
        """

        agent = RBCAgent1()
        obs,_ = env.reset()
        #print(env.central_agent)
        obs_dict = env_reset(env)
        actions = agent.register_reset(obs_dict)

        count=0
        probability_to_add_noise = 0
        noise = 0.1
        total_timesteps= timesteps

        while True:
            observations,reward,done,_ ,_= env.step(actions)


            
            actions= agent.compute_action(observations)
            
            
            #actions = add_noise(actions,noise,probability_to_add_noise)

            count +=1


            if done:
                print("Here Done")
                probability_to_add_noise +=0.05
                obs,_ = env.reset()
                obs_dict = env_reset(env)
                actions = agent.register_reset(obs_dict)
                #actions = add_noise(actions,noise,probability_to_add_noise)
            
            if count >= total_timesteps:
                break
        
        data = env.dataset


        

    
    data_path = PATH_DATA_INTERACTIONS+ "/" + model_str + "/" + "model_{}_timesteps_{}_rf_{}_phase_2_8760.pkl".format(model_str,timesteps,reward_name)
    #model_path = PATH_MODEL_SB + "/" + model_str + "/" + "model_{}_timesteps_{}_rf_{}_seed_{}".format(model_str,timesteps,reward_name,seed)
    Dataset.from_dict({k: [s[k] for s in data] for k in data[0].keys()}).save_to_disk(data_path)


if __name__ == "__main__":

    schema = "citylearn_challenge_2022_phase_2"

    train_agent(schema,timesteps = 8760,seed =28,model_str= "PPO")

    