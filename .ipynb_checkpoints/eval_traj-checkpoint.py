import os
import torch
import argparse
import numpy as np

from tqdm.auto import trange
from omegaconf import OmegaConf

from stable_baselines3.common.vec_env import DummyVecEnv

from trajectory.models.gpt import GPT
from trajectory.utils.common import set_seed
from trajectory.utils.env import create_env, rollout, vec_rollout
from trajectory.utils.CityStableEnv import *
import gym

from citylearn.citylearn import CityLearnEnv
import itertools

from tqdm.auto import trange
from trajectory.planning.beam import beam_plan, batch_beam_plan

def eval(checkpoints_path , config, run_config, device,schema = "citylearn_challenge_2022_phase_2"):
    env = CityLearnEnv(schema=schema)
    env = EnvCityGym(env)

    beam_context_size = 3 #config.beam_context 5
    beam_width = 10 #config.beam_width 32
    beam_steps = 2 #config.beam_steps 5
    plan_every = config.plan_every
    sample_expand = config.sample_expand
    k_act = config.k_act
    k_obs = config.k_obs
    k_reward = config.k_reward
    temperature = config.temperature
    discount = 0.9  #config.discount
    max_steps = 5

    value_placeholder = 1e6

    discretizer = torch.load(os.path.join(checkpoints_path, "discretizer.pt"), map_location=device)

    model = GPT(**run_config.model)
    model.eval()
    model.to(device)

    model.load_state_dict(torch.load(os.path.join(checkpoints_path, "model_last.pt"), map_location=device))


    transition_dim, obs_dim, act_dim = model.transition_dim, model.observation_dim, model.action_dim
    # trajectory of tokens for model action planning
    # +1 just to avoid index error while updating context on the last step
    context = torch.zeros(1, model.transition_dim * (max_steps + 1), dtype=torch.long).to(device)

    obs = env.reset()
    obs = np.array(obs)
    total_reward = 0
    obs_tokens = discretizer.encode(obs, subslice=(0, obs_dim)).squeeze()

    context[:, :model.observation_dim] = torch.as_tensor(obs_tokens, device=device)  # initial tokens for planning

    for step in (pbar := trange(max_steps, desc="Rollout steps", leave=False)):
    #print(step)
        if step % plan_every == 0:
                # removing zeros from future, keep only context updated so far
            context_offset = model.transition_dim * (step + 1) - model.action_dim - 2
                # prediction: [a, s, a, s, ...]
            prediction_tokens = beam_plan(
                    model, discretizer, context[:, :context_offset],
                    steps=beam_steps,
                    beam_width=beam_width,
                    context_size=beam_context_size,
                    k_act=k_act, k_obs=k_obs, k_reward=k_reward,
                    temperature=temperature,
                    discount=discount,
                    sample_expand=sample_expand
                )
        else:
                # shift one transition forward in plan
            prediction_tokens = prediction_tokens[transition_dim:]
        action_tokens = prediction_tokens[:act_dim]
        action = discretizer.decode(action_tokens.cpu().numpy(), subslice=(obs_dim, obs_dim + act_dim)).squeeze()
        obs, reward, done, _ = env.step(action)
        obs = np.array(obs)
        
        total_reward +=reward
        
        pbar.set_postfix({'Reward': total_reward})
        if done:
            break
            
        obs_tokens = discretizer.encode(obs, subslice=(0, obs_dim)).squeeze()
        reward_tokens = discretizer.encode(
                np.array([reward, value_placeholder]),
                subslice=(transition_dim - 2, transition_dim)
            )
        # updating context with new action and obs
        context_offset = model.transition_dim * step
            # [s, ...] -> [s, a, ...]
        context[:, context_offset + obs_dim:context_offset + obs_dim + act_dim] = torch.as_tensor(action_tokens, device=device)
            # [s, a, ...] -> [s, a, r, v, ...]
        context[:, context_offset + transition_dim - 2:context_offset + transition_dim] = torch.as_tensor(reward_tokens, device=device)
            # [s, a, r, v, ...] -> [s, a, r, v, s, ...]
        context[:, context_offset + model.transition_dim:context_offset + model.transition_dim + model.observation_dim] = torch.as_tensor(obs_tokens, device=device)
    
    df_evaluate = env.env.evaluate()
    df_evaluate.to_csv(checkpoints_path + '/results.csv')
    

if __name__ == '__main__':
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    checkpoints_path =  "checkpoints/city_learn/traj_config_v2/quantile/baseline"

    run_config = "configs/medium/city_learn_traj_v2.yaml"
    run_config = OmegaConf.load(run_config)

    config = "configs/eval_base.yaml"
    config = OmegaConf.load(config)

 