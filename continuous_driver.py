import os
import sys
import time
import random
import numpy as np
import argparse
import logging
import pickle
import torch
from distutils.util import strtobool
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from encoder_init import EncodeState
from networks.on_policy.ppo.agent import PPOAgent
from simulation.connection import ClientConnection
from simulation.multi_environment import CarlaEnvironment
from parameters import *

cars = 2

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, help='name of the experiment')
    parser.add_argument('--env-name', type=str, default='carla', help='name of the simulation environment')
    parser.add_argument('--learning-rate', type=float, default=PPO_LEARNING_RATE, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=SEED, help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=TOTAL_TIMESTEPS, help='total timesteps of the experiment')
    parser.add_argument('--action-std-init', type=float, default=ACTION_STD_INIT, help='initial exploration noise')
    parser.add_argument('--test-timesteps', type=int, default=TEST_TIMESTEPS, help='timesteps to test our model')
    parser.add_argument('--episode-length', type=int, default=EPISODE_LENGTH, help='max timesteps in an episode')
    parser.add_argument('--train', default=True, type=boolean_string, help='is it training?')
    parser.add_argument('--town', type=str, default="Town07", help='which town do you like?')
    parser.add_argument('--load-checkpoint', type=bool, default=MODEL_LOAD, help='resume training?')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, cuda will not be enabled by deafult')
    args = parser.parse_args()
    
    return args

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'



def runner():

    #========================================================================
    #                           BASIC PARAMETER & LOGGING SETUP
    #========================================================================
    
    args = parse_args()
    exp_name = args.exp_name
    train = args.train
    town = args.town
    checkpoint_load = args.load_checkpoint
    total_timesteps = args.total_timesteps
    action_std_init = args.action_std_init

    try:
        if exp_name == 'ppo':
            run_name = "PPO"
        else:
            """
            
            Here the functionality can be extended to different algorithms.

            """ 
            sys.exit() 
    except Exception as e:
        print(e.message)
        sys.exit()
    
    if train == True:
        writer = SummaryWriter(f"runs/{run_name}_{action_std_init}_{int(total_timesteps)}/{town}")
    else:
        writer = SummaryWriter(f"runs/{run_name}_{action_std_init}_{int(total_timesteps)}_TEST/{town}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])))


    #Seeding to reproduce the results 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    
    action_std_decay_rate = 0.05
    min_action_std = 0.05   
    action_std_decay_freq = 5e5
    timestep = 0
    episode = [0]*cars
    cumulative_score = 0
    episodic_length = [list()]*cars
    scores = [list()]*cars
    deviation_from_center = 0
    distance_covered = 0

    #========================================================================
    #                           CREATING THE SIMULATION
    #========================================================================

    try:
        client, world = ClientConnection(town).setup()
        logging.info("Connection has been setup successfully.")
    except:
        logging.error("Connection has been refused by the server.")
        ConnectionRefusedError
    if train:
        env = CarlaEnvironment(client, world,town)
    else:
        env = CarlaEnvironment(client, world,town, checkpoint_frequency=None, cars=cars)
    encode = EncodeState(LATENT_DIM)


    #========================================================================
    #                           ALGORITHM
    #========================================================================

    time.sleep(0.5)
    agent = PPOAgent(town, action_std_init)
    if train:
        pass
    else:
        #Testing
        while timestep < args.test_timesteps:
            observations = env.reset_all()
            print("reset done!")
            encoded_observations = []
            for o in observations:
                o = encode.process(o)
                encoded_observations.append(o)

            current_ep_reward = [0]*cars
            t1 = datetime.now()
            for t in range(args.episode_length):
                # select action with policy
                action = []
                for o in encoded_observations:
                    a = agent.get_action(o, train=False)
                    action.append(a)

                observations, rewards, dones, infos = env.step(action)
                print("dones: ", dones)
                # if observation is None:
                #     break
                encoded_observations = []
                for o in observations:
                    o = encode.process(o)
                    encoded_observations.append(o)
                # observation = encode.process(observation)
                
                timestep +=1
                for number in range(cars):
                    current_ep_reward[number] += rewards[number]
                # break; if the episode is over
                
                    if dones[number]:
                        episode[number] += 1

                        t2 = datetime.now()
                        t3 = t2-t1
                        
                        episodic_length[number].append(abs(t3.total_seconds()))
                        # break
            # deviation_from_center += info[1]
            # distance_covered += info[0]
            
                        scores[number].append(current_ep_reward)
                        cumulative_score = np.mean(scores[number])

                        print(f"car{number}: ",f"Episode: {episode[number]}", f"Timestep: {timestep}", f"Reward: {current_ep_reward[number]:.2f}", f"Average Reward: {cumulative_score:.2f}")
                        # print('car {}'.format(number),'Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score))
            

            # episodic_length = list()
            # deviation_from_center = 0
            # distance_covered = 0

        print("Terminating the run.")
        sys.exit()




if __name__ == "__main__":      
    runner()

