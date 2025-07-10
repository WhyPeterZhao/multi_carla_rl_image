import os
import sys
import time
import random
import numpy as np
import argparse
import logging
import torch
from distutils.util import strtobool
from datetime import datetime
from encoder_init import EncodeState
from networks.on_policy.ppo.agent import PPOAgent
from simulation.connection import ClientConnection
from simulation.environment2 import CarlaEnvironment
from parameters import *
import carla
import threading

def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default='ppo',help='name of the experiment')
    parser.add_argument('--env-name', type=str, default='carla', help='name of the simulation environment')
    parser.add_argument('--learning-rate', type=float, default=PPO_LEARNING_RATE, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=SEED, help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=TOTAL_TIMESTEPS, help='total timesteps of the experiment')
    parser.add_argument('--action-std-init', type=float, default=ACTION_STD_INIT, help='initial exploration noise')
    parser.add_argument('--test-timesteps', type=int, default=TEST_TIMESTEPS, help='timesteps to test our model')
    parser.add_argument('--episode-length', type=int, default=EPISODE_LENGTH, help='max timesteps in an episode')
    parser.add_argument('--train', default=True, type=boolean_string, help='is it training?')
    parser.add_argument('--town', type=str, default="Town02", help='which town do you like?')
    parser.add_argument('--load-checkpoint', type=bool, default=MODEL_LOAD, help='resume training?')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, cuda will not be enabled by deafult')
    args = parser.parse_args()
    
    return args

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'









import redis
import json
import uuid # 用于生成唯一ID
import numpy as np
import base64

POOL = redis.ConnectionPool(host='192.168.3.189', port=6379, db=0, decode_responses=True)
R_CLIENT = redis.Redis(connection_pool=POOL)
TIMEOUT = 5

def policy(obs, id, log=False):
    # 1a & 1b: 创建唯一的回复队列名和关联ID
    reply_key = f"reply:{uuid.uuid4()}"
    correlation_id = str(uuid.uuid4())

    # 1c: 准备要发送的消息
    request_data = {
        "image": {
            "dtype": str(obs[0].dtype),
            "shape": obs[0].shape,
            "data": base64.b64encode(obs[0].tobytes()).decode('utf-8'), 
        },
        "obs1": obs[1].tolist(),
        "reply_to": reply_key,  # 告诉服务器在哪里回复
        "correlation_id": correlation_id # 用于匹配请求和响应
    }
    message = json.dumps(request_data)

    # 1d: 将请求推送到服务器的工作队列
    # 服务器需要监听 'carla_image' 这个列表
    R_CLIENT.lpush("carla_image", message)
    if log:
        print(f"Car {id}: [{correlation_id}] Request sent. Waiting for reply on '{reply_key}'...")

    # 1e: 阻塞等待回复，设置超时
    # BRPOP 返回一个元组 (queue_name, value)
    response = R_CLIENT.brpop(reply_key, timeout=TIMEOUT)
    
    # 3d: 清理临时队列
    R_CLIENT.delete(reply_key)

    if response is None:
        print(f"[{correlation_id}] Error: Timed out after {TIMEOUT} seconds waiting for a reply.")
        raise RuntimeError(f"[{correlation_id}] Error: Timed out after {TIMEOUT} seconds waiting for a reply.")

    # 3a: 解除阻塞，收到了回复
    _, reply_message_raw = response
    reply_message = json.loads(reply_message_raw)
    
    # 3b: 验证关联ID
    if reply_message.get("correlation_id") != correlation_id:
        print(f"[{correlation_id}] Error: Received a reply with mismatched correlation ID.")
        raise RuntimeError(f"[{correlation_id}] Error: Received a reply with mismatched correlation ID.")

    # 3c: 返回最终的动作
    action = reply_message.get("action")
    action = np.array(action)
    if log:
        print(f"Car {id}: [{correlation_id}] Action received: {action}")
    return action

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

    # try:
    #     if exp_name == 'ppo':
    #         run_name = "PPO"
    #     else:
    #         """
            
    #         Here the functionality can be extended to different algorithms.

    #         """ 
    #         sys.exit() 
    # except Exception as e:
    #     print(e.message)
    #     sys.exit()
    
    #Seeding to reproduce the results 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    


    #========================================================================
    #                           CREATING THE SIMULATION
    #========================================================================

    try:
        client, world = ClientConnection(town).setup()
        logging.info("Connection has been setup successfully.")
    except:
        logging.error("Connection has been refused by the server.")
        ConnectionRefusedError

    SECONDS_PER_EPISODE = 45
    FIXED_DELTA_SECONDS = 0.04
    NO_RENDERING = False    
    SYNCHRONOUS_MODE = False
    SPIN = 10
    HEIGHT = 480
    WIDTH = 640

    settings = world.get_settings()
    settings.no_rendering_mode = NO_RENDERING
    settings.synchronous_mode = SYNCHRONOUS_MODE
    settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
    world.apply_settings(settings)

    threads = [] # 创建一个列表来存放所有线程

    # 循环创建并启动所有线程
    for i in range(16):
        print(f"Starting thread for car {i+1}")
        thread = threading.Thread(target=agent, kwargs={"id":i,
                                                        "client":client,
                                                        "world":world,
                                                        "town":town,
                                                        "args":args,
                                                        "action_std_init":action_std_init,
                                                        })
        thread.start()
        threads.append(thread) # 将启动的线程添加到列表中

    # 在所有线程都启动后，再用一个独立的循环来等待它们全部结束
    for i, thread in enumerate(threads):
        thread.join() # 主线程将在此等待每个子线程结束
        print(f"Thread for car {i+1} has finished.")

def agent(id,client,world,town,args,action_std_init):
    timestep = 0
    episode = 0
    distance_covered = 0
    deviation_from_center = 0
    cumulative_score = 0
    episodic_length = list()
    scores = list()
    env = CarlaEnvironment(client, world,town, checkpoint_frequency=None,id=id)
    encode = EncodeState(LATENT_DIM)

    #========================================================================
    #                           ALGORITHM
    #========================================================================

    time.sleep(0.5)
    agent = PPOAgent(town, action_std_init)
    agent.load()
    for params in agent.old_policy.actor.parameters():
        params.requires_grad = False

    # Testing
    while timestep < args.test_timesteps:
        observation = env.reset()
        # observation = encode.process(observation)

        current_ep_reward = 0
        t1 = datetime.now()
        for t in range(args.episode_length):
            # select action with policy

            # action = agent.get_action(observation, train=False)
            action = policy(obs=observation,id=id)

            observation, reward, done, info = env.step(action)
            if observation is None:
                break
            # observation = encode.process(observation)
            
            timestep +=1
            current_ep_reward += reward
            # break; if the episode is over
            if done:
                episode += 1

                t2 = datetime.now()
                t3 = t2-t1
                
                episodic_length.append(abs(t3.total_seconds()))
                break
        deviation_from_center += info[1]
        distance_covered += info[0]
        
        scores.append(current_ep_reward)
        cumulative_score = np.mean(scores)

        print(f'[Agent {id}]','Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score))
        
        episodic_length = list()
        deviation_from_center = 0
        distance_covered = 0

    print("Terminating the run.")



if __name__ == "__main__":
    runner()
