import argparse
import math
from collections import namedtuple
from itertools import count
import numpy as np
from eval import eval_model_q
import copy
import torch
import torch.nn.functional as F
from ddpg_vec import DDPG
import random

from replay_memory import ReplayMemory, Transition
from utils import *
import os
import time
from utils import get_n_actions, copy_actor_policy, average_model_weights
from ddpg_vec import hard_update
import torch.multiprocessing as mp
from multiprocessing import Queue
from multiprocessing.sharedctypes import Value
import sys
import utils
import json
from models import bandit
from torch.optim import Adam

def gen_config():
    parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
    parser.add_argument('--scenario', type=str, default='simple_coop_push_n30',
                        help='name of the environment to run')
    parser.add_argument('--gamma', type=float, default=0.95, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.01, metavar='G',
                        help='discount factor for model (default: 0.001)')
    parser.add_argument('--ou_noise', type=bool, default=True)
    parser.add_argument('--param_noise', type=bool, default=False)
    parser.add_argument('--train_noise', default=False, action='store_true')
    parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                        help='initial noise scale (default: 0.3)')
    parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                        help='final noise scale (default: 0.3)')
    parser.add_argument('--exploration_end', type=int, default=40000, metavar='N',  #60000
                        help='number of episodes with noise (default: 100)')
    parser.add_argument('--seed', type=int, default=9, metavar='N',
                        help='random seed (default: 4)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 128)')
    parser.add_argument('--num_steps', type=int, default=25, metavar='N',
                        help='max episode length (default: 1000)')
    parser.add_argument('--num_episodes', type=int, default=40000, metavar='N',   #60000
                        help='number of episodes (default: 1000)')
    parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                        help='number of episodes (default: 128)')
    parser.add_argument('--updates_per_step', type=int, default=8, metavar='N',
                        help='model updates per simulator step (default: 5)')
    parser.add_argument('--critic_updates_per_step', type=int, default=8, metavar='N',
                        help='model updates per simulator step (default: 5)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 1000000)')
    parser.add_argument('--actor_lr', type=float, default=1e-2,
                        help='(default: 1e-4)')
    parser.add_argument('--critic_lr', type=float, default=1e-2,
                        help='(default: 1e-3)')
    parser.add_argument('--fixed_lr', default=False, action='store_true')
    parser.add_argument('--num_eval_runs', type=int, default=100, help='number of runs per evaluation (default: 5)')
    parser.add_argument("--exp_name", type=str, help="name of the experiment")
    parser.add_argument("--save_dir", type=str, default="./ckpt_plot",
                        help="directory in which training state and model should be saved")
    parser.add_argument('--static_env', default=False, action='store_true')
    parser.add_argument('--critic_type', type=str, default='mlp', help="Supports [mlp, gcn_mean, gcn_max]")
    parser.add_argument('--actor_type', type=str, default='mlp', help="Supports [mlp, gcn_max]")
    parser.add_argument('--critic_dec_cen', default='dec')
    parser.add_argument("--env_agent_ckpt", type=str, default='ckpt_plot/simple_tag_v5_al0a10_4/agents.ckpt')
    parser.add_argument('--shuffle', default=None, type=str, help='None|shuffle|sort')
    parser.add_argument('--episode_per_update', type=int, default=4, metavar='N',
                        help='max episode length (default: 1000)')
    parser.add_argument('--episode_per_actor_update', type=int, default=4)
    parser.add_argument('--episode_per_critic_update', type=int, default=4)
    parser.add_argument('--steps_per_actor_update', type=int, default=100)
    parser.add_argument('--steps_per_critic_update', type=int, default=100)
    #parser.add_argument('--episodes_per_update', type=int, default=4)
    parser.add_argument('--target_update_mode', default='soft', help='soft | hard')
    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--eval_freq', type=int, default=1000) #1000

    #c_net_type
    parser.add_argument('--pos_bias', type=bool, default=False)
    
    parser.add_argument('--temperature', type=float, default=1)
    parser.add_argument('--non_avg_prob', type=float, default=0)
    parser.add_argument('--bandit_lr', type=float, default=0.2)
    parser.add_argument('--c_net_lr', type=float, default=1e-3,
                        help='(default: 1e-3)')
    parser.add_argument('--avg_freq', type=int, default=1)
    parser.add_argument('--c_net_type', type=str, default='MADDPG') #transformer, nn, Independent, MADDPG, random, 'hyper'
    parser.add_argument('--e_net_type', type=str, default='gcn_max') #mlp,
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--c_threshold', type=float, default=0.5)
    parser.add_argument('--g_loss_type', type=str, default='CLAMP') #CLAMP, L1, L2
    parser.add_argument('--degree', type=int, default=1)
    parser.add_argument('--w_graph_type', type=str, default='random') #complete, random, never
    parser.add_argument('--avg_policy', type=bool, default=True)
    parser.add_argument('--avg_c_net', type=bool, default=True)
    parser.add_argument('--same_init_weights', type=bool, default=True)

    args = parser.parse_args()
    config = vars(args)
    return config

def main(config):

    if config['exp_name'] is None:
        config['exp_name'] = config['scenario'] + '_' + config['critic_type'] + '_' + config['target_update_mode'] + '_hiddensize' \
                        + str(config['hidden_size']) + '_' + str(config['seed'])

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() and config['cuda'] else "cpu")

    env = make_env(config['scenario'], None)
    #env = make_modified_env(config['scenario'], None)
    n_others = int(config['scenario'].split('_')[-1])
    n_agents = env.n
    env.seed(config['seed'])
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    num_adversary = 0

    if config['c_net_type'] == 'MADDPG':
        if config['w_graph_type'] not in ['complete','never']:
            tag = config['scenario'] + ',all_one,' + str(config['w_graph_type']) + ",degree:" + str(config['degree']) + "freq:" + str(config['avg_freq']) + ',seed:' + str(config['seed'])
        else:
            tag = config['scenario'] + ',all_one,' + str(config['w_graph_type']) + "freq:" + str(config['avg_freq']) + ',seed:' + str(config['seed'])
    elif config['c_net_type'] == 'Independent':
        if config['w_graph_type'] not in ['complete','never']:
            tag = config['scenario'] + ',all_zero,' + str(config['w_graph_type']) + ",degree:" + str(config['degree']) + "freq:" + str(config['avg_freq']) + ',seed:' + str(config['seed'])
        else:
            tag = config['scenario'] + ',all_zero,' + str(config['w_graph_type']) + "freq:" + str(config['avg_freq']) + ',seed:' + str(config['seed'])
    elif config['c_net_type'] in ['random','transformer','GCN','hyper','farthest','nearest']: #random, transformer, 'GCN', 'hyper'
        temp = ',' + str(config['alpha']) + ',' + str(config['c_threshold']) + ',' + config['g_loss_type'] + ',' if config['c_net_type'] in ['transformer','GCN','hyper'] else ''
        if config['w_graph_type'] not in ['complete','never']:
            tag = config['scenario'] + ',{},'.format(config['c_net_type']) + temp + str(config['w_graph_type']) + ",degree:" + str(config['degree']) + "freq:" + str(config['avg_freq']) + ',seed:' + str(config['seed'])
        else:
            tag = config['scenario'] + ',{},'.format(config['c_net_type']) + temp + str(config['w_graph_type']) + "freq:" + str(config['avg_freq']) + ',seed:' + str(config['seed'])
    else:
        raise NotImplementedError()

    if config['c_net_type'] in ['transformer','GCN','hyper']:
        tag = tag + 'temperature:'+ str(config['temperature'])

    if config['avg_policy']:
        tag='avg_policy'+tag
    else:
        tag='not_avg_policy'+tag

    if config['avg_c_net']:
        tag='avg_c_net'+tag
    else:
        tag='not_avg_c_net'+tag

    if config['same_init_weights']:
        tag = 'same_initial_weight' + tag
    else:
        tag = 'diff_initial_weight' + tag
    #tag = 'alpha:' + str(config['alpha']) + ',' + 'e_net_type:' + config['e_net_type'] + ',' + tag
    tag += ',c_net_lr:' + str(config['c_net_lr'])
    config['save_dir'] = tag
    with open('./config.json', 'w') as fp:
        json.dump(config, fp) 
    c_logger = utils.get_logger(tag = tag, dir = 'c_tf_log')
    g_logger = utils.get_logger(tag = tag, dir = 'g_tf_log')

    n_actions = get_n_actions(env.action_space)
    #n_actions = modified_get_n_actions(env.action_space)

    temp_obs_n = env.reset()
    obs_dims = [len(temp_obs_n[i][0]) for i in range(n_agents)]
    obs_dims.insert(0, 0)
    
    agent = DDPG(config['gamma'], config['tau'], config['hidden_size'],
                obs_dims[1], n_actions[0], n_agents, n_others, obs_dims, 0,
                config['actor_lr'], config['critic_lr'], config['c_net_lr'],
                config['fixed_lr'], config['critic_type'], config['actor_type'], config['train_noise'], config['num_episodes'],
                config['num_steps'], config['critic_dec_cen'], config['target_update_mode'], device, config['c_net_type'], config['e_net_type'], config['alpha'], config['c_threshold'], config['g_loss_type'], config['w_graph_type'], config['same_init_weights'], config['temperature'], config['pos_bias'])
    eval_agent = DDPG(config['gamma'], config['tau'], config['hidden_size'],
                obs_dims[1], n_actions[0], n_agents, n_others, obs_dims, 0,
                config['actor_lr'], config['critic_lr'], config['c_net_lr'],
                config['fixed_lr'], config['critic_type'], config['actor_type'], config['train_noise'], config['num_episodes'],
                config['num_steps'], config['critic_dec_cen'], config['target_update_mode'], 'cpu', config['c_net_type'], config['e_net_type'], config['alpha'], config['c_threshold'], config['g_loss_type'], config['w_graph_type'], config['same_init_weights'], config['temperature'], config['pos_bias'])

    #bandit arms for weight sharing
    if config['w_graph_type'] == 'bandit':
        bandit_policy_list = [bandit.ExpWeights(agent_id=i, num_arms = n_agents-1, lr = config['bandit_lr']) for i in range(n_agents)]
        no_w_avg_rates = []
        level_2_arms = []
    
    memory = ReplayMemory(config['replay_size'])
    feat_dims = []
    for i in range(n_agents):
        feat_dims.append(env.observation_space[i].shape[0])

    # Find main agents index
    unique_dims = list(set(feat_dims))
    agents0 = [i for i, feat_dim in enumerate(feat_dims) if feat_dim == unique_dims[0]]
    if len(unique_dims) > 1:
        agents1 = [i for i, feat_dim in enumerate(feat_dims) if feat_dim == unique_dims[1]]
        main_agents = agents0 if len(agents0) >= len(agents1) else agents1
    else:
        main_agents = agents0
    total_numsteps = 0
    updates = 0
    exp_save_dir = './'
    os.makedirs(exp_save_dir, exist_ok=True)
    best_eval_reward, best_good_eval_reward, best_adversary_eval_reward = -1000000000, -1000000000, -1000000000
    start_time = time.time()
    copy_actor_policy(agent, eval_agent)
    if not os.path.exists('./model_weights'):
        os.makedirs('./model_weights')
    torch.save({'agents': eval_agent}, os.path.join('./model_weights', 'agents_' + str(0) + '.ckpt'))
    plot = {'good_rewards': [], 'adversary_rewards': [], 'rewards': [], 'steps': [], 'q_loss': [], 'gcn_q_loss': [],
        'p_loss': [], 'final': [], 'abs': []}

    total_steps = 0
    prev_episode_reward = 0

    if config['c_net_type'] in ['transformer','GCN','hyper']:
        gate_open_rates = []
        gate_open_dist = []
    for i_episode in range(config['num_episodes']):
        obs_n = env.reset()
        obs_n, indices, positions = split_obs(obs_n)
        episode_reward = 0
        episode_step = 0
        agents_rew = [[] for _ in range(n_agents)]
        avg_gate = []
        avg_gate_dist = []
        
        if config['w_graph_type'] == 'complete':
            weight_average_graph = utils.get_complete_graph(n_agents)
        elif config['w_graph_type'] == 'random':
            weight_average_graph = utils.get_random_graph(n_agents, config['degree'], config['non_avg_prob'])
        elif config['w_graph_type'] == 'never':
            weight_average_graph = utils.get_identity_graph(n_agents)
        elif config['w_graph_type'] == 'bandit':
            weight_average_graph = []
            no_w_avg_rate = []
            level_2_arm = []
            for bandit_policy in bandit_policy_list:
                graph, no_w_avg, arm2 = bandit_policy.sample()
                weight_average_graph.append(graph)
                no_w_avg_rate.append(no_w_avg)
                level_2_arm.append(arm2)
            no_w_avg_rates.append(np.mean(no_w_avg_rate))
            level_2_arms.append(level_2_arm)
        else:
            raise NotImplementedError()
        utils.average_agent_weights(agent, weight_average_graph, config['avg_policy'], config['avg_c_net'])

        while True:
            total_steps += 1
            action_n = []
            for id in range(len(obs_n)):
                action_n.append(agent.select_action(id, torch.Tensor(obs_n[id]).view(1,-1).to(device), action_noise=True,
                                           param_noise=False).squeeze().cpu().numpy())

            next_obs_n, reward_n, done_n, info = env.step(action_n)
            next_obs_n, next_indices, next_positions = split_obs(next_obs_n)
            total_numsteps += 1
            episode_step += 1
            terminal = (episode_step >= config['num_steps'])
            action = torch.Tensor(action_n).view(1, -1)
            mask = torch.Tensor([[not done for done in done_n]])
            next_x = torch.Tensor(np.concatenate(next_obs_n, axis=0)).view(1, -1)
            reward = torch.Tensor([reward_n])
            x = torch.Tensor(np.concatenate(obs_n, axis=0)).view(1, -1)
            temp_indices = torch.Tensor(np.concatenate(indices, axis=0)).view(1, -1)
            temp_next_indices = torch.Tensor(np.concatenate(next_indices, axis=0)).view(1, -1)

            temp_positions = torch.Tensor(np.concatenate(positions, axis=0)).view(1, -1)
            temp_next_positions = torch.Tensor(np.concatenate(next_positions, axis=0)).view(1, -1)

            memory.push(x, temp_indices, temp_positions, action, mask, next_x, temp_next_indices, temp_next_positions, reward)
            for i, r in enumerate(reward_n):
                agents_rew[i].append(r)
            episode_reward += np.sum(reward_n)
            obs_n = next_obs_n
            indices = next_indices
            positions = next_positions
            n_update_iter = 5
            if done_n[0] or terminal:
                episode_step = 0
                break
        if len(memory) > config['batch_size']:
            policy_loss = g_loss  = 0
            for id in range(len(obs_n)):
                transitions = memory.sample(config['batch_size'])
                batch = Transition(*zip(*transitions))
                p_loss_i, g_loss_i, avg_gate_i, avg_gate_dist_i = agent.update_actor_parameters(id, batch, n_agents, config['shuffle'])
                policy_loss += p_loss_i
                g_loss += g_loss_i
                avg_gate.append(avg_gate_i)
                avg_gate_dist.append(avg_gate_dist_i)
            gate_loss = g_loss / len(obs_n)

            policy_loss = policy_loss / len(obs_n)
            updates += 1
            print('episode {}, p loss {}, p_lr {}'.format(i_episode, policy_loss, agent.actor_lr, flush = True))
            
            v_loss = perturb_out = unclipped_norm = 0
            for id in range(len(obs_n)):
                transitions = memory.sample(config['batch_size'])
                batch = Transition(*zip(*transitions))
                v_loss_i, perturb_out_i, unclipped_norm_i = agent.update_critic_parameters(id, batch, config['shuffle'])
                v_loss += v_loss_i
                perturb_out += perturb_out_i
                unclipped_norm += unclipped_norm_i
    
            updates += 1
            value_loss = (v_loss + perturb_out + unclipped_norm) / 3 / len(obs_n)

        #update bandit
        if config['w_graph_type'] == 'bandit':
            feedback = episode_reward - prev_episode_reward
            for bandit_policy in bandit_policy_list:
                bandit_policy.update_dists(feedback)
            prev_episode_reward = episode_reward

        if config['c_net_type'] in ['transformer','GCN', 'hyper'] and avg_gate != []:
            c_logger.add_scalar(tag, np.mean(avg_gate), i_episode)
            g_logger.add_scalar(tag, gate_loss, i_episode)
            gate_open_rates.append(np.mean(avg_gate))
            gate_open_dist.append((torch.stack(avg_gate_dist).mean(dim=0)).tolist())


        if not config['fixed_lr']:
            agent.adjust_lr(i_episode)

        if (i_episode + 1) % config['eval_freq'] == 0:
            tr_log = {'num_adversary': 0,
                      'best_good_eval_reward': best_good_eval_reward,
                      'best_adversary_eval_reward': best_adversary_eval_reward,
                      'exp_save_dir': exp_save_dir, 'total_numsteps': total_numsteps,
                      'value_loss': value_loss.item(), 'policy_loss': policy_loss,
                      'i_episode': i_episode, 'start_time': start_time}
            copy_actor_policy(agent, eval_agent)
            eval_model_q(eval_agent, tr_log, config, i_episode, plot)
    utils.dict2csv(plot, os.path.join(tr_log['exp_save_dir'], 'train_curve.csv'))
    if config['c_net_type'] in ['transformer','GCN','hyper']:
        with open('./gate_open_rates.json', 'w') as fp:
            json.dump(gate_open_rates, fp)
        with open('./gate_open_dist.json', 'w') as fp:
            json.dump(gate_open_dist, fp)
    if config['w_graph_type'] == 'bandit':
        with open('./no_w_avg_rates.json', 'w') as fp:
            json.dump(no_w_avg_rates, fp)
        with open('./level_2_arms.json', 'w') as fp:
            json.dump(level_2_arms, fp)
    env.close()

