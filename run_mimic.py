from environment import GymEnvironment
from mimic_agent import MimicAgent
from experts.lunar_lander_heuristic import LunarLanderExpert
from experts.bipedal_walker_heuristic import BipedalWalkerExpert

import numpy as np
import sys
import random
import os
import json
import pdb
import argparse

def main(args):
    use_cuda = args.cuda
    is_test = args.test

    model_path = args.model_path
    conf_path = os.path.join(model_path, 'config.json')
    json_data = open(conf_path).read()
    conf = json.loads(json_data)

    # update relative path to model_def module
    conf['agent']['model_def'] = os.path.join(model_path, conf['agent']['model_def'])

    run = Runner(conf['env'], conf['agent'], use_cuda)

    # load_path = args.load_path
    if is_test:
        # conf['test']['load_path'] = load_path
        run.test_mimic(conf['test'])
    else:
        save_path = args.save_path
        conf['train']['save_path'] = save_path
        # conf['train']['load_path'] = load_path
        rews = []
        for i in range(30):
            # states, actions, _, _ = run.sample_expert(2000)
            # run.train_mimic(states, actions, conf['train'], num_epochs=200)
            run.train_dagger(conf['train'], num_tuples=2000, num_epochs=200)
            rew = run.test_mimic(conf['test'])
            rews.append(rew)
            run = Runner(conf['env'], conf['agent'], use_cuda)

        print(rews)
        rews = np.asarray(rews)
        print(np.mean(rews))
        print(np.std(rews))

class Runner:
    def __init__(self, env_config, agent_config, use_cuda = True):
        self.env = GymEnvironment(name = env_config["name"])
        self.action_size = self.env.action_size[0]
        self.state_size = self.env.obs_size[0]
        # initialize mimic agent
        self.agent = MimicAgent(action_size = self.action_size,
                                state_size = self.state_size,
                                **agent_config, use_cuda = use_cuda)

        # if train_config.get('load_path')
            # self.agent.load_models(train_config.get('load_path'))

        # initialize expert
        # self.expert = LunarLanderExpert()
        self.expert = LunarLanderExpert()

    def sample_expert(self, num_tuples, do_render = False):
        '''
            Accumulates experience tuples from the expert for num tuples.
            Returns states, action, rewards and done flags as np arrays.
        '''
        state_size = self.state_size
        action_size = self.action_size
        capacity = num_tuples

        actions = np.empty((capacity, action_size), dtype = np.float16)
        states = np.empty((capacity, state_size), dtype = np.float16)
        rewards = np.empty(capacity, dtype = np.float16)
        dones = np.empty(capacity, dtype = np.bool)

        self.env.new_episode()

        for i in range(num_tuples):
            print('{} / {}'.format(i+1, num_tuples))
            cur_obs = self.env.cur_obs
            cur_action = self.expert.get_next_action(cur_obs)

            next_state, reward, done = self.env.next_obs(cur_action, render = (i % 16 == 0) and do_render)

            actions[i] = cur_action
            states[i] = cur_obs
            rewards[i] = reward
            dones[i] = done

        print('Ave expert reward: ', np.mean(rewards))
        return states, actions, rewards, dones

    def train_mimic(self, states, actions, train_config, num_epochs = 4000):
        # Load model
        #train_config.get('num_epochs')
        self.agent.train_epochs(states, actions, num_epochs, states.shape[0])

    def train_dagger(self, train_config, num_tuples, num_epochs, do_render = False):
        # Load model
        #train_config.get('num_epochs')
        state_size = self.state_size
        action_size = self.action_size
        capacity = num_tuples

        actions = np.empty((capacity, action_size), dtype = np.float16)
        states = np.empty((capacity, state_size), dtype = np.float16)
        rewards = np.empty(capacity, dtype = np.float16)
        dones = np.empty(capacity, dtype = np.bool)

        self.env.new_episode()

        beta = 1.0
        tuples_per_epoch = int(num_tuples/num_epochs)
        epochs = 0
        for i in range(num_tuples):
            print('{} / {}'.format(i+1, num_tuples))
            cur_obs = self.env.cur_obs
            cur_action = None
            expert_action = None
            if beta > np.random.rand():
                cur_action = self.expert.get_next_action(cur_obs)
                expert_action = cur_action
            else:
                expert_action = self.expert.get_next_action(cur_obs)
                cur_action = np.squeeze(self.agent.get_next_action(cur_obs), axis=0)
                cur_action = np.clip(cur_action, -1, 1)

            next_state, reward, done = self.env.next_obs(cur_action, render = (i % 16 == 0)and do_render)

            actions[i] = expert_action
            states[i] = cur_obs
            rewards[i] = reward
            dones[i] = done

            beta = 1.0-float(i)/num_tuples

            if ((i+1)%tuples_per_epoch) == 0:
                self.agent.train_epochs(states[:i], actions[:i], 1, 500)
                epochs += 1

        # print('Ave expert reward: ', np.mean(rewards))


    def test_mimic(self, test_config, do_render = False):
        test_steps = test_config['steps']

        self.env.new_episode()

        tot_reward = 0
        for step in range(test_steps):
            cur_obs = self.env.cur_obs
            cur_action = np.squeeze(self.agent.get_next_action(cur_obs), axis=0)
            cur_action = np.clip(cur_action, -1, 1)
            _, reward, _ = self.env.next_obs(cur_action, render = step%16==0 and do_render)
            tot_reward += reward

        print('Ave test reward: {}'.format(tot_reward/test_steps))
        return tot_reward/test_steps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch DDPG')
    # parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
    # parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 2)')
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    # parser.add_argument('--coach-model', type=str, required=True, metavar='cm', help='path to folder where coach model is defined')
    # parser.add_argument('--coach-weights', type=str, required=True, metavar='cw', help='path to coach weights file')
    parser.add_argument('--model-path', type=str, required=True, metavar='m', help='path to folder where model is defined')
    parser.add_argument('--save-path', type=str, metavar='s', help='filename where mimic weights should be saved to')
    # parser.add_argument('--load-path', type=str, default=None, metavar='l', help='filename where mimic weights should be loaded from')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--test', action='store_true', default=False, help='run a test')
    args = parser.parse_args()
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    assert(os.path.isdir(args.model_path))
    main(args)
