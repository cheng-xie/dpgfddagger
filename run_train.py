from environment import GymEnvironment

from ddpg_agent import DDPGAgent
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

    load_path = args.load_path
    if is_test:
        conf['test']['load_path'] = load_path
        run.test(conf['test'])
    else:
        save_path = args.save_path
        conf['train']['save_path'] = save_path
        conf['train']['load_path'] = load_path
        run.train(conf['train'])

class Runner:
    def __init__(self, env_config, agent_config, use_cuda = True):
        self.env = GymEnvironment(name = env_config["name"])
        self.agent = DDPGAgent(action_size = self.env.action_size[0],
                                state_size = self.env.obs_size[0],
                                **agent_config, use_cuda = use_cuda)

    def train(self, train_config):
        # Load model
        if train_config.get('load_path'):
            self.agent.load_models(train_config.get('load_path'))

        # Fill experience replay
        self.env.new_episode()
        ma_reward = 0

        prefill = train_config['prefill']
        if prefill > 0:
            temp_reward = 0
            temp_done = False
            for step in range(prefill):
                cur_obs = self.env.cur_obs
                _ = self.agent.get_next_action(cur_obs)
                cur_action = np.asarray([random.random()*2.0-1.0]*self.env.action_size[0])
                next_state, reward, done = self.env.next_obs(cur_action, render = (step % 8 == 0))

                temp_reward = reward
                temp_done = done
                self.agent.log_reward(temp_reward, temp_done)
                ma_reward = ma_reward*0.99 + reward*0.01

        # Start training
        train_steps = train_config['steps']

        temp_reward = 0
        temp_done = True
        for step in range(train_steps):
            cur_obs = self.env.cur_obs
            # TODO: This step probably belongs somewhere else
            cur_action = np.squeeze(self.agent.get_next_action(cur_obs), axis=0)
            if (any(np.isnan(cur_obs))):
                pdb.set_trace()
            next_state, reward, done = self.env.next_obs(cur_action, render = (step % 8 == 0))

            temp_reward = reward
            temp_done = done

            self.agent.log_reward(temp_reward, temp_done)

            self.agent.train()
            ma_reward = ma_reward*0.995 + reward*0.005
            if(step % 500 == 0):
                print(cur_obs, ' ', cur_action, 'Reward:', ma_reward)
                print('Eps',self.agent.epsilon)
            if(step % 5000 == 0):
                print('Saving weights')
                self.agent.save_models(train_config['save_path'])


    def test(self, test_config):
        if test_config.get('load_path'):
            self.agent.load_models(test_config.get('load_path'))
        else:
            print('Warning: did not parse load path. Running random init model')

        test_steps = test_config['steps']

        self.env.new_episode()

        temp_reward = 0
        temp_done = False
        for step in range(test_steps):
            cur_obs = self.env.cur_obs
            cur_action = np.squeeze(self.agent.get_next_action(cur_obs, is_test = True), axis=0)
            cur_action = np.clip(cur_action, -1, 1)
            next_state, reward, done = self.env.next_obs(cur_action, render = True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch DDPG')
    # parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 64)')
    # parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 2)')
    # parser.add_argument('--no-cuda', action='store_true', default=False, help='enables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N', help='how many batches to wait before logging training status')
    parser.add_argument('--model-path', type=str, required=True, metavar='m', help='path to folder where model is defined')
    parser.add_argument('--save-path', type=str, metavar='s', help='filename where weights should be saved to')
    parser.add_argument('--load-path', type=str, default=None, metavar='l', help='filename where weights should be loaded from')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--test', action='store_true', default=False, help='enables CUDA training')
    args = parser.parse_args()
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    assert(os.path.isdir(args.model_path))
    main(args)
