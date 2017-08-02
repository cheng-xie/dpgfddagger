import torch
import torch.optim as opt
from torch.autograd import Variable
from torch import FloatTensor as FT
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import agent
import numpy as np
from torch.utils.serialization import load_lua
import importlib.util
import random

#Default hyperparameter values
LEARNING_RATE_ACTOR = 0.01
BATCH_SIZE = 100

class MimicAgent():
    """An agent that mimics another agent

    Attributes:
        actor: The actor model that takes a state
        and returns a new action.
    """

    """
    @property
    def actor(self):
        return self.actor
    """

    def __init__(self,
            model_def,
            state_size = 1,
            action_size = 1,
            actor_alpha = LEARNING_RATE_ACTOR,
            batch_size = BATCH_SIZE,
            use_cuda = True):
        """Constructor for the DDPG_agent

        Args:
            buffer_size: size of the replay buffer

            alpha: The learning rate
        """
        self._use_cuda = use_cuda

        #initialize parameters
        self._actor_alpha = actor_alpha
        self._batch_size = batch_size
        self._state_size = state_size
        self._action_size = action_size

        # import the specified model_defs
        spec = importlib.util.spec_from_file_location("model_def", model_def)
        ModelDefModule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ModelDefModule)
        self.ActorModuleClass = ModelDefModule.Actor

        #initialize models
        self.load_models()

        #Initialize optimizers
        self._actor_optimizer = opt.Adam(self.actor.parameters(), lr=self._actor_alpha)

        # Use MSELoss for now
        self.criterion = nn.MSELoss()
        if self._use_cuda:
            self.criterion = self.criterion.cuda()

    def train_iters(self, states, actions, epochs, batch_size, iters_per_log = 1):
        states = torch.from_numpy(states.astype(np.float32))
        actions = torch.from_numpy(actions.astype(np.float32))
        dataset = TensorDataset((states), (actions))
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        tot_loss = 0
        log_iters_count = 0
        for epoch in range(epochs):
            for cur_iter, (state_batch, action_batch) in enumerate(self.dataloader):
                # Cuda
                if self._use_cuda:
                    state_batch = (state_batch.cuda())
                if self._use_cuda:
                    action_batch = (action_batch.cuda())
                print(type(state_batch))
                state_batch= Variable(state_batch)
                action_batch= Variable(action_batch)

                # Forward pass
                pred_actions = self.actor.forward(state_batch)

                # Calculate the loss
                loss = self.criterion(pred_actions, action_batch)

                # Zero out the gradients
                self._actor_optimizer.zero_grad()

                # Calculate some gradients
                loss.backward()

                # Run update step
                self._actor_optimizer.step()

                tot_loss += loss.data[0]
                log_iters_count += 1

                if log_iters_count % iters_per_log == 0:
                    print('Epoch {} Iter {}'.format(epoch, cur_iter))
                    print('Loss', tot_loss/iters_per_log)
                    tot_loss = 0

    def train_epochs(self, states, actions, epochs, batch_size, iters_per_log = 1):
        """Trains the agent for a bit.

            Args:
            Returns:
        """
        states = torch.from_numpy(states.astype(np.float32))
        actions = torch.from_numpy(actions.astype(np.float32))
        dataset = TensorDataset((states), (actions))
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        tot_loss = 0
        log_iters_count = 0
        for epoch in range(epochs):
            for cur_iter, (state_batch, action_batch) in enumerate(self.dataloader):
                # Cuda
                if self._use_cuda:
                    state_batch = (state_batch.cuda())
                if self._use_cuda:
                    action_batch = (action_batch.cuda())
                state_batch= Variable(state_batch)
                action_batch= Variable(action_batch)

                # Forward pass
                pred_actions = self.actor.forward(state_batch)

                # Calculate the loss
                loss = self.criterion(pred_actions, action_batch)

                # Zero out the gradients
                self._actor_optimizer.zero_grad()

                # Calculate some gradients
                loss.backward()

                # Run update step
                self._actor_optimizer.step()

                tot_loss += loss.data[0]
                log_iters_count += 1

                if log_iters_count % iters_per_log == 0:
                    print('Epoch {} Iter {}'.format(epoch, cur_iter))
                    print('Loss', tot_loss/iters_per_log)
                    tot_loss = 0

    def get_next_action(self,
            cur_state):
        """Get the next action from the agent.
            Args:
                cur_state: The current state of the enviroment
            Returns:
                The next action that the agent with the given
                agent_id will carry out given the current state
        """
        cur_action = None
        a = self.actor.forward(self.upcast(np.expand_dims(cur_state,axis=0)))
        cur_action = a.data.cpu().numpy()
        return cur_action

    def save_models(self, location=None):
        """Save the model to a given location

            Args:
                Location: where to save the model
            Returns:
                None
        """
        #Return all weights and buffers to the cpu
        self.actor.cpu()

        weight_dict = {'actor': self.actor.state_dict()}

        #Save both models
        torch.save(weight_dict, location)

    def load_models(self, location=None):
        # TODO: Make it actually do what it says
        #TODO: Remove hard coding of data
        """Loads the models from given location

            Args:
                Location: from where to load the model
            Returns:
                None
        """
        self.actor = self.ActorModuleClass(self._state_size,self._action_size)

        if location is not None:
            weight_dict = torch.load(location)
            self.actor.load_state_dict(weight_dict['actor'])

        #Move weights and bufffers to the gpu if possible
        if self._use_cuda:
            self.actor.cuda()

    def upcast(self, x):
        ''' Upcasts x to a torch Variable.
        '''
        #TODO: Where does this go?
        if self._use_cuda:
            return Variable(FT(x.astype(np.float32))).cuda()
        else:
            return Variable(FT(x.astype(np.float32)))

