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
LEARNING_RATE = 0.01
BATCH_SIZE = 100

class FDModel():
    """An agent that mimics another agent

    Attributes:
        fd: The fd model that takes a state
        and returns a new next_state.
    """

    """
    @property
    def fd(self):
        return self.fd
    """

    def __init__(self,
            model_def,
            action_size = 1,
            state_size = 1,
            alpha = LEARNING_RATE,
            batch_size = BATCH_SIZE,
            use_cuda = True):
        """Constructor for the DDPG_agent

        Args:
            buffer_size: size of the replay buffer

            alpha: The learning rate
        """
        self._use_cuda = use_cuda

        #initialize parameters
        self._fd_alpha = alpha
        self._batch_size = batch_size
        self._state_size = state_size
        self._action_size = state_size
        self._next_state_size = state_size

        # import the specified model_defs
        spec = importlib.util.spec_from_file_location("model_def", model_def)
        ModelDefModule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ModelDefModule)
        self.FDModuleClass = ModelDefModule.FD

        #initialize models
        self.load_models()

        #Initialize optimizers
        self._fd_optimizer = opt.Adam(self.fd.parameters(), lr=self._fd_alpha)

        # Use MSELoss for now
        self.criterion = nn.MSELoss()
        if self._use_cuda:
            self.criterion = self.criterion.cuda()

    '''
    def train_iters(self, states, actions, next_states, epochs, batch_size, iters_per_log = 1):
        state_actions = torch.from_numpy(np.concatenate((states.astype(np.float32), actions.astype(np.float32))))
        next_states = torch.from_numpy(next_states.astype(np.float32))

        dataset = TensorDataset((state_actions), (next_states))
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        tot_loss = 0
        log_iters_count = 0
        for epoch in range(epochs):
            for cur_iter, (state_action_batch, next_state_batch) in enumerate(self.dataloader):
                # Cuda
                if self._use_cuda:
                    state_batch = (state_batch.cuda())
                if self._use_cuda:
                    next_state_batch = (next_state_batch.cuda())
                print(type(state_batch))
                state_batch= Variable(state_batch)
                next_state_batch= Variable(next_state_batch)

                # Forward pass
                pred_next_states = self.fd.forward(state_batch)

                # Calculate the loss
                loss = self.criterion(pred_next_states, next_state_batch)

                # Zero out the gradients
                self._fd_optimizer.zero_grad()

                # Calculate some gradients
                loss.backward()

                # Run update step
                self._fd_optimizer.step()

                tot_loss += loss.data[0]
                log_iters_count += 1

                if log_iters_count % iters_per_log == 0:
                    print('Epoch {} Iter {}'.format(epoch, cur_iter))
                    print('Loss', tot_loss/iters_per_log)
                    tot_loss = 0
    '''
    
    def train_epochs(self, states, actions, next_states, epochs, batch_size, iters_per_log = 1):
        """Trains the agent for a bit.

            Args:
            Returns:
        """
        # concatenate states and actions in feature dimension
        state_actions = torch.from_numpy(np.concatenate((states.astype(np.float32), actions.astype(np.float32)), axis=-1))
        next_states = torch.from_numpy(next_states.astype(np.float32))
        dataset = TensorDataset((state_actions), (next_states))
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        tot_loss = 0
        log_iters_count = 0
        for epoch in range(epochs):
            for cur_iter, (state_action_batch, next_state_batch) in enumerate(self.dataloader):
                # Cuda
                if self._use_cuda:
                    state_action_batch = (state_action_batch.cuda())
                if self._use_cuda:
                    next_state_batch = (next_state_batch.cuda())
                state_action_batch= Variable(state_action_batch)
                next_state_batch= Variable(next_state_batch)

                # Forward pass
                pred_next_states = self.fd.forward(state_action_batch)

                # Calculate the loss
                loss = self.criterion(pred_next_states, next_state_batch)

                # Zero out the gradients
                self._fd_optimizer.zero_grad()

                # Calculate some gradients
                loss.backward()

                # Run update step
                self._fd_optimizer.step()

                tot_loss += loss.data[0]
                log_iters_count += 1

                if log_iters_count % iters_per_log == 0:
                    print('Epoch {} Iter {}'.format(epoch, cur_iter))
                    print('Loss', tot_loss/iters_per_log)
                    tot_loss = 0

    def get_next_next_state(self,
            cur_state, cur_action):
        """Get the next next_state from the agent.
            Args:
                cur_state: The current state of the enviroment
            Returns:
                The next next_state that the agent with the given
                agent_id will carry out given the current state
        """
        cur_state_actions = np.concatenate((states.astype(np.float32), actions.astype(np.float32)))
        cur_next_state = None
        a = self.fd.forward(self.upcast(np.expand_dims(cur_state_actions,axis=0)))
        cur_next_state = a.data.cpu().numpy()
        return cur_next_state

    def save_models(self, location=None):
        """Save the model to a given location

            Args:
                Location: where to save the model
            Returns:
                None
        """
        #Return all weights and buffers to the cpu
        self.fd.cpu()

        weight_dict = {'fd': self.fd.state_dict()}

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
        self.fd = self.FDModuleClass(self._state_size+self._action_size, self._next_state_size)

        if location is not None:
            weight_dict = torch.load(location)
            self.fd.load_state_dict(weight_dict['fd'])

        #Move weights and bufffers to the gpu if possible
        if self._use_cuda:
            self.fd.cuda()

    def upcast(self, x):
        ''' Upcasts x to a torch Variable.
        '''
        #TODO: Where does this go?
        if self._use_cuda:
            return Variable(FT(x.astype(np.float32))).cuda()
        else:
            return Variable(FT(x.astype(np.float32)))

