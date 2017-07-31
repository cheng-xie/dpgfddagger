import torch
import torch.optim as opt
from torch.autograd import Variable
from torch import FloatTensor as FT
import agent
from replay_buffer import ExperienceReplay
import numpy as np
from torch.utils.serialization import load_lua
import importlib.util
# import model_defs.ddpg_models.mountain_cart.actor as actor
# import model_defs.ddpg_models.mountain_cart.critic as critic
import random

#Default hyperparameter values
LEARNING_RATE_CRITIC = 0.01
LEARNING_RATE_ACTOR = 0.01
BATCH_SIZE = 100

class MimicAgent():
    """An agent that mimics another agent


    The agent stores a replay buffer along with
    two models of the data, an actor and a critic.

    Attributes:
        auxiliary_losses: The list of enabled
        auxiliary rewards for this agent

        actor: The actor model that takes a state
        and returns a new action.

        critic: The critic model that takes a state
        and an action and returns the expected
        reward

        replay_buffer: The DDPGAgent replay buffer
    """

    """
    @property
    def actor(self):
        return self.actor

    @property
    def critic(self):
        return self.critic

    @property
    def replay_buffer(self):
        return self.replay_buffer
    """

    def __init__(self,
            model_def,
            state_size = 1,
            action_size = 1,
            buffer_size = REPLAY_BUFFER_SIZE,
            actor_alpha = LEARNING_RATE_ACTOR,
            actor_iter_count = ACTOR_ITER_COUNT,
            batch_size = BATCH_SIZE,
            use_cuda = True):
        """Constructor for the DDPG_agent

        Args:
            buffer_size: size of the replay buffer

            alpha: The learning rate

            gamma: The discount factor

        Returns:
            A DDPGAgent object
        """
        self._use_cuda = use_cuda

        #initialize parameters
        self._actor_alpha = actor_alpha
        self._actor_iter_count = actor_iter_count
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

    def train_epochs(self, states, actions, epochs, batch_size, iters_per_log = 1):
        """Trains the agent for a bit.

            Args:
                None
            Returns:
                None
        """
        dataset = TensorDataset(FT(states), FT(actions))
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        tot_loss = 0
        log_iters_count = 0
        for epoch in range(epochs):
            for cur_iter, (state_batch, action_batch) in enumerate(self.dataloader):
                # Zero out the gradients
                self.actor.zero_grad()

                # Cuda
                if self.use_cuda:
                    state_batch = state_batch.cuda()
                if self.use_cuda:
                    action_batch = action_batch.cuda()

                # Forward pass
                pred_actions = self.actor.forward(state_batch)

                # Calculate the loss
                loss = self.criterion(recon, target)

                # Calculate some gradients
                loss.backward()

                # Run update step
                self.enc_optim.step()
                self.dec_optim.step()

                tot_loss += loss.data[0]
                log_iters_count += 1

                if log_iters_count % iters_per_log == 0:
                    print('Epoch {} Iter {}'.format(epoch, cur_iter))
                    print('Loss', tot_loss/iters_per_log)
                    tot_loss = 0

        #update_actor
        for i in range(self._actor_iter_count):
            s_t, a_t, r_t, s_t1, done = self.replay_buffer.batch_sample(self._batch_size)
            done = self.upcast(done)
            s_t = self.upcast(s_t)
            a_t = self.upcast(a_t)
            s_t1 = self.upcast(s_t1)
            r_t = self.upcast(r_t)
            a_t1,aux_actions = self.actor.forward(s_t1,self.auxiliary_losses.keys())
            expected_reward = self.critic.forward(s_t1,a_t1)

            total_loss = -1*expected_reward
            for key,aux_reward_tuple in self.auxiliary_losses.items():
                aux_weight,aux_module = aux_reward_tuple
                total_loss += aux_weight*aux_module(aux_actions[key],s_t,a_t,r_t,s_t1,a_t1)

            mean_loss = torch.mean(total_loss)

            #print('LOSS:', mean_loss, 'Eps', self.epsilon)
            #preform one optimization update
            self._actor_optimizer.zero_grad()
            mean_loss.backward()
            self._actor_optimizer.step()

    def get_next_action(self,
            cur_state,
            agent_id=None):
        """Get the next action from the agent.

            Takes a state,reward and possibly auxiliary reward
            tuple and returns the next action from the agent.

            Args:
                cur_state: The current state of the enviroment
                prev_reward: The previous reward from the enviroment
                is_done: Signals if a given episode is done.
                is_test: Check to see if the agent is done
                agent_id=None
            Returns:
                The next action that the agent with the given
                agent_id will carry out given the current state
        """
        cur_action = None
        a, _ = self.actor.forward(self.upcast(np.expand_dims(cur_state,axis=0)),[])
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

