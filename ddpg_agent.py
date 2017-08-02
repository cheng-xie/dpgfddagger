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
REPLAY_BUFFER_SIZE = 1000000
DISCOUNT_FACTOR = 1
LEARNING_RATE_CRITIC = 0.01
LEARNING_RATE_ACTOR = 0.01
ACTOR_ITER_COUNT = 1000
CRITIC_ITER_COUNT = 1000
BATCH_SIZE = 100
EPSILON = 0.01
FREEZE_TARGET_STEPS = 1

class DDPGAgent(agent.Agent):
    """An agent that implements the DDPG algorithm

    An agent that implements the deep deterministic
    policy gradient algorithm for continuous control.
    A description of the algorithm can be found at
    https://arxiv.org/pdf/1509.02971.pdf.

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
            gamma = DISCOUNT_FACTOR,
            actor_alpha = LEARNING_RATE_ACTOR,
            critic_alpha = LEARNING_RATE_CRITIC,
            actor_iter_count = ACTOR_ITER_COUNT,
            critic_iter_count = CRITIC_ITER_COUNT,
            freeze_target_steps = FREEZE_TARGET_STEPS,
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

        #Initialize experience replay buffer
        self.replay_buffer = ExperienceReplay(state_size, action_size, buffer_size)
        #TODO

        #initialize parameters
        self.epsilon = 0.35
        self._actor_alpha = actor_alpha
        self._critic_alpha = critic_alpha
        self._actor_iter_count = actor_iter_count
        self._critic_iter_count = critic_iter_count
        self._freeze_target_steps = freeze_target_steps
        self._freeze_target_step = 0
        self._gamma = gamma
        self._batch_size = batch_size
        self._state_size = state_size
        self._action_size = action_size

        # import the specified model_defs
        spec = importlib.util.spec_from_file_location("model_def", model_def)
        ModelDefModule = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ModelDefModule)
        self.ActorModuleClass = ModelDefModule.Actor
        self.CriticModuleClass = ModelDefModule.Critic

        #initialize models
        self.load_models()

        #Move weights and bufffers to the gpu if possible
        if self._use_cuda:
            self.actor = self.actor.cuda()
            self.critic = self.critic.cuda()
            self._target_critic = self._target_critic.cuda()

        #Initialize optimizers
        self._actor_optimizer = opt.Adam(self.actor.parameters(), lr=self._actor_alpha)
        self._critic_optimizer = opt.Adam(self.critic.parameters(), lr=self._critic_alpha)

        # if self._use_cuda:
        #     self._actor_optimizer = self._actor_optimizer.cuda()
        #     self._critic_optimizer = self._critic_optimizer.cuda()

    def train(self):
        """Trains the agent for a bit.

            Args:
                None
            Returns:
                None
        """
        self.epsilon = self.epsilon * 0.99995
        #update_critic
        for i in range(self._critic_iter_count):
            s_t, a_t, r_t, s_t1, done = self.replay_buffer.batch_sample(self._batch_size)
            done = self.upcast(done)
            s_t = self.upcast(s_t)
            a_t = self.upcast(a_t)
            s_t1 = self.upcast(s_t1)
            r_t = self.upcast(r_t)
            a_t1 = self.actor.forward(s_t1)
            critic_target = r_t + self._gamma*(1-done)*self._target_critic.forward(s_t1,a_t1)
            td_error = (self.critic.forward(s_t,a_t)-critic_target)**2

            #preform one optimization update
            self._critic_optimizer.zero_grad()
            mean_td_error = torch.mean(td_error)
            mean_td_error.backward()
            self._critic_optimizer.step()


        #update_actor
        for i in range(self._actor_iter_count):
            s_t, a_t, r_t, s_t1, done = self.replay_buffer.batch_sample(self._batch_size)
            done = self.upcast(done)
            s_t = self.upcast(s_t)
            a_t = self.upcast(a_t)
            s_t1 = self.upcast(s_t1)
            r_t = self.upcast(r_t)
            a_t1 = self.actor.forward(s_t1)
            expected_reward = self.critic.forward(s_t1,a_t1)

            total_loss = -1*expected_reward
            mean_loss = torch.mean(total_loss)

            #print('LOSS:', mean_loss, 'Eps', self.epsilon)
            #preform one optimization update
            self._actor_optimizer.zero_grad()
            mean_loss.backward()
            self._actor_optimizer.step()

        # TODO: Freeze less often
        self._freeze_target_step += 1
        if (self._freeze_target_step >= self._freeze_target_steps):
            self._target_critic.load_state_dict(self.critic.state_dict())
            self._freeze_target_step = 0



    def get_next_action(self,
            cur_state,
            agent_id=None,
            is_test=False):
        """Get the next action from the agent.

            Takes a state,reward and possibly auxiliary reward
            tuple and returns the next action from the agent.
            The agent may cache the reward and state

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
        cur_state_up = self.upcast(np.expand_dims(cur_state,axis=0))
        if is_test:
            a = self.actor.forward(cur_state_up)
            cur_action = a.data.cpu().numpy()
        elif random.random() < self.epsilon:
            a = self.actor.forward(cur_state_up)
            cur_action = a.data.cpu().numpy()
            cur_action += 0.3*np.expand_dims(np.random.randn(self._action_size),axis=0)
            self.replay_buffer.put_act(cur_state,cur_action)
        else:
            a = self.actor.forward(cur_state_up)
            cur_action = a.data.cpu().numpy()
            self.replay_buffer.put_act(cur_state,cur_action)
        return cur_action

    def log_reward(self,reward,is_done):
            self.replay_buffer.put_rew(reward,is_done)

    def save_models(self, location=None):
        """Save the model to a given location

            Args:
                Location: where to save the model
            Returns:
                None
        """
        #Return all weights and buffers to the cpu
        self.actor.cpu()
        self.critic.cpu()

        weight_dict = {'actor': self.actor.state_dict(), 'critic': self.critic.state_dict()}
        #Save both models
        torch.save(weight_dict, location)

        #Move weights and bufffers to the gpu if possible
        if self._use_cuda:
            self.actor = self.actor.cuda()
            self.critic = self.critic.cuda()
            self._target_critic = self._target_critic.cuda()


    def load_models(self, location=None):
        # TODO: Make it actually do what it says
        #TODO: Remove hard coding of data
        """Loads the models from given location

            Args:
                Location: from where to load the model
            Returns:
                None
        """
        self.actor = self.ActorModuleClass(self._state_size,self._action_size) #dill.load(actor_file)
        self.critic = self.CriticModuleClass(self._state_size + self._action_size, 1)#dill.load(critic_file)
        self._target_critic = self.CriticModuleClass(self._state_size + self._action_size,1)#dill.load(critic_file)

        if location is not None:
            weight_dict = torch.load(location)
            self.actor.load_state_dict(weight_dict['actor'])
            self.critic.load_state_dict(weight_dict['critic'])

    def upcast(self, x):
        ''' Upcasts x to a torch Variable.
        '''
        #TODO: Where does this go?
        if self._use_cuda:
            return Variable(FT(x.astype(np.float32))).cuda()
        else:
            return Variable(FT(x.astype(np.float32)))

